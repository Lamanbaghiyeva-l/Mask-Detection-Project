"""
Memory Efficient Pipeline
Qwen → SAM3 → trimap → CLIP + ViTMatte
"""

import os, sys, argparse, random
import numpy as np
import cv2
import torch
import torch.nn.functional as TF
from PIL import Image
from os.path import join as opj
from torchvision.transforms.functional import to_pil_image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

VITMATTE_ROOT = "/mnt/pool1/sharehome/laman/vitmatte1/vitmatte1"
SAM3_ROOT     = "/mnt/pool1/sharehome/laman/vlm_investigate_sam3"

if VITMATTE_ROOT not in sys.path:
    sys.path.insert(0, VITMATTE_ROOT)

if SAM3_ROOT not in sys.path:
    sys.path.insert(0, SAM3_ROOT)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from transformers import (
    CLIPModel,
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from torchvision.transforms import functional as F
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer


# ====================================================
# QWEN
# ====================================================

def init_qwen(model_dir):
    print("🔄 Loading Qwen...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="cpu",
        torch_dtype=torch.float32,
        local_files_only=True
    )

    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=False
    )

    model.eval()
    print("✅ Qwen ready")
    return model, processor


def run_qwen(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((384, 384))

    question = (
        "Describe this image in one detailed paragraph for image matting. "
        "Mention the foreground object, background, hair, transparency and boundaries. "
        "Do NOT use markdown, bullet points, or section titles; only plain continuous text."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    }]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    caption = processor.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return caption


# ====================================================
# SAM3  –  trimap helpers
# ====================================================

def load_sam3(checkpoint: str, device: str):
    """
    Build a SAM3 image model and its processor.
    Returns (model, processor).
    """
    print("🔄 Loading SAM3...")
    model = build_sam3_image_model(checkpoint=checkpoint)
    model = model.to(device).eval()
    processor = Sam3Processor()
    print("✅ SAM3 ready")
    return model, processor


def sam3_get_mask(sam_tuple, img_pil: Image.Image, caption: str) -> np.ndarray:
    """
    Run SAM3 on *img_pil* guided by *caption*.
    Uses a centre-point foreground prompt plus the text caption.
    Returns a uint8 numpy mask (H×W, values 0 or 255).
    """
    model, processor = sam_tuple
    img_rgb = img_pil.convert("RGB")
    W, H = img_rgb.size

    point_coords = np.array([[W // 2, H // 2]])   # centre point
    point_labels = np.array([1])                   # foreground

    inputs = processor(
        images=img_rgb,
        text=caption,
        point_coords=point_coords,
        point_labels=point_labels,
        return_tensors="pt"
    )
    # Move tensors to the same device as the model
    inputs = {
        k: v.to(next(model.parameters()).device) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.pred_masks: (1, num_masks, H, W) or (num_masks, H, W)
    masks = outputs.pred_masks.squeeze(0).cpu().numpy()   # (num_masks, H, W)

    if masks.ndim == 3:
        # Pick the mask with the highest IoU score when available
        if hasattr(outputs, "iou_scores"):
            scores = outputs.iou_scores.squeeze().cpu().numpy()
        else:
            scores = np.array([m.sum() for m in masks])
        mask = masks[int(np.argmax(scores))]
    else:
        mask = masks

    return (mask > 0.5).astype(np.uint8) * 255


def mask_to_trimap(mask: np.ndarray,
                   kmin: int = 1, kmax: int = 30,
                   fixed_k: int = -1) -> np.ndarray:
    """
    Convert a binary mask (uint8, 0/255) to a trimap:
      0   → definite background
      128 → unknown / transition region
      255 → definite foreground

    The width of the unknown band is controlled by the kernel size *kk*,
    which is chosen randomly from [kmin, kmax] unless *fixed_k* > 0.
    """
    H, W = mask.shape[:2]

    def _make_odd(k: int) -> int:
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1

    if fixed_k is not None and int(fixed_k) > 0:
        kk = _make_odd(fixed_k)
    else:
        kk = _make_odd(random.randint(kmin, kmax))

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    fg_eroded  = cv2.erode(mask,  kernel, iterations=1)   # sure foreground
    fg_dilated = cv2.dilate(mask, kernel, iterations=1)   # outer boundary

    trimap = np.full((H, W), 128, dtype=np.uint8)         # unknown = 128
    trimap[fg_dilated == 0]   = 0                          # sure background
    trimap[fg_eroded  == 255] = 255                        # sure foreground

    return trimap


# ====================================================
# SAM2  (kept for reference / fallback)
# ====================================================

def init_sam2(checkpoint: str = "facebook/sam2-hiera-large", device: str = "cuda"):
    predictor = SAM2ImagePredictor.from_pretrained(checkpoint, device=device)
    return predictor


def get_trimap_from_sam2(predictor, comp_pil: Image.Image,
                          kmin: int = 1, kmax: int = 30,
                          fixed_k: int = -1) -> Image.Image:
    """
    Runs SAM2 with a centre-point prompt, picks the best mask,
    then delegates to mask_to_trimap().  Returns a PIL Image.
    """
    img_np = np.array(comp_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    predictor.set_image(img_np)

    point_coords = np.array([[W // 2, H // 2]])
    point_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8) * 255

    trimap = mask_to_trimap(mask, kmin=kmin, kmax=kmax, fixed_k=fixed_k)
    return Image.fromarray(trimap)


# ====================================================
# CLIP + VITMATTE
# ====================================================

def load_clip(model_dir, device):
    model = CLIPModel.from_pretrained(
        model_dir,
        local_files_only=True
    ).to(device).eval()

    processor = CLIPProcessor.from_pretrained(
        model_dir,
        local_files_only=True
    )

    return model, processor


def load_vitmatte(model_name, checkpoint, joint_ckpt, device):
    cfg = LazyConfig.load(
        opj(VITMATTE_ROOT, "configs/common/model.py")
    )

    model = instantiate(cfg.model)
    model.to(device).eval()

    DetectionCheckpointer(model).load(checkpoint)
    if joint_ckpt and os.path.exists(joint_ckpt):
        ckpt = torch.load(joint_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    print("✅ ViTMatte ready")
    return model


def get_clip_embed(model, proc, img, text):
    inp = proc(
        text=[text],
        images=[img],
        return_tensors="pt",
        padding=True
    )

    inp = {k: v.to(model.device) for k, v in inp.items()}

    with torch.no_grad():
        out = model(**inp)
        ie = out.image_embeds
        te = out.text_embeds
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)

    return torch.cat([ie, te], dim=-1)


@torch.no_grad()
def vitmatte_infer(model, image_t, trimap_t, clip_emb, device):
    import torch.nn.functional as F

    B, C, H, W = image_t.shape
    MAX_SIZE = 512

    scale = MAX_SIZE / max(H, W)

    if scale < 1:
        new_h, new_w = int(H * scale), int(W * scale)

        image_t = F.interpolate(
            image_t,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        trimap_t = F.interpolate(
            trimap_t,
            size=(new_h, new_w),
            mode="nearest"
        )

    batch = {
        "image":      image_t.to(device),
        "trimap":     trimap_t.to(device),
        "clip_embed": clip_emb.to(device)
    }

    out = model(batch)["phas"].clamp(0, 1)

    if scale < 1:
        out = F.interpolate(
            out,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

    return to_pil_image(out.squeeze().cpu())


# ====================================================
# MAIN PIPELINE
# ====================================================

def main(args):
    device = args.device
    split  = "Test" if args.split.lower() == "test" else "Train"
    fg_dir = opj(args.data_root, split, "FG")
    lst    = opj(args.data_root, split, "fg_test.txt")

    with open(lst) as f:
        names = [x.strip() for x in f]

    alpha_dir  = opj(args.output_dir, "alpha_pred")
    trimap_dir = opj(args.output_dir, "trimap_used")
    cap_dir    = opj(args.output_dir, "captions")

    os.makedirs(alpha_dir,  exist_ok=True)
    os.makedirs(trimap_dir, exist_ok=True)
    os.makedirs(cap_dir,    exist_ok=True)

    # =========================
    # STAGE 1 — QWEN
    # =========================

    qwen, qwen_proc = init_qwen(args.qwen_model_dir)
    captions = {}

    for name in names:
        path    = opj(fg_dir, name)
        caption = run_qwen(qwen, qwen_proc, path)
        captions[name] = caption
        with open(opj(cap_dir, name + ".txt"), "w") as f:
            f.write(caption)

    del qwen, qwen_proc
    torch.cuda.empty_cache()

    # =========================
    # STAGE 2 — SAM3 → trimap
    # =========================

    sam = load_sam3(args.sam_checkpoint, device)
    trimaps = {}

    for name in names:
        path = opj(fg_dir, name)
        img  = Image.open(path).convert("RGB")

        # Get binary mask via SAM3 (text + centre-point guided)
        mask   = sam3_get_mask(sam, img, captions[name])

        # Convert mask → 3-class trimap (numpy uint8: 0/128/255)
        trimap = mask_to_trimap(mask)

        trimaps[name] = trimap                              # store as numpy
        Image.fromarray(trimap).save(opj(trimap_dir, name))

    del sam
    torch.cuda.empty_cache()

    # =========================
    # STAGE 3 — CLIP + VITMATTE
    # =========================

    vit              = load_vitmatte(args.vit_model, args.vit_checkpoint,
                                     args.joint_checkpoint, device)
    clip, clip_proc  = load_clip(args.clip_model_dir, device)

    for name in names:
        img_path = opj(fg_dir, name)
        img      = Image.open(img_path).convert("RGB")
        trimap   = trimaps[name]                            # numpy uint8

        clip_emb = get_clip_embed(clip, clip_proc, img, captions[name])

        image_t  = F.to_tensor(img).unsqueeze(0)
        trimap_t = F.to_tensor(Image.fromarray(trimap)).unsqueeze(0)

        alpha = vitmatte_infer(vit, image_t, trimap_t, clip_emb, device)
        alpha.save(opj(alpha_dir, name))

    print("✅ DONE")


# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root",        required=True)
    parser.add_argument("--split",            default="Test")
    parser.add_argument("--output-dir",       required=True)

    parser.add_argument("--sam-checkpoint",   required=True)
    parser.add_argument("--vit-checkpoint",   required=True)
    parser.add_argument("--joint-checkpoint", default="")

    parser.add_argument("--vit-model",        default="vitmatte-s")

    parser.add_argument("--clip-model-dir",   required=True)
    parser.add_argument("--qwen-model-dir",   required=True)

    parser.add_argument("--device",           default="cuda")

    args = parser.parse_args()
    main(args)


# cd /mnt/pool1/sharehome/laman/vitmatte1/vitmatte1
#
# TOKENIZERS_PARALLELISM=false python /mnt/pool1/sharehome/laman/vlm_investigate/pipline/sam3_trimap.py \
#   --data-root   /mnt/pool1/sharehome/laman/vlm_investigate/data/Distinctions-646 \
#   --split Test \
#   --output-dir  /mnt/pool1/sharehome/laman/vlm_investigate/experiments/qwen_sam3_vitmatte \
#   --sam-checkpoint /mnt/pool1/sharehome/laman/vlm_investigate_sam3/checkpoints/sam3.pt \
#   --vit-checkpoint ViTMatte_S_DIS.pth \
#   --joint-checkpoint /mnt/pool1/sharehome/laman/vlm_investigate/checkpoints/joint_finetune/clip_proj_best.pth \
#   --clip-model-dir /mnt/pool1/sharehome/laman/vlm_investigate/models/CLIP_HF \
#   --qwen-model-dir /mnt/pool1/sharehome/laman/vlm_investigate/models/Qwen2.5-VL-7B-Instruct \
#   --device cuda
