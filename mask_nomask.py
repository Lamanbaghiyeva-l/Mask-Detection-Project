import os
import cv2
import json
import random

annotation_path = "dataset/Medical mask/Medical mask/Medical Mask/annotations"
image_dir = "dataset/Medical mask/Medical mask/Medical Mask/images"

TARGET_PER_CLASS = 114

def getJSON(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def safe_crop_xyxy(img, bbox):
    # bbox = [x1, y1, x2, y2]
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    face = img[y1:y2, x1:x2]
    if face is None or face.size == 0:
        return None
    return face

jsonfiles = []
for fn in os.listdir(annotation_path):
    if fn.endswith(".json"):
        jsonfiles.append(getJSON(os.path.join(annotation_path, fn)))

random.shuffle(jsonfiles)

os.makedirs("out/with_mask", exist_ok=True)
os.makedirs("out/with_no_mask", exist_ok=True)

mask_count = 0
nomask_count = 0

for jf in jsonfiles:
    if mask_count >= TARGET_PER_CLASS and nomask_count >= TARGET_PER_CLASS:
        break

    filename = jf.get("FileName")
    if not filename:
        continue

    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    for ann in jf.get("Annotations", []):
        cls = ann.get("classname")
        bbox = ann.get("BoundingBox")
        if not bbox or len(bbox) != 4:
            continue

        if cls == "face_with_mask" and mask_count < TARGET_PER_CLASS:
            face = safe_crop_xyxy(img, bbox)
            if face is None:
                continue
            mask_count += 1
            out_path = f"out/with_mask/mask_{mask_count:04d}.png"
            cv2.imwrite(out_path, face)

        elif cls == "face_no_mask" and nomask_count < TARGET_PER_CLASS:
            face = safe_crop_xyxy(img, bbox)
            if face is None:
                continue
            nomask_count += 1
            out_path = f"out/with_no_mask/nomask_{nomask_count:04d}.png"
            cv2.imwrite(out_path, face)

        if mask_count >= TARGET_PER_CLASS and nomask_count >= TARGET_PER_CLASS:
            break

print("DONE")
print("with_mask:", mask_count)
print("with_no_mask:", nomask_count)
