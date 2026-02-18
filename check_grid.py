import os, random, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sample(folder, n):
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    random.shuffle(files)
    return files[:n]

mask = sample("out/with_mask", 10)
nomask = sample("out/with_no_mask", 10)
all_imgs = mask + nomask

plt.figure(figsize=(14, 4))
for i, p in enumerate(all_imgs):
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 10, i+1)
    plt.imshow(img)
    plt.axis("off")

plt.tight_layout()
plt.savefig("out/sample_grid.png", bbox_inches="tight")
print("Saved: out/sample_grid.png")
