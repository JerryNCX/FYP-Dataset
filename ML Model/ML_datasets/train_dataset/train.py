import os
import random
import shutil
from pathlib import Path
from PIL import Image

# ====== CONFIG ======
SOURCE_DIR = Path("./split_dataset")
OUTPUT_DIR = Path("./yolo_dataset")
TRAIN_SPLIT = 0.8

# Class mapping (folder_name → class_id)
CLASS_MAP = {
    "cpu": 0,
    "video-card": 1,
    "motherboard": 2,
    "memory": 3,
    "internal-hard-drive": 4,
    "power-supply": 5,
    "case": 6,
    "cpu-cooler": 7
}
# ====================

# Create final folders
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

def create_yolo_label(img_path, label_path, class_id):
    """Creates YOLO txt: full-image bounding box"""
    img = Image.open(img_path)
    w, h = img.size

    # full image bounding box in YOLO format
    x_center = 0.5
    y_center = 0.5
    width = 1.0
    height = 1.0

    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}")

def process_class(class_name, class_id):
    folder = SOURCE_DIR / class_name
    images = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_SPLIT)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    print(f"Class '{class_name}'  →  train: {len(train_imgs)}, val: {len(val_imgs)}")

    # Process training images
    for img in train_imgs:
        dest_img = OUTPUT_DIR / "images/train" / img.name
        dest_lbl = OUTPUT_DIR / "labels/train" / (img.stem + ".txt")
        shutil.copy(img, dest_img)
        create_yolo_label(dest_img, dest_lbl, class_id)

    # Process validation images
    for img in val_imgs:
        dest_img = OUTPUT_DIR / "images/val" / img.name
        dest_lbl = OUTPUT_DIR / "labels/val" / (img.stem + ".txt")
        shutil.copy(img, dest_img)
        create_yolo_label(dest_img, dest_lbl, class_id)

# Process every class folder
for cname, cid in CLASS_MAP.items():
    process_class(cname, cid)

# Write data.yaml
yaml_path = OUTPUT_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write("train: ./images/train\n")
    f.write("val: ./images/val\n\n")
    f.write(f"nc: {len(CLASS_MAP)}\n")
    f.write("names: [ " + ", ".join([f"'{c}'" for c in CLASS_MAP.keys()]) + " ]\n")

print("\nDataset created at:", OUTPUT_DIR)