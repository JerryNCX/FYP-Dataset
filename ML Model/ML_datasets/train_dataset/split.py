import os
import shutil
import random
from pathlib import Path

# ========== CONFIG ==========
SOURCE_DIR = Path("./image")  # your current folder with 8 classes
OUTPUT_DIR = Path("./split_dataset")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# ============================

# Create output folder structure
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

# Loop through each class folder
for class_folder in SOURCE_DIR.iterdir():
    if not class_folder.is_dir():
        continue
    
    class_name = class_folder.name
    print(f"Processing class: {class_name}")

    # Create class subfolders
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    # Collect all images
    images = [f for f in class_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # Copy files
    for img in train_imgs:
        shutil.copy(img, OUTPUT_DIR / "train" / class_name / img.name)

    for img in val_imgs:
        shutil.copy(img, OUTPUT_DIR / "val" / class_name / img.name)

    for img in test_imgs:
        shutil.copy(img, OUTPUT_DIR / "test" / class_name / img.name)

print("\n✨ DONE! Dataset split into:")
print("70% → train/")
print("20% → val/")
print("10% → test/")
print("\nFinal dataset saved at:", OUTPUT_DIR)
