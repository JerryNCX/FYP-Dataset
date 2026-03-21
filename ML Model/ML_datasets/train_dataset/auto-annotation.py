import os
from pathlib import Path
from PIL import Image

# ========================
# CONFIG
SOURCE_DIR = Path("./split_dataset")  # WHERE train/ val/ test/ ARE
OUTPUT_DIR = Path("./yolo_dataset")   # FINAL YOLO DATASET
# ========================

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

# Create YOLO folders
for subset in ["train", "val", "test"]:
    (OUTPUT_DIR / "images" / subset).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / subset).mkdir(parents=True, exist_ok=True)

def create_label(img_path, label_path, class_id):
    """Write YOLO label with full-image box."""
    with Image.open(img_path) as img:
        # full-image bounding box
        label = f"{class_id} 0.5 0.5 1 1"

    with open(label_path, "w") as f:
        f.write(label)

# Process splits
for split in ["train", "val", "test"]:
    print(f"Processing {split}...")

    split_path = SOURCE_DIR / split

    # Loop through class folders
    for class_name, class_id in CLASS_MAP.items():
        class_folder = split_path / class_name
        if not class_folder.exists():
            continue

        # List images
        imgs = [f for f in class_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        for img in imgs:
            # Copy image
            out_img = OUTPUT_DIR / "images" / split / img.name
            out_lbl = OUTPUT_DIR / "labels" / split / (img.stem + ".txt")

            # Copy image file
            with open(img, "rb") as src_file:
                with open(out_img, "wb") as dst_file:
                    dst_file.write(src_file.read())

            # Create YOLO label
            create_label(out_img, out_lbl, class_id)

print("\n✨ Auto-annotation complete.")
print("YOLO-ready dataset created at:", OUTPUT_DIR)

# Create data.yaml
yaml_path = OUTPUT_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"train: {OUTPUT_DIR}/images/train\n")
    f.write(f"val: {OUTPUT_DIR}/images/val\n")
    f.write(f"test: {OUTPUT_DIR}/images/test\n\n")
    f.write(f"nc: {len(CLASS_MAP)}\n")
    f.write("names: [ " + ", ".join([f"'{n}'" for n in CLASS_MAP.keys()]) + " ]\n")

print("\n📄 data.yaml created.")
