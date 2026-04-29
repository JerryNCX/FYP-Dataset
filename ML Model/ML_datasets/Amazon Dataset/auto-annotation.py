import os
import shutil
from pathlib import Path
from PIL import Image
import yaml  # ← better yaml writing

# ========================
# CONFIG – change only here
# ========================
SOURCE_ROOT   = Path("./Amazon Images")         # contains train/val/test
OUTPUT_ROOT   = Path("./Amazon Labels")         # final YOLO dataset

CLASS_MAP = {
    "cpu":                  0,
    "video-card":           1,
    "motherboard":          2,
    "memory":               3,
    "internal-hard-drive":  4,
    "power-supply":         5,
    "case":                 6,
    "cpu-cooler":           7
}

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}  # add more if needed

# ========================

def create_full_image_label(label_path: Path, class_id: int):
    """YOLO format: class x_center y_center width height  (all normalized 0–1)"""
    label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
    label_path.write_text(label_content.strip())


def main():
    if not SOURCE_ROOT.is_dir():
        print(f"❌ Source folder not found: {SOURCE_ROOT}")
        return

    print(f"Auto-annotation started...")
    print(f"  Source : {SOURCE_ROOT.resolve()}")
    print(f"  Output : {OUTPUT_ROOT.resolve()}\n")

    # Prepare output structure
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        split_dir = SOURCE_ROOT / split
        if not split_dir.exists():
            print(f"  Skip {split} — folder not found")
            continue

        print(f"→ Processing {split} ...")

        for class_name, class_id in CLASS_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue

            images = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            ]

            if not images:
                continue

            print(f"    {class_name:18} → {len(images):3d} images")

            for img_path in images:
                # Output paths
                out_img = OUTPUT_ROOT / "images" / split / img_path.name
                out_lbl = OUTPUT_ROOT / "labels" / split / f"{img_path.stem}.txt"

                # Copy image (use shutil.copy2 to preserve metadata)
                shutil.copy2(img_path, out_img)

                # Create YOLO label (whole image = object)
                create_full_image_label(out_lbl, class_id)

                stats[split] += 1

    # Summary
    total = sum(stats.values())
    print("\n" + "═" * 50)
    print("Finished auto-annotation")
    print(f"  Train : {stats['train']:4d} images")
    print(f"  Val   : {stats['val']:4d} images")
    print(f"  Test  : {stats['test']:4d} images")
    print(f"  Total : {total:4d} images/labels")
    print("═" * 50)

    # Create data.yaml (cleaner with yaml module)
    data = {
        "path": str(OUTPUT_ROOT.resolve()),  # optional but recommended
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",             # optional
        "nc": len(CLASS_MAP),
        "names": list(CLASS_MAP.keys())
    }

    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"\nCreated YOLO config: {yaml_path}")
    print("You can now train with:")
    print(f"   yolo task=detect mode=train model=yolov8s.pt data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()