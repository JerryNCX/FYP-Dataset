from ultralytics import YOLO
import os
from pathlib import Path

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt for better accuracy

# Dataset paths
base_path = Path("Amazon Train Dataset")  # change if needed

splits = ["train", "val", "test"]

for split in splits:
    image_dir = base_path / "images" / split
    label_dir = base_path / "labels" / split

    # Create label directory if not exists
    os.makedirs(label_dir, exist_ok=True)

    print(f"\nProcessing {split} images...")

    # Run prediction
    results = model.predict(
        source=str(image_dir),
        save_txt=True,        # save YOLO format labels
        save_conf=False,      # optional
        conf=0.25             # confidence threshold
    )

    # Move generated labels into correct folder
    pred_label_dir = Path("runs/detect/predict/labels")

    if pred_label_dir.exists():
        for file in pred_label_dir.glob("*.txt"):
            target = label_dir / file.name
            file.replace(target)

    print(f"{split} done!")

print("\n✅ Auto-labeling completed!")