import os
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

# 1. Setup Paths
MODEL_PATH = r"C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train7 yolov8n/weights/best.pt"
SOURCE_DIR = Path(r"C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/Amazon Dataset/Amazon Images")
OUTPUT_DIR = Path(r"C:/Users/Admin/Desktop/Final Year Project/ML Model/ML_datasets/Amazon Dataset/Own Model Dataset")
def debug_sort():
    # CHECKPOINT 1: Does the Source even exist?
    if not SOURCE_DIR.exists():
        print(f"❌ ERROR: Source directory NOT FOUND at: {SOURCE_DIR}")
        return
    
    # CHECKPOINT 2: Are there images inside?
    # Checking for .jpg, .jpeg, .png, .webp (case insensitive)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG')
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(valid_extensions)]
    print(f"📂 Found {len(files)} total images in source folder.")
    
    if len(files) == 0:
        print("❌ ERROR: No images found! Check your path or file extensions.")
        return

    # CHECKPOINT 3: Load Model
    print("Loading model...")
    model = YOLO(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
    class_names = model.names
    print(f"📦 Model loaded with {len(class_names)} classes.")

    # CHECKPOINT 4: Create Folders (Doing this BEFORE the loop)
    print(f"📁 Creating base directory at: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for name in class_names.values():
        folder_path = OUTPUT_DIR / name
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created folder: {name}") # This MUST print in your console

    # 5. Run Prediction
    print("🚀 Starting inference...")
    results = model.predict(source=str(SOURCE_DIR), conf=0.25, stream=True)

    processed_count = 0
    for result in results:
        img_path = Path(result.path)
        
        # Determine class
        if len(result.boxes) > 0:
            label = class_names[int(result.boxes[0].cls.item())]
        else:
            label = "unclassified"
            (OUTPUT_DIR / label).mkdir(exist_ok=True)

        # Move File
        dest = OUTPUT_DIR / label / img_path.name
        shutil.copy(img_path, dest)
        processed_count += 1

    print(f"\n✅ Finished! Processed {processed_count} images.")

if __name__ == "__main__":
    debug_sort()