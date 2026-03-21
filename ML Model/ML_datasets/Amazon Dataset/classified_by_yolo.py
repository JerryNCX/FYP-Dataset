import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# 1. Load your model
# You can use a local .pt file or a Roboflow model ID
# Example: model = YOLO("pc-parts-detector.pt") 
model = YOLO("C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train7 yolov8n/weights/best.pt") # Use 'n' for speed, 'm' or 'l' for accuracy

# Define your classes (Ensure these match your model's class indices)
class_names = [
    'cpu', 'video-card', 'motherboard', 'memory', 
    'internal-hard-drive', 'power-supply', 'case', 'cpu-cooler'
]

source_dir = Path("Amazon Images")
output_dir = Path("Amazon classified Dataset")

# Create folders
for name in class_names:
    (output_dir / name).mkdir(parents=True, exist_ok=True)
(output_dir / "unclassified").mkdir(parents=True, exist_ok=True)

# 2. Run Inference
results = model.predict(source=str(source_dir), conf=0.5, save=False)

for result in results:
    img_path = Path(result.path)
    
    # Check if anything was detected
    if len(result.boxes) > 0:
        # Get the detection with the highest confidence
        top_box = result.boxes[0] 
        class_id = int(top_box.cls.item())
        
        # Guard against index errors if model has more/different classes
        if class_id < len(class_names):
            pred_class = class_names[class_id]
        else:
            pred_class = "unclassified"
            
        print(f"✅ {img_path.name} -> {pred_class} ({top_box.conf.item():.2f})")
    else:
        pred_class = "unclassified"
        print(f"❓ {img_path.name} -> Nothing detected")

    # 3. Move/Copy file
    dest = output_dir / pred_class / img_path.name
    shutil.copy(img_path, dest)

print("\n🚀 YOLO Classification Complete!")