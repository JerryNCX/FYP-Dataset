import os
import cv2
import csv
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer

# ----------------------
# CONFIG
# ----------------------
image_folder = "C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/data/image"
output_dir = "mmocr_results"
crop_dir = os.path.join(output_dir, "crops")
csv_path = os.path.join(output_dir, "results.csv")

os.makedirs(crop_dir, exist_ok=True)

# ----------------------
# LOAD MODELS
# ----------------------
yolo_model = YOLO("C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train7 yolov8n/weights/best.pt")
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')

# ----------------------
# CSV SETUP
# ----------------------
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["image", "crop", "recognized_text"])

# ----------------------
# PROCESS IMAGES
# ----------------------
for file in os.listdir(image_folder):

    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, file)
    print(f"Processing: {file}")

    img = cv2.imread(image_path)

    # YOLO detection
    results = yolo_model(image_path)
    boxes = results[0].boxes

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]

        crop_name = f"{file}_crop_{i}.jpg"
        crop_path = os.path.join(crop_dir, crop_name)

        # Save crop
        cv2.imwrite(crop_path, crop)

        # OCR on crop
        ocr_result = ocr(crop_path)

        # Extract text safely
        try:
            texts = ocr_result['predictions'][0]['rec_texts']
            final_text = " ".join(texts)
        except:
            final_text = ""

        print(f" -> OCR: {final_text}")

        # Save to CSV
        writer.writerow([file, crop_name, final_text])

csv_file.close()

print("\n✅ All results saved in:", output_dir)