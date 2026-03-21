import os
import cv2
import csv
import easyocr
import numpy as np

# ------------------------
# CONFIG
# ------------------------
INPUT_DIR = "C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/data/image"     # your collected images
OUTPUT_DIR = "EasyOCR_dataset"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

MIN_W = 40
MIN_H = 15

os.makedirs(IMG_DIR, exist_ok=True)

# ------------------------
# EasyOCR
# ------------------------
reader = easyocr.Reader(['en'], gpu=True)

# ------------------------
# CSV
# ------------------------
csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["filename", "text"])

counter = 0

# ------------------------
# PROCESS IMAGES
# ------------------------
for img_name in os.listdir(INPUT_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = reader.readtext(img, detail=1)

    for bbox, text, conf in results:
        if conf < 0.4:
            continue

        pts = np.array(bbox).astype(int)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()

        if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
            continue

        crop = img[y1:y2, x1:x2]

        filename = f"text_{counter:05d}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, filename), crop)

        writer.writerow([filename, text])
        counter += 1

csv_file.close()

print(f"Dataset built: {counter} samples")
