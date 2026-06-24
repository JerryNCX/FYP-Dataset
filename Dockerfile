FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY start.sh .
COPY backend/ backend/
COPY ["ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train/weights/best.pt", "/app/models/best.pt"]

RUN mkdir -p backend/app/static

EXPOSE 8000

CMD ["bash", "start.sh"]
