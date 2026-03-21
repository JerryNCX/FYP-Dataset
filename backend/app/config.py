import os
from functools import lru_cache


class Settings:
    """
    Central configuration for the backend.
    Adjust YOLO_WEIGHTS_PATH to point to your trained model file, e.g.:
    C:\\Users\\Admin\\Desktop\\Final Year Project\\ML Model\\ML_datasets\\train_dataset\\yolo_dataset\\runs\\detect\\train7\\weights\\best.pt
    """

    # Path to YOLO weights (.pt). This must be updated to an existing file.
    YOLO_WEIGHTS_PATH: str = os.getenv(
        "YOLO_WEIGHTS_PATH",
        "C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train7 yolov8n/weights/best.pt",  # TODO: update to real path on this machine
    )

    # Default OCR backend identifier: "easyocr" or "tesseract"
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "easyocr")

    # Supabase configuration (you must set these in your environment)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://fkciyvajmrbbpdmrexuk.supabase.co")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZrY2l5dmFqbXJiYnBkbXJleHVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NDg2NTcsImV4cCI6MjA3OTEyNDY1N30.UH9W3fXwAxWks2rsyOFR950L3tdv4m8tQnoCmggTQzo6")


@lru_cache()
def get_settings() -> Settings:
    return Settings()