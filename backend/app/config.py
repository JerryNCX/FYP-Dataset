import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

supabase_url = os.getenv("VITE_SUPABASE_URL")
supabase_publishable_key = os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")


class Settings:
    """
    Central configuration for the backend.
    Adjust YOLO_WEIGHTS_PATH to point to your trained model file, e.g.:
    C:\\Users\\Admin\\Desktop\\Final Year Project\\ML Model\\ML_datasets\\train_dataset\\yolo_dataset\\runs\\detect\\train7\\weights\\best.pt
    """

    # Path to YOLO weights (.pt). This must be updated to an existing file.
    YOLO_WEIGHTS_PATH: str = os.getenv(
        "YOLO_WEIGHTS_PATH",
        "C:/Users/Admin/Desktop/FYP Dataset/ML Model/ML_datasets/train_dataset/yolo_dataset/runs/detect/train/weights/best.pt",  # TODO: update to real path on this machine
    )

    # OCR backend: "easyocr", "tesseract"/"pytesseract", or "mmocr"
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "tesseract")
    # Optional absolute path to tesseract.exe on Windows.
    # Default points to your local installation path.
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD",
        r"C:/Program Files/Tesseract-OCR/tesseract.exe",
    )

    # MMOCR (OpenMMLab): detection + recognition model names from MMOCR model zoo.
    # See https://mmocr.readthedocs.io/en/latest/user_guides/inference.html
    MMOCR_DEVICE: str = os.getenv("MMOCR_DEVICE", "cuda:0")
    MMOCR_DET: str = os.getenv("MMOCR_DET", "DBNet")
    MMOCR_REC: str = os.getenv("MMOCR_REC", "CRNN")

    # Supabase configuration (set in environment variables)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", supabase_url)
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", supabase_publishable_key)


@lru_cache()
def get_settings() -> Settings:
    return Settings()