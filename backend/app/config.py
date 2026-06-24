import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Walk up from this file's directory to find Final Year Project/.env
_env_candidate = Path(__file__).resolve().parent.parent.parent.parent / 'Final Year Project' / '.env'
if _env_candidate.exists():
    load_dotenv(_env_candidate)
else:
    load_dotenv()


class Settings:
    """
    Central configuration for the backend.
    """

    YOLO_WEIGHTS_PATH: str = os.getenv(
        "YOLO_WEIGHTS_PATH",
        "/app/models/best.pt",
    )

    # OCR backend: "easyocr", "tesseract"/"pytesseract", or "mmocr"
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "tesseract")
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD",
        "/usr/bin/tesseract",
    )

    MMOCR_DEVICE: str = os.getenv("MMOCR_DEVICE", "cuda:0")
    MMOCR_DET: str = os.getenv("MMOCR_DET", "DBNet")
    MMOCR_REC: str = os.getenv("MMOCR_REC", "CRNN")

    # Supabase configuration (reads VITE_* vars from shared .env)
    SUPABASE_URL: str = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY", "")


@lru_cache()
def get_settings() -> Settings:
    return Settings()