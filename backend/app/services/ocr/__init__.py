from typing import List, Protocol, Tuple

import cv2
import numpy as np

from backend.app.config import get_settings


class OCREngine(Protocol):
    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        ...


def preprocess_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    """
    Basic preprocessing to improve OCR:
      - convert to grayscale
      - apply adaptive thresholding
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Light denoise / blur can help reduce artifacts.
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    return thresh


class EasyOCREngine:
    def __init__(self) -> None:
        import easyocr

        # English only for now; adjust as needed.
        self.reader = easyocr.Reader(["en"], gpu=True)

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        # EasyOCR expects RGB
        processed = preprocess_for_ocr(image_bgr)
        image_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        results = self.reader.readtext(image_rgb, detail=1)
        outputs: List[Tuple[str, float]] = []
        for _, text, conf in results:
            outputs.append((text, float(conf)))
        return outputs


class TesseractOCREngine:
    def __init__(self) -> None:
        import pytesseract

        self.pytesseract = pytesseract

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        processed = preprocess_for_ocr(image_bgr)
        text = self.pytesseract.image_to_string(processed)
        # Tesseract does not return a confidence per text line via this API;
        # we assign a placeholder confidence.
        text = text.strip()
        if not text:
            return []
        return [(text, 0.5)]


_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    settings = get_settings()
    backend = settings.OCR_BACKEND.lower()
    if backend == "tesseract":
        _engine: OCREngine = TesseractOCREngine()
    else:
        _engine = EasyOCREngine()

    _ocr_engine = _engine
    return _ocr_engine


