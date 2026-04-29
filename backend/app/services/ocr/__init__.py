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

        settings = get_settings()
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        self.pytesseract = pytesseract

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split()).strip()

    def _build_variants(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Try multiple preprocessing variants because component labels vary a lot
        (dark background, low contrast, tiny text).
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Upscale small crops to help Tesseract recognition.
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.medianBlur(gray, 3)

        # Variant 1: Otsu threshold
        v1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Variant 2: Inverted Otsu threshold (for bright-on-dark labels)
        v2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # Variant 3: Adaptive threshold
        v3 = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        # Variant 4: Raw denoised grayscale
        v4 = denoised
        return [v1, v2, v3, v4]

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        variants = self._build_variants(image_bgr)
        # psm 6: assume a block of text; psm 7: single text line
        configs = ["--oem 3 --psm 6", "--oem 3 --psm 7"]

        best_text = ""
        best_conf = -1.0

        for img in variants:
            for cfg in configs:
                data = self.pytesseract.image_to_data(
                    img, output_type=self.pytesseract.Output.DICT, config=cfg
                )
                words: List[str] = []
                confs: List[float] = []

                for i, raw_text in enumerate(data.get("text", [])):
                    text = self._clean_text(raw_text)
                    if not text:
                        continue
                    try:
                        conf = float(data["conf"][i])
                    except (ValueError, TypeError, KeyError, IndexError):
                        conf = -1.0
                    if conf < 30:
                        continue
                    words.append(text)
                    confs.append(conf)

                if not words:
                    continue

                combined = self._clean_text(" ".join(words))
                avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
                if combined and avg_conf > best_conf:
                    best_text = combined
                    best_conf = avg_conf

        if not best_text:
            return []

        return [(best_text, best_conf / 100.0)]


class MMOCREngine:
    """
    OpenMMLab MMOCR end-to-end OCR (det + rec) on each YOLO crop.
    Install: see https://mmocr.readthedocs.io/en/latest/get_started/installation.html
    (mmcv/mmengine versions must match your PyTorch/CUDA; Windows may need prebuilt wheels).
    """

    def __init__(self) -> None:
        try:
            from mmocr.apis import MMOCRInferencer
        except ImportError as e:
            raise RuntimeError(
                "MMOCR is not installed. Install mmocr (and mmcv/mmengine per docs), "
                "or set OCR_BACKEND=easyocr or tesseract."
            ) from e

        settings = get_settings()
        self.inferencer = MMOCRInferencer(
            det=settings.MMOCR_DET,
            rec=settings.MMOCR_REC,
            device=settings.MMOCR_DEVICE,
        )

    @staticmethod
    def _parse_mmocr_predictions(result: dict) -> List[Tuple[str, float]]:
        preds = result.get("predictions") or []
        if not preds:
            return []
        p0 = preds[0]
        texts = p0.get("rec_texts") or []
        scores = p0.get("rec_scores") or []
        out: List[Tuple[str, float]] = []
        if texts and scores and len(texts) == len(scores):
            for t, s in zip(texts, scores):
                t = (t or "").strip()
                if t:
                    try:
                        out.append((t, float(s)))
                    except (TypeError, ValueError):
                        out.append((t, 0.0))
            return out
        # Recognition-only inferencer shape: single text per image
        text = (p0.get("text") or "").strip()
        if text:
            try:
                sc = float(p0.get("scores", 0.0))
            except (TypeError, ValueError):
                sc = 0.0
            return [(text, sc)]
        return []

    def recognize(self, image_bgr: np.ndarray) -> List[Tuple[str, float]]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.inferencer(image_rgb, return_vis=False)
        items = self._parse_mmocr_predictions(result)
        items.sort(key=lambda x: x[1], reverse=True)
        return items


_ocr_engine: OCREngine | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    settings = get_settings()
    backend = settings.OCR_BACKEND.lower()
    if backend in {"tesseract", "pytesseract"}:
        _engine: OCREngine = TesseractOCREngine()
    elif backend == "mmocr":
        _engine = MMOCREngine()
    else:
        _engine = EasyOCREngine()

    _ocr_engine = _engine
    return _ocr_engine


