from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.schemas import AnalyzeResponse, BoundingBox, DetectionResult, OCRResult
from backend.app.services.ocr import get_ocr_engine
from backend.app.services.supabase import search_components
from backend.app.services.yolo import get_yolo_detector


YOLO_TO_DB_CATEGORY = {
    "cpu": "CPU",
    "gpu": "GPU",
    "video-card": "GPU",
    "ram": "RAM",
    "memory": "RAM",
    "storage": "SSD",
    "internal-hard-drive": "SSD",
    "motherboard": "Motherboard",
    "cooling": "CPU Cooler",
    "chassis": "PC Case",
    "power supply": "PSU",
    "psu": "PSU",
}


app = FastAPI(title="Component Analyzer (YOLO + OCR)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/html",
          StaticFiles(directory="backend/app/static", html=True),
          name="static")


def load_image_from_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_image(
    file: UploadFile = File(...),
) -> AnalyzeResponse:
    image_bgr = load_image_from_upload(file)

    yolo = get_yolo_detector()
    ocr_engine = get_ocr_engine()

    detections_raw = yolo.detect(image_bgr)
    detection_results: List[DetectionResult] = []

    for det in detections_raw:
        x1, y1, x2, y2 = det["bbox"]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # OCR on the cropped component.
        raw_ocr = ocr_engine.recognize(crop)
        ocr_results: List[OCRResult] = []
        for text, conf in raw_ocr:
            if not text:
                continue
            ocr_results.append(OCRResult(text=text, confidence=conf))

        # Map YOLO category to database category
        mapped_category = YOLO_TO_DB_CATEGORY.get(det["category"], det["category"])

        # Use the top OCR text (if any) to query Supabase.
        matches = []
        if ocr_results:
            best_text = ocr_results[0].text
            if best_text and len(best_text.strip()) > 1:
                matches = search_components(mapped_category, best_text)

        # Fallback: if no matches or OCR failed, search by category only
        if not matches:
            matches = search_components(mapped_category, "")

        detection_results.append(
            DetectionResult(
                category=det["category"],
                confidence=det["confidence"],
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                ocr_results=ocr_results,
                matches=matches,
            )
        )

    return AnalyzeResponse(detections=detection_results)


@app.get("/search")
def search_products(
    category: str = Query(..., description="Category: CPU, GPU, RAM, SSD, Motherboard, CPU Cooler, PC Case, PSU"),
    q: str = Query("", description="Search text (optional)")
):
    """
    Simple search endpoint for testing product search.
    Returns products matching the category and optional search text.
    """
    if not category:
        return {"error": "Category is required", "results": []}
    
    matches = search_components(category, q)
    return {"category": category, "query": q, "count": len(matches), "results": matches}

