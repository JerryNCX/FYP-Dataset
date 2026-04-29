from typing import List, Optional

from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class OCRResult(BaseModel):
    text: str
    confidence: float


class ComponentMatch(BaseModel):
    id: Optional[int] = None
    category: str
    product_name: Optional[str] = None
    asin: Optional[str] = None
    stars: Optional[str] = None
    rating_count: Optional[str] = None
    review_page: Optional[str] = None
    current_price: Optional[str] = None
    original_price: Optional[str] = None
    in_stock: Optional[str] = None
    recent_purchase: Optional[str] = None
    image: Optional[str] = None
    score: Optional[float] = None
    extra: Optional[dict] = None


class DetectionResult(BaseModel):
    category: str
    confidence: float
    bbox: BoundingBox
    ocr_results: List[OCRResult]
    matches: List[ComponentMatch]


class AnalyzeResponse(BaseModel):
    detections: List[DetectionResult]

