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


class SpecProduct(BaseModel):
    id: int
    table_name: str
    name: str
    price: Optional[str] = None
    image_urls: Optional[str] = None
    manufacturer: Optional[str] = None
    stock: Optional[int] = None
    specs: dict = {}


class DetectionResult(BaseModel):
    category: str
    confidence: float
    bbox: BoundingBox
    ocr_results: List[OCRResult]
    matches: List[SpecProduct]


class AnalyzeResponse(BaseModel):
    detections: List[DetectionResult]


class SearchResponse(BaseModel):
    query: str
    category: Optional[str] = None
    count: int
    results: List[SpecProduct]


class SuggestRequest(BaseModel):
    table: str
    id: int


class SuggestResponse(BaseModel):
    cpu: List[SpecProduct] = []
    gpu: List[SpecProduct] = []
    motherboard: List[SpecProduct] = []
    memory: List[SpecProduct] = []
    case: List[SpecProduct] = []
    psu: List[SpecProduct] = []
    cpu_cooler: List[SpecProduct] = []
    internal_drive: List[SpecProduct] = []
