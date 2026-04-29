import re
from typing import List, Optional

from supabase import create_client, Client

from backend.app.config import get_settings
from backend.app.schemas import ComponentMatch


_supabase_client: Client | None = None


KEYWORDS = {
    "CPU": [
        "AMD", "INTEL", "RYZEN", "CORE", "I9", "I7", "I5", "I3",
        "9800X3D", "9700X", "9600X", "7800X3D", "7700X", "7600X", "7500F",
        "5800X", "5700X", "5600X", "5600", "5500", "5400",
        "12700K", "12600K", "12400", "14400F", "13900HK",
        "265K", "245KF", "225F", "14300F",
        "GEEKOM", "NZXT", "COOLER MASTER", "ID-COOLING", "UPHERE",
        "ACEMAGICIAN", "ACEMAGIC"
    ],
    "GPU": [
        "NVIDIA", "AMD", "RADEON", "RTX", "GTX", "RX",
        "4090", "4080", "4070", "4060", "3090", "3080", "3070", "3060",
        "7900XTX", "7900XT", "7800XT", "7700XT", "7600XT", "6600XT", "6500XT",
        "ASUS", "MSI", "GIGABYTE", "EVGA", "ZOTAC", "PNY", "GALAX",
        "ROG STRIX", "GAMING X", "FE", "FOUNDER EDITION"
    ],
    "RAM": [
        "CORSAIR", "KINGSTON", "G.SKILL", "CRUCIAL", "PATRIOT", "TEAMGROUP",
        "VENGEANCE", "DOMINATOR", "TRIDENT", "FLARE", "FURY", "RIPJAW",
        "DDR4", "DDR5", "3200MHZ", "3600MHZ", "4000MHZ", "4800MHZ", "5200MHZ", "5600MHZ", "6000MHZ", "6400MHZ", "7200MHZ",
        "8GB", "16GB", "32GB", "64GB",
        "2X8GB", "2X16GB", "4X8GB", "4X16GB", "KIT"
    ],
    "SSD": [
        "SAMSUNG", "WD", "WESTERN DIGITAL", "CRUCIAL", "KINGSTON", "SEAGATE",
        "SABRENT", "ADATA", "MICRON",
        "980", "990", "990 PRO", "980 PRO", "970", "960", "950",
        "SN850X", "SN850", "SN770", "SN750", "SN570", "SN550",
        "P5", "P2", "P1",
        "MX500", "MX300",
        "A2000", "KC3000", "KC2500",
        "NVME", "M.2", "SATA", "PCIE", "GEN4", "GEN5"
    ],
    "Motherboard": [
        "ASUS", "MSI", "GIGABYTE", "ASROCK",
        "ROG", "ROG STRIX", "TUF", "PRIME", "STRIX",
        "MAG", "MORTAR", "BAZOOKA", "PRO",
        "AORUS", "ULTRA", "GAMING", "ELITE", "VISION",
        "PHANTOM", "TAICHI",
        "Z790", "Z690", "Z650", "Z590", "Z490",
        "B650", "B550", "B450", "B660", "B760",
        "A620", "X670", "X570", "X470",
        "LGA1700", "LGA1200", "AM5", "AM4"
    ],
    "CPU Cooler": [
        "NOCTUA", "COOLER MASTER", "CORSAIR", "NZXT", "BE QUIET", "THERMALTAKE",
        "ID-COOLING", "DEEPCOOL", "ARCTIC", "SCYTHE", "SILVERSTONE",
        "NH-D15", "NH-U12S", "U12A", "C14S",
        "MASTERLIQUID", "HYPER", "V8", "NR200P",
        "iCUE", "KRAKEN", "Kraken",
        "DARK ROCK", "SILENT LOOP", "PURE ROCK",
        "FROZN", "SE-234", "SE-224",
        "LGA1700", "LGA1200", "AM5", "AM4", "AM3"
    ],
    "PC Case": [
        "NZXT", "CORSAIR", "FRACTAL", "PHANTEKS", "LIAN LI", "THERMALTAKE",
        "BE QUIET", "COOLER MASTER", "MSI", "ASUS", "GAMDIAS",
        "H5", "H7", "H9", "H510", "H710",
        "4000D", "5000D", "5000X", "7000D", "900D",
        "TORQUE", "EVOLVE", "MEK", "FLUX",
        "TEMPEST", "SETTLER", "U", "O11", "O12", "LANCOOL",
        "VIEW", "SUPERNOVA", "A300", "A400",
        "MID TOWER", "FULL TOWER", "MINI ITX", "ATX"
    ],
    "PSU": [
        "CORSAIR", "EVGA", "SEASONIC", "BE QUIET", "THERMALTAKE", "COOLER MASTER",
        "ASUS", "MSI", "GIGABYTE", "FSP", "SILVERSTONE", "ENERMAX",
        "RM", "RMX", "RMi", "RM850X", "RM1000X",
        "SF", "SFX", "Toughpower", "Tough Power",
        "SuperNOVA", "G2", "G3", "G5", "G6", "G7",
        "PRIME", "FOCUS", "CORE",
        "STRAIGHT POWER", "PURE POWER", "DARK POWER",
        "80+ GOLD", "80+ PLATINUM", "80+ BRONZE", "80+ TITANIUM",
        "850W", "750W", "650W", "1000W", "1200W", "600W"
    ]
}


def extract_keywords(ocr_text: str) -> List[str]:
    """Extract keywords from OCR text that match known brand/model patterns."""
    if not ocr_text or not ocr_text.strip():
        return []
    
    text_upper = ocr_text.upper()
    found_keywords = set()
    
    words = re.findall(r'[A-Za-z0-9]+', text_upper)
    
    for category, keyword_list in KEYWORDS.items():
        for keyword in keyword_list:
            if keyword in text_upper:
                found_keywords.add(keyword)
    
    for word in words:
        if len(word) >= 2 and len(word) <= 20:
            for category, keyword_list in KEYWORDS.items():
                if word in keyword_list or word.startswith(tuple(k for k in keyword_list if len(k) >= 3)):
                    found_keywords.add(word)
    
    return list(found_keywords)


def build_keyword_query(category: str, keywords: List[str], limit: int = 5) -> List[dict]:
    """Build OR query based on detected keywords using multiple queries."""
    if not keywords:
        return []
    
    sb = get_supabase()
    
    all_results = []
    seen_ids = set()
    
    for kw in keywords:
        try:
            query = sb.table("Product_Catalog").select("*")
            query = query.eq("category", category)
            query = query.ilike("Product_name", f"%{kw}%")
            query = query.limit(limit)
            
            resp = query.execute()
            rows = resp.data or []
            
            for row in rows:
                row_id = row.get("id")
                if row_id and row_id not in seen_ids:
                    all_results.append(row)
                    seen_ids.add(row_id)
                    if len(all_results) >= limit:
                        break
            
            if len(all_results) >= limit:
                break
                
        except Exception as e:
            print(f"Query error for keyword '{kw}': {e}")
            continue
    
    return all_results[:limit]


def get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    settings = get_settings()
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        raise RuntimeError(
            "Supabase URL/KEY not configured. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    return _supabase_client


def search_components(
    category: str,
    ocr_text: str,
    limit: int = 5,
) -> List[ComponentMatch]:
    """
    Query Supabase Product_Catalog table using category + keyword extraction.
    Strategy:
      1) Extract keywords from OCR text
      2) Build OR query using extracted keywords
      3) Fallback to category-only search if no keywords or no results
    """
    keywords = extract_keywords(ocr_text)
    print(f"Category: {category}, OCR: '{ocr_text}', Keywords: {keywords}")
    
    rows = []
    if keywords:
        rows = build_keyword_query(category, keywords, limit)
    
    if not rows:
        sb = get_supabase()
        query = (
            sb.table("Product_Catalog")
            .select("*")
            .eq("category", category)
            .limit(limit)
        )
        try:
            resp = query.execute()
            rows = resp.data or []
        except Exception as e:
            print(f"Fallback search error: {e}")
            rows = []

    matches: List[ComponentMatch] = []
    for r in rows:
        matches.append(
            ComponentMatch(
                id=r.get("id"),
                category=r.get("category", category),
                product_name=r.get("Product_name"),
                asin=r.get("ASIN"),
                stars=r.get("Stars"),
                rating_count=r.get("Rating_count"),
                review_page=r.get("Review_Page"),
                current_price=r.get("Current_price"),
                original_price=r.get("Original_price"),
                in_stock=r.get("In_stock"),
                recent_purchase=r.get("Recent_purchase"),
                image=r.get("Image"),
                score=None,
                extra={k: v for k, v in r.items() if k not in {"id", "category", "Product_name", "ASIN", "Stars", "Rating_count", "Review_Page", "Current_price", "Original_price", "In_stock", "Recent_purchase", "Image"}},
            )
        )
    print(f"Found {len(matches)} matches")
    return matches