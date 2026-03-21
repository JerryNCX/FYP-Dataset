import csv
import re
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = Path("Amazon Scraper Dataset.csv")
OUTPUT_CSV = Path("components_catalog_clean.csv")

# Map your scraped "Keyword" to YOLO category names
KEYWORD_TO_CATEGORY = {
    "CPU": "cpu",
    "Processor": "cpu",
    "Graphics Card": "video-card",
    "GPU": "video-card",
    "Motherboard": "motherboard",
    "RAM": "memory",
    "Memory": "memory",
    "HDD": "internal-hard-drive",
    "SSD": "internal-hard-drive",
    "Power Supply": "power-supply",
    "PSU": "power-supply",
    "Case": "case",
    "PC Case": "case",
    "CPU Cooler": "cpu-cooler",
    "Cooler": "cpu-cooler",
}

# Known brand tokens; extend as you like
BRANDS = [
    "AMD", "Intel", "Corsair", "Seasonic", "MSI", "ASUS", "Gigabyte", "GIGABYTE",
    "EVGA", "Cooler Master", "NZXT", "Thermaltake", "Be Quiet", "be quiet", "Noctua",
    "Samsung", "Crucial", "Kingston", "G.Skill", "G.SKILL", "Western Digital",
    "WD", "Seagate", "XFX", "Sapphire", "ZOTAC", "PowerColor"
]


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def detect_category(keyword: str) -> str | None:
    if not keyword:
        return None
    k = keyword.strip()
    # Try direct mapping
    if k in KEYWORD_TO_CATEGORY:
        return KEYWORD_TO_CATEGORY[k]
    # Fallback: partial contains
    for key, cat in KEYWORD_TO_CATEGORY.items():
        if key.lower() in k.lower():
            return cat
    return None


def is_row_plausible_for_category(category: str, name: str) -> bool:
    """Rough filters to discard obviously wrong rows."""
    name_up = (name or "").upper()

    if category == "cpu":
        cpu_tokens = ["AMD", "INTEL", "RYZEN", "CORE", "I3", "I5", "I7", "I9", "XEON"]
        bad_tokens = ["MOTHERBOARD", "CASE", "COOLER", "GPU", "GRAPHICS", "RTX", "GTX"]
        if not any(t in name_up for t in cpu_tokens):
            return False
        if any(t in name_up for t in bad_tokens):
            return False
        return True

    if category == "video-card":
        gpu_tokens = ["RTX", "GTX", "RADEON", "RX", "NVIDIA", "GEFORCE"]
        if not any(t in name_up for t in gpu_tokens):
            return False
        return True

    if category == "motherboard":
        tokens = ["MOTHERBOARD", "B450", "B550", "X570", "Z490", "Z590", "Z690", "X670"]
        return any(t in name_up for t in tokens)

    if category == "memory":
        tokens = ["DDR4", "DDR5", "RAM", "SO-DIMM", "SODIMM"]
        return any(t in name_up for t in tokens)

    if category == "internal-hard-drive":
        tokens = ["HDD", "SSD", "NVME", "M.2", "M2", "2.5\"", '3.5"']
        return any(t in name_up for t in tokens)

    if category == "power-supply":
        tokens = [
            "PSU", "POWER SUPPLY", "80+", "80 PLUS", "BRONZE", "GOLD", "PLATINUM",
            "650W", "700W", "750W", "850W", "1000W", "1200W"
        ]
        return any(t in name_up for t in tokens)

    if category == "case":
        tokens = ["PC CASE", "MID TOWER", "FULL TOWER", "MICRO ATX", "ATX CASE"]
        # Many case names are vague, but at least require "CASE" or "TOWER"
        return any(t in name_up for t in tokens)

    if category == "cpu-cooler":
        tokens = ["COOLER", "AIO", "LIQUID COOLER", "AIR COOLER", "CPU COOLER"]
        return any(t in name_up for t in tokens)

    return True  # default: keep


def extract_brand_and_model(product_name: str) -> tuple[str | None, str | None]:
    """
    Very rough heuristic:
    - Brand = first known brand token found.
    - Model = the remaining part of the string after that brand token.
    """
    if not product_name:
        return None, None

    name = normalize_text(product_name)
    name_up = name.upper()

    # Try multi-word brands first (e.g. "Cooler Master", "Be Quiet")
    multi_brands = [b for b in BRANDS if " " in b]
    for b in multi_brands:
        if b.upper() in name_up:
            idx = name_up.index(b.upper())
            brand = name[idx : idx + len(b)]
            model = name[idx + len(b) :].strip(" -,:")
            return brand, model or None

    # Single-word brands
    single_brands = [b for b in BRANDS if " " not in b]
    for b in single_brands:
        if b.upper() in name_up:
            idx = name_up.index(b.upper())
            brand = name[idx : idx + len(b)]
            model = name[idx + len(b) :].strip(" -,:")
            return brand, model or None

    # Fallback: no known brand detected
    return None, name


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f_in, OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["category", "brand", "model", "search_text", "raw_name", "raw_keyword"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        kept = 0
        total = 0

        for row in reader:
            total += 1
            keyword = row.get("Keyword") or row.get("keyword") or ""
            product_name = row.get("Product_name") or row.get("Product Name") or row.get("title") or ""

            category = detect_category(keyword)
            if not category:
                continue

            if not is_row_plausible_for_category(category, product_name):
                continue

            brand, model = extract_brand_and_model(product_name)
            brand_norm = normalize_text(brand or "")
            model_norm = normalize_text(model or "")

            search_text = normalize_text(
                f"{category} {brand_norm} {model_norm} {product_name}"
            )

            writer.writerow(
                {
                    "category": category,
                    "brand": brand_norm or None,
                    "model": model_norm or None,
                    "search_text": search_text,
                    "raw_name": product_name,
                    "raw_keyword": keyword,
                }
            )
            kept += 1

    print(f"Finished. Read {total} rows, kept {kept} rows into {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()