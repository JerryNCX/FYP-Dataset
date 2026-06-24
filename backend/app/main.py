import json
import re
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.schemas import (
    AnalyzeResponse,
    BoundingBox,
    DetectionResult,
    OCRResult,
    SearchResponse,
    SpecProduct,
    SuggestRequest,
    SuggestResponse,
)
from backend.app.services.ocr import get_ocr_engine
from backend.app.services.supabase import (
    ALL_TABLES,
    CATEGORY_TABLE_MAP,
    _supabase as _sb,
    search_components_ocr,
    search_all_categories,
    search_table,
    search_table_count,
    get_product,
    update_product,
    delete_product,
    create_authenticated_client,
    check_admin_email,
    get_category_info_list,
    get_manufacturers,
    save_user_build,
    list_user_builds,
    get_user_build,
    delete_user_build,
    sync_user_profile,
    get_user_profile,
    upsert_user_profile,
    create_order,
    list_user_orders,
    list_all_orders,
    get_user_order,
    calculate_shipping,
    get_cart,
    sync_cart,
    delete_cart_items,
    clear_cart,
    set_cart_build,
    catalog_add_item,
    catalog_remove_item,
    validate_cart_stock,
    reserve_stock,
    confirm_payment_stock,
    release_reservation,
    update_order_status,
)
from backend.app.services.yolo import get_yolo_detector


YOLO_TO_DB_CATEGORY = {
    "cpu": "CPU",
    "gpu": "GPU",
    "video-card": "GPU",
    "ram": "Memory",
    "memory": "Memory",
    "storage": "Internal Drive",
    "internal-hard-drive": "Internal Drive",
    "motherboard": "Motherboard",
    "cooling": "CPU Cooler",
    "chassis": "Case",
    "power supply": "PSU",
    "psu": "PSU",
}


app = FastAPI(title="Component Analyzer (YOLO + OCR + Spec Tables)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/html",
    StaticFiles(directory="backend/app/static", html=True),
    name="static",
)

print("Loading YOLO model at startup...")
get_yolo_detector()
print("YOLO model loaded.")


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

        raw_ocr = ocr_engine.recognize(crop)
        ocr_results: List[OCRResult] = []
        for text, conf in raw_ocr:
            if not text:
                continue
            ocr_results.append(OCRResult(text=text, confidence=conf))

        mapped_category = YOLO_TO_DB_CATEGORY.get(det["category"], det["category"])

        matches = []
        if ocr_results:
            best_text = ocr_results[0].text
            if best_text and len(best_text.strip()) > 1:
                matches = search_components_ocr(mapped_category, best_text)

        if not matches:
            matches = search_components_ocr(mapped_category, "")

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


@app.get("/search", response_model=SearchResponse)
def search_products(
    q: str = Query("", description="Search query"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    limit: int = Query(20, description="Max results"),
    offset: int = Query(0, ge=0, description="Result offset for pagination"),
    sort_by: str = Query("default", description="Sort: price_asc, price_desc, name_asc, name_desc, default"),
    in_stock_first: bool = Query(True, description="Show in-stock products first"),
    manufacturer: Optional[str] = Query(None, description="Filter by manufacturer/brand"),
    price_min: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    price_max: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    filters: Optional[str] = Query(None, description='JSON filters e.g. {"socket":"AM5","core_count":{"min":6}}'),
):
    parsed_filters = json.loads(filters) if filters else None
    kwargs = dict(manufacturer=manufacturer, price_min=price_min, price_max=price_max, filters=parsed_filters)

    if category and category in CATEGORY_TABLE_MAP:
        from backend.app.services.supabase import search_table, search_table_count

        table_name = CATEGORY_TABLE_MAP[category]
        total = search_table_count(table_name, q, **kwargs)
        results = search_table(table_name, q, limit=limit, offset=offset, sort_by=sort_by, in_stock_first=in_stock_first, **kwargs)
    else:
        from backend.app.services.supabase import search_all_tables_count

        total = search_all_tables_count(q, **kwargs)
        results = search_all_categories(q, limit=limit, offset=offset, sort_by=sort_by, in_stock_first=in_stock_first, **kwargs)

    return SearchResponse(query=q, category=category, count=total, results=results)


@app.get("/search/filters/{category}")
def get_category_filters(category: str):
    if category not in CATEGORY_TABLE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown category '{category}'")
    from backend.app.services.supabase import get_filters
    return get_filters(category)


@app.get("/products/{table_name}/{product_id}")
def get_product_endpoint(table_name: str, product_id: int):
    from backend.app.services.supabase import ALL_TABLES

    if table_name not in ALL_TABLES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown table '{table_name}'. Valid tables: {', '.join(ALL_TABLES)}",
        )

    product = get_product(table_name, product_id)
    if not product:
        raise HTTPException(
            status_code=404,
            detail=f"Product not found in {table_name} with id {product_id}",
        )

    return product


@app.get("/categories")
def list_categories():
    return get_category_info_list()


@app.get("/brands")
def list_brands():
    return get_manufacturers()


def _extract_int(value) -> Optional[int]:
    """Extract the first integer from a string like '320 mm', '304 W', '3', etc."""
    if value is None:
        return None
    s = str(value).strip().lower().replace(",", "")
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return None


def _has_m2_slots(specs: dict) -> bool:
    """Check if motherboard has M.2 slots by looking for 'm2' in spec keys."""
    for k, v in specs.items():
        if "m2" in k.lower() or "m." in k.lower():
            s = str(v).lower()
            if s and s != "0" and s != "no" and s != "none" and s != "":
                return True
    return False


FORM_FACTOR_HIERARCHY = {
    "Mini ITX": ["Mini ITX", "Micro ATX", "ATX", "E-ATX"],
    "Micro ATX": ["Micro ATX", "ATX", "E-ATX"],
    "ATX": ["ATX", "E-ATX"],
    "E-ATX": ["E-ATX"],
}

PSU_TYPE_MAP = {
    "mini itx": "SFX",
    "micro atx": "ATX",
    "atx": "ATX",
    "atx mid tower": "ATX",
    "atx full tower": "ATX",
}

def _expand_form_factors(form_factor: str) -> list[str]:
    return FORM_FACTOR_HIERARCHY.get(form_factor, [form_factor])

def _reverse_form_factors(supported_ff: str) -> list[str]:
    result = set()
    parts = [p.strip() for p in supported_ff.replace('\n', ',').split(',') if p.strip()]
    for part in parts:
        for k, v in FORM_FACTOR_HIERARCHY.items():
            if part in v:
                result.add(k)
    return list(result)

def _infer_psu_type(case_type: str) -> str:
    key = case_type.strip().lower()
    for pattern, psu_type in PSU_TYPE_MAP.items():
        if pattern in key:
            return psu_type
    return "ATX"

def _matches_any(target: str, values: list[str]) -> bool:
    tl = target.lower()
    parts = [p.strip() for p in tl.replace('\n', '|').replace(',', '|').split('|') if p.strip()]
    for v in values:
        vl = v.lower()
        for part in parts:
            if vl == part or part.startswith(vl + ' ') or part.endswith(' ' + vl) or (' ' + vl + ' ') in part:
                return True
    return False

# CONSTRAINT_MAP[(from_table, target_table)] = (source_spec_key, resolver_type, target_spec_key_or_none)
# resolver_type: ilike | hierarchy | hierarchy_rev | gte | gte_rev | math:psu_wattage | multi:m2 | multi:drive_bays | infer:psu_type
CONSTRAINT_MAP = {
    ("CPU", "Motherboard"):        ("socket",             "ilike",            "socket"),
    ("CPU", "CPU Cooler"):         ("socket",             "ilike",            "cpu_socket"),
    ("Motherboard", "CPU"):        ("socket",             "ilike",            "socket"),
    ("Motherboard", "Memory"):     ("memory_type",        "ilike",            "form_factor"),
    ("Motherboard", "Case"):       ("form_factor",        "hierarchy",        "motherboard_form_factor"),
    ("Motherboard", "CPU Cooler"): ("socket",             "ilike",            "cpu_socket"),
    ("Motherboard", "Internal Drive"): ("m2_slots",       "multi:m2",         None),
    ("GPU", "Case"):               ("length",             "gte",              "max_gpu_length"),
    ("GPU", "PSU"):                ("tdp",                "math:psu_wattage", None),
    ("Case", "Motherboard"):       ("motherboard_form_factor", "hierarchy_rev", "form_factor"),
    ("Case", "GPU"):               ("max_gpu_length",     "gte_rev",          "length"),
    ("Case", "PSU"):               ("type",               "infer:psu_type",   None),
    ("Case", "Internal Drive"):    ("internal_35_bays",   "multi:drive_bays", None),
    ("Memory", "Motherboard"):     ("speed",              "ilike",            None),
    ("PSU", "Case"):               ("type",               "ilike",            None),
    ("CPU Cooler", "CPU"):         ("cpu_socket",         "ilike",            "socket"),
    ("CPU Cooler", "Motherboard"): ("cpu_socket",         "ilike",            "socket"),
    ("Internal Drive", "Motherboard"): ("form_factor",    "ilike",            None),
    ("Internal Drive", "Case"):    ("form_factor",        "drive_bay_fit",    None),
}


def _apply_constraint(
    from_product: SpecProduct,
    target_table: str,
    resolver_type: str,
    source_key: str,
    target_key: Optional[str],
    context_cpu_tdp: Optional[int] = None,
    context_gpu_tdp: Optional[int] = None,
) -> list[SpecProduct]:
    specs = from_product.specs

    if resolver_type == "ilike":
        val = specs.get(source_key, "")
        if not val:
            return search_table(target_table, "", limit=100)
        if target_key:
            from backend.app.services.supabase import search_table_by_column
            candidates = search_table_by_column(target_table, target_key, str(val), limit=100)
        else:
            candidates = search_table(target_table, str(val), limit=100)
        return candidates

    elif resolver_type == "hierarchy":
        ff = specs.get(source_key, "")
        if not ff:
            return search_table(target_table, "", limit=100)
        compatible = _expand_form_factors(ff)
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            pv = p.specs.get(target_key, "") if target_key else p.name
            if _matches_any(str(pv), compatible):
                results.append(p)
        return results

    elif resolver_type == "hierarchy_rev":
        ff = specs.get(source_key, "")
        if not ff:
            return search_table(target_table, "", limit=100)
        compatible = _reverse_form_factors(ff)
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            pv = p.specs.get(target_key, "") if target_key else p.name
            if _matches_any(str(pv), compatible):
                results.append(p)
        return results

    elif resolver_type == "gte":
        src_val = _extract_int(specs.get(source_key))
        if src_val is None:
            return search_table(target_table, "", limit=100)
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            tv = _extract_int(p.specs.get(target_key) if target_key else p.name)
            if tv is not None and tv >= src_val:
                results.append(p)
        return results

    elif resolver_type == "gte_rev":
        max_val = _extract_int(specs.get(source_key))
        if max_val is None:
            return search_table(target_table, "", limit=100)
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            tv = _extract_int(p.specs.get(target_key) if target_key else p.name)
            if tv is not None and tv <= max_val:
                results.append(p)
        return results

    elif resolver_type == "math:psu_wattage":
        gpu_tdp = _extract_int(specs.get(source_key))
        if gpu_tdp is None and context_gpu_tdp is not None:
            gpu_tdp = context_gpu_tdp
        cpu_tdp = context_cpu_tdp or 0
        gpu_tdp = gpu_tdp or 0
        needed = cpu_tdp + gpu_tdp + 200
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            w = _extract_int(p.specs.get("wattage"))
            if w is not None and w >= needed:
                results.append(p)
        return results

    elif resolver_type == "multi:m2":
        if _has_m2_slots(specs):
            return search_table(target_table, "M.2", limit=100)
        return search_table(target_table, "", limit=100)

    elif resolver_type == "multi:drive_bays":
        bays = _extract_int(specs.get(source_key))
        if bays is None or bays <= 0:
            return []
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            pf = str(p.specs.get("form_factor", ""))
            if "3.5" in pf or bays >= 1:
                results.append(p)
        return results

    elif resolver_type == "infer:psu_type":
        case_type = specs.get(source_key, "")
        if not case_type:
            return search_table(target_table, "", limit=100)
        expected = _infer_psu_type(case_type)
        return search_table(target_table, expected, limit=100)

    elif resolver_type == "drive_bay_fit":
        drive_ff = specs.get(source_key, "")
        if not drive_ff:
            return search_table(target_table, "", limit=100)
        all_products = search_table(target_table, "", limit=500)
        results = []
        for p in all_products:
            bays = _extract_int(p.specs.get("internal_35_bays"))
            if bays is not None and bays >= 1:
                results.append(p)
        return results

    return search_table(target_table, "", limit=100)


@app.get("/compatible-products", response_model=SearchResponse)
def compatible_products(
    from_table: str = Query(...),
    from_id: int = Query(...),
    target_table: str = Query(...),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context_cpu_id: Optional[int] = Query(None),
    context_gpu_id: Optional[int] = Query(None),
):
    if from_table not in ALL_TABLES or target_table not in ALL_TABLES:
        raise HTTPException(status_code=404, detail=f"Invalid table. Valid: {', '.join(ALL_TABLES)}")

    source = get_product(from_table, from_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Product not found in {from_table} with id {from_id}")

    context_cpu_tdp = None
    context_gpu_tdp = None
    if context_cpu_id is not None:
        cpu_source = get_product("CPU", context_cpu_id)
        if cpu_source:
            context_cpu_tdp = _extract_int(cpu_source.specs.get("tdp"))
    if context_gpu_id is not None:
        gpu_source = get_product("GPU", context_gpu_id)
        if gpu_source:
            context_gpu_tdp = _extract_int(gpu_source.specs.get("tdp"))

    key = (from_table, target_table)
    constraint = CONSTRAINT_MAP.get(key)

    if constraint:
        source_key, resolver_type, target_key = constraint
        all_candidates = _apply_constraint(
            source, target_table, resolver_type, source_key, target_key,
            context_cpu_tdp=context_cpu_tdp, context_gpu_tdp=context_gpu_tdp,
        )
    else:
        all_candidates = search_table(target_table, "", limit=500)

    total = len(all_candidates)
    page = all_candidates[offset: offset + limit]

    cat_name = target_table
    return SearchResponse(query="", category=cat_name, count=total, results=page)


@app.post("/suggest", response_model=SuggestResponse)
def suggest_compatible(request: SuggestRequest):
    from backend.app.services.supabase import ALL_TABLES, search_table

    if request.table not in ALL_TABLES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown table '{request.table}'",
        )

    product = get_product(request.table, request.id)
    if not product:
        raise HTTPException(
            status_code=404,
            detail=f"Product not found in {request.table} with id {request.id}",
        )

    result = SuggestResponse()
    specs = product.specs
    table_name = request.table

    if table_name == "Motherboard":
        socket = specs.get("socket", "")
        memory_type = specs.get("memory_type", "")
        form_factor = specs.get("form_factor", "")
        if socket:
            result.cpu = search_table("CPU", socket)
            result.cpu_cooler = search_table("CPU Cooler", socket)
        if memory_type:
            result.memory = search_table("Memory", memory_type)
        if form_factor:
            result.case = search_table("Case", form_factor)
        if _has_m2_slots(specs):
            result.internal_drive = search_table("Internal Drive", "NVMe")

    elif table_name == "CPU":
        socket = specs.get("socket", "")
        if socket:
            result.motherboard = search_table("Motherboard", socket)
            result.cpu_cooler = search_table("CPU Cooler", socket)

    elif table_name == "GPU":
        length = _extract_int(specs.get("length"))
        tdp = _extract_int(specs.get("tdp"))
        if length:
            cases = search_table("Case", "")
            result.case = [
                c
                for c in cases
                if _extract_int(c.specs.get("max_gpu_length"))
                and _extract_int(c.specs["max_gpu_length"]) >= length
            ]
        if tdp:
            psus = search_table("PSU", "")
            result.psu = [
                p
                for p in psus
                if _extract_int(p.specs.get("wattage"))
                and _extract_int(p.specs["wattage"]) >= tdp + 200
            ]

    elif table_name == "Case":
        motherboard_form_factor = specs.get("motherboard_form_factor", "")
        max_gpu_length = _extract_int(specs.get("max_gpu_length"))
        if motherboard_form_factor:
            result.motherboard = search_table("Motherboard", motherboard_form_factor)
        if max_gpu_length:
            gpus = search_table("GPU", "")
            result.gpu = [
                g
                for g in gpus
                if _extract_int(g.specs.get("length"))
                and _extract_int(g.specs["length"]) <= max_gpu_length
            ]

    elif table_name == "Memory":
        speed = specs.get("speed", "")
        if speed:
            result.motherboard = search_table("Motherboard", speed)

    elif table_name == "CPU Cooler":
        cpu_socket = specs.get("cpu_socket", "")
        if cpu_socket:
            result.cpu = search_table("CPU", cpu_socket)
            result.motherboard = search_table("Motherboard", cpu_socket)

    elif table_name == "PSU":
        wattage = _extract_int(specs.get("wattage"))
        if wattage:
            gpus = search_table("GPU", "")
            result.gpu = [
                g
                for g in gpus
                if _extract_int(g.specs.get("tdp"))
                and _extract_int(g.specs["tdp"]) <= wattage - 200
            ]

    elif table_name == "Internal Drive":
        form_factor = specs.get("form_factor", "")
        interface = specs.get("interface", "")
        if form_factor:
            result.motherboard = search_table("Motherboard", form_factor)
        if interface:
            result.motherboard = search_table("Motherboard", interface)

    return result


@app.get("/admin/check")
def admin_check(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    is_admin, email = check_admin_email(token)
    return {"admin": is_admin, "email": email}


@app.put("/admin/products/{table_name}/{product_id}")
def admin_update_product(
    table_name: str, product_id: int, data: dict,
    authorization: Optional[str] = Header(None)
):
    if table_name not in ALL_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table_name}'")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    client = create_authenticated_client(token)
    existing = get_product(table_name, product_id, client=client)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")
    allowed = {"price", "stock", "manufacturer", "image_urls"}
    update_payload = {k: v for k, v in data.items() if k in allowed and v is not None}
    if not update_payload:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    result = update_product(table_name, product_id, update_payload, client=client)
    if not result:
        raise HTTPException(status_code=500, detail="Update failed")
    return result


@app.delete("/admin/products/{table_name}/{product_id}")
def admin_delete_product(
    table_name: str, product_id: int,
    authorization: Optional[str] = Header(None)
):
    if table_name not in ALL_TABLES:
        raise HTTPException(status_code=404, detail=f"Unknown table '{table_name}'")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    client = create_authenticated_client(token)
    existing = get_product(table_name, product_id, client=client)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")
    ok = delete_product(table_name, product_id, client=client)
    if not ok:
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"deleted": True, "table": table_name, "id": product_id}


@app.get("/admin/orders")
def admin_list_orders(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    is_admin, _ = check_admin_email(token)
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    orders = list_all_orders()
    return {"data": orders}


ALLOWED_ORDER_STATUSES = {"pending_payment", "confirmed", "processing", "shipped", "delivered", "cancelled"}

VALID_ADMIN_TRANSITIONS = {
    "pending_payment": {"confirmed", "cancelled"},
    "confirmed": {"processing", "cancelled", "delivered"},
    "processing": {"shipped", "cancelled", "delivered"},
    "shipped": {"delivered", "cancelled"},
    "delivered": set(),
    "cancelled": set(),
}


@app.put("/admin/orders/{order_id}")
def admin_update_order(order_id: str, data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    is_admin, _ = check_admin_email(token)
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    new_status = data.get("status", "")
    if new_status not in ALLOWED_ORDER_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(sorted(ALLOWED_ORDER_STATUSES))}")

    client = create_authenticated_client(token)
    response = client.table("orders").select("*").eq("id", order_id).limit(1).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Order not found")
    order = response.data[0]
    current_status = order.get("status", "")

    if new_status not in VALID_ADMIN_TRANSITIONS.get(current_status, set()):
        raise HTTPException(status_code=400, detail=f"Cannot transition from '{current_status}' to '{new_status}'")

    snapshot = order.get("build_snapshot", {})
    cart_items = snapshot.get("cartItems", [])

    if new_status == "delivered":
        for item in cart_items:
            table_name = item.get("_tableName") or item.get("table_name")
            product_id = item.get("_productId") or item.get("product_id")
            quantity = item.get("quantity") or item.get("qty") or 1
            if table_name and product_id:
                confirm_payment_stock(table_name, product_id, quantity)

    elif new_status == "cancelled":
        for item in cart_items:
            table_name = item.get("_tableName") or item.get("table_name")
            product_id = item.get("_productId") or item.get("product_id")
            quantity = item.get("quantity") or item.get("qty") or 1
            if table_name and product_id:
                release_reservation(table_name, product_id, quantity)

    try:
        update_order_status(order_id, new_status, client=client)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update order status: {str(e)}")
    return {"success": True}


def _get_user_id_from_token(token: str) -> Optional[str]:
    try:
        user = _sb.auth.get_user(token)
        return user.user.id
    except Exception:
        return None


@app.post("/api/builds")
def create_build(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    name = data.get("name", "Untitled Build")
    components = data.get("components", {})
    client = create_authenticated_client(token)
    result = save_user_build(user_id, name, components, client=client)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to save build")
    return result


@app.get("/api/builds")
def list_builds(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    builds = list_user_builds(user_id, client=client)
    return {"data": builds}


@app.get("/api/builds/{build_id}")
def get_build(build_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    build = get_user_build(build_id, user_id, client=client)
    if not build:
        raise HTTPException(status_code=404, detail="Build not found")
    return build


@app.delete("/api/builds/{build_id}")
def delete_build(build_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    delete_user_build(build_id, user_id, client=client)
    return {"deleted": True, "id": build_id}


@app.post("/api/users/sync")
def user_sync(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    email = data.get("email", "")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    client = create_authenticated_client(token)
    result = sync_user_profile(user_id, email, client=client)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to sync profile")
    return result


@app.get("/api/users/profile")
def user_profile(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    profile = get_user_profile(user_id, client=client)
    return profile or {}


@app.put("/api/users/profile")
def update_user_profile(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    if "email" not in data:
        try:
            user = _sb.auth.get_user(token)
            if user and user.user and user.user.email:
                data["email"] = user.user.email
        except Exception:
            pass
    client = create_authenticated_client(token)
    result = upsert_user_profile(user_id, data, client=client)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to update profile")
    return result


@app.get("/api/orders")
def user_orders(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    orders = list_user_orders(user_id, client=client)
    return {"data": orders}


@app.get("/api/orders/{order_id}")
def get_single_order(order_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    order = get_user_order(order_id, user_id, client=client)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.post("/api/orders")
def place_order(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    shipping_address = data.get("shipping_address", {})
    build_snapshot = data.get("build_snapshot", {})
    total = data.get("total", 0)
    shipping_cost = data.get("shipping_cost", 0)
    delivery_method = data.get("delivery_method", "Express")

    cart_items = build_snapshot.get("cartItems", [])
    reserved_items = []
    if cart_items:
        failures = validate_cart_stock(cart_items, client=client)
        if failures:
            names = ", ".join(f.get("name", "Unknown") for f in failures)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Some items are out of stock",
                    "out_of_stock_items": failures,
                }
            )
        for item in cart_items:
            table_name = item.get("_tableName") or item.get("table_name")
            product_id = item.get("_productId") or item.get("product_id")
            quantity = item.get("quantity") or item.get("qty") or 1
            if table_name and product_id:
                if not reserve_stock(table_name, product_id, quantity):
                    for rtable, rpid, rqty in reserved_items:
                        release_reservation(rtable, rpid, rqty)
                    raise HTTPException(status_code=500, detail=f"Failed to reserve stock for {table_name} #{product_id}")
                reserved_items.append((table_name, product_id, quantity))

    try:
        result = create_order(user_id, shipping_address, build_snapshot, total, shipping_cost=shipping_cost, delivery_method=delivery_method, status="pending_payment", client=client)
        if not result:
            raise HTTPException(status_code=500, detail="Order creation returned no data — check RLS and table existence")
        return result
    except HTTPException:
        for rtable, rpid, rqty in reserved_items:
            release_reservation(rtable, rpid, rqty)
        raise
    except Exception as e:
        for rtable, rpid, rqty in reserved_items:
            release_reservation(rtable, rpid, rqty)
        raise HTTPException(status_code=500, detail=f"Order creation failed: {str(e)}")


@app.post("/api/orders/{order_id}/confirm-payment")
def confirm_payment_endpoint(order_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    order = get_user_order(order_id, user_id, client=client)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.get("status") != "pending_payment":
        raise HTTPException(status_code=400, detail="Order is not pending payment")

    update_order_status(order_id, "confirmed", client=client)
    return {"success": True, "message": "Payment confirmed successfully"}


@app.post("/api/orders/{order_id}/cancel")
def cancel_order_endpoint(order_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    order = get_user_order(order_id, user_id, client=client)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.get("status") in ("delivered", "cancelled"):
        raise HTTPException(status_code=400, detail="Can only cancel orders that are not yet delivered")

    snapshot = order.get("build_snapshot", {})
    for item in snapshot.get("cartItems", []):
        table_name = item.get("_tableName") or item.get("table_name")
        product_id = item.get("_productId") or item.get("product_id")
        quantity = item.get("quantity") or item.get("qty") or 1
        if table_name and product_id:
            release_reservation(table_name, product_id, quantity)

    update_order_status(order_id, "cancelled", client=client)
    return {"success": True, "message": "Order cancelled successfully"}


@app.get("/api/cart")
def user_cart(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    items = get_cart(user_id, client=client)
    return {"data": items}


@app.post("/api/cart/sync")
def cart_sync(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    items = data.get("items", [])
    sync_cart(user_id, items, client=client)
    return {"success": True}


@app.post("/api/cart/items")
def cart_add_item(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    result = catalog_add_item(user_id, data["table_name"], data["product_id"], data.get("quantity", 1), product_data=data.get("product_data"), client=client)
    return result


@app.delete("/api/cart/items/{item_id}")
def cart_remove_item(item_id: str, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    catalog_remove_item(user_id, item_id, client=client)
    return {"success": True}


@app.post("/api/cart/build")
def cart_set_build(data: dict, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    items = data.get("items", [])
    set_cart_build(user_id, items, client=client)
    return {"success": True}


@app.delete("/api/cart")
def cart_clear(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization[7:]
    user_id = _get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    client = create_authenticated_client(token)
    clear_cart(user_id, client=client)
    return {"success": True}


@app.post("/shipping/calculate")
def shipping_calculate(data: dict):
    origin_zip = data.get("origin_zip", "50000")
    dest_zip = data.get("dest_zip", "")
    if not dest_zip:
        raise HTTPException(status_code=400, detail="dest_zip is required")
    rates = calculate_shipping(origin_zip, dest_zip)
    return {"origin_zip": origin_zip, "dest_zip": dest_zip, "rates": rates}
