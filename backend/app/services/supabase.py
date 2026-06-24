import json
import re
from typing import Dict, List, Optional

from supabase import create_client

from backend.app.config import get_settings
from backend.app.schemas import SpecProduct

_settings = get_settings()
_supabase = create_client(_settings.SUPABASE_URL, _settings.SUPABASE_KEY)


def create_authenticated_client(token: str):
    """Creates a Supabase client authenticated with a user's JWT token.
    Used for admin endpoints where RLS policies need the user's identity."""
    client = create_client(_settings.SUPABASE_URL, _settings.SUPABASE_KEY)
    client.postgrest.auth(token)
    return client

CATEGORY_TABLE_MAP: Dict[str, str] = {
    "CPU": "CPU",
    "GPU": "GPU",
    "Motherboard": "Motherboard",
    "Case": "Case",
    "Memory": "Memory",
    "PSU": "PSU",
    "CPU Cooler": "CPU Cooler",
    "Internal Drive": "Internal Drive",
}

TABLE_CATEGORY_MAP: Dict[str, str] = {v: k for k, v in CATEGORY_TABLE_MAP.items()}

ALL_TABLES: List[str] = list(CATEGORY_TABLE_MAP.values())


def row_to_spec_product(table_name: str, row: dict) -> SpecProduct:
    spec_keys = {"id", "name", "price", "image_urls", "manufacturer", "stock"}
    specs = {k: v for k, v in row.items() if k not in spec_keys and v is not None}

    image_urls = row.get("image_urls", "")
    if isinstance(image_urls, list):
        image_urls = json.dumps(image_urls)

    raw_stock = row.get("stock") or 0
    effective_stock = max(0, raw_stock)

    return SpecProduct(
        id=row.get("id", 0),
        table_name=table_name,
        name=row.get("name", ""),
        price=row.get("price", ""),
        image_urls=image_urls,
        manufacturer=row.get("manufacturer", ""),
        stock=effective_stock,
        specs=specs,
    )


def _parse_price(price_str: str) -> float:
    if not price_str:
        return 0.0
    cleaned = price_str.replace("$", "").replace("£", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def search_table(
    table_name: str,
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "default",
    in_stock_first: bool = False,
    manufacturer: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    filters: Optional[dict] = None,
) -> List[SpecProduct]:
    if table_name not in ALL_TABLES:
        return []

    sb = _supabase.table(table_name).select("*")

    if query.strip():
        sb = sb.ilike("name", f"%{query.strip()}%")

    if manufacturer:
        sb = sb.ilike("manufacturer", manufacturer)

    if price_min is not None:
        sb = sb.gte("price", price_min)

    if price_max is not None:
        sb = sb.lte("price", price_max)

    if filters:
        for col, val in filters.items():
            if isinstance(val, dict):
                if "min" in val:
                    sb = sb.gte(col, val["min"])
                if "max" in val:
                    sb = sb.lte(col, val["max"])
            else:
                sb = sb.ilike(col, f"%{val}%")

    if in_stock_first:
        sb = sb.order("stock", desc=True, nullsfirst=False)

    if sort_by == "price_asc":
        sb = sb.order("price", desc=False)
    elif sort_by == "price_desc":
        sb = sb.order("price", desc=True)
    elif sort_by == "name_asc":
        sb = sb.order("name", desc=False)
    elif sort_by == "name_desc":
        sb = sb.order("name", desc=True)

    sb = sb.range(offset, offset + limit - 1)

    response = sb.execute()
    return [row_to_spec_product(table_name, r) for r in (response.data or [])]


def search_table_by_column(
    table_name: str, column: str, query: str = "", limit: int = 100
) -> List[SpecProduct]:
    if table_name not in ALL_TABLES:
        return []
    sb = _supabase.table(table_name).select("*")
    if query.strip():
        sb = sb.ilike(column, f"%{query.strip()}%")
    sb = sb.limit(limit)
    response = sb.execute()
    return [row_to_spec_product(table_name, r) for r in (response.data or [])]


def search_table_count(
    table_name: str,
    query: str = "",
    manufacturer: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    filters: Optional[dict] = None,
) -> int:
    if table_name not in ALL_TABLES:
        return 0
    sb = _supabase.table(table_name).select("id", count="exact")
    if query.strip():
        sb = sb.ilike("name", f"%{query.strip()}%")
    if manufacturer:
        sb = sb.ilike("manufacturer", manufacturer)
    if price_min is not None:
        sb = sb.gte("price", price_min)
    if price_max is not None:
        sb = sb.lte("price", price_max)
    if filters:
        for col, val in filters.items():
            if isinstance(val, dict):
                if "min" in val:
                    sb = sb.gte(col, val["min"])
                if "max" in val:
                    sb = sb.lte(col, val["max"])
            else:
                sb = sb.ilike(col, f"%{val}%")
    response = sb.limit(0).execute()
    return response.count or 0


def search_all_tables_count(
    query: str = "",
    manufacturer: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    filters: Optional[dict] = None,
) -> int:
    return sum(search_table_count(t, query, manufacturer=manufacturer, price_min=price_min, price_max=price_max, filters=filters) for t in ALL_TABLES)


def search_all_tables(
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "default",
    in_stock_first: bool = False,
    manufacturer: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    filters: Optional[dict] = None,
) -> List[SpecProduct]:
    results: List[SpecProduct] = []

    kwargs = dict(manufacturer=manufacturer, price_min=price_min, price_max=price_max, filters=filters)

    if sort_by != "default":
        per_table = max(1, (limit + offset) // len(ALL_TABLES))
        for table_name in ALL_TABLES:
            table_results = search_table(table_name, query, limit=per_table, offset=0, sort_by=sort_by, in_stock_first=in_stock_first, **kwargs)
            results.extend(table_results)
        return results[offset : offset + limit]

    if query.strip():
        for table_name in ALL_TABLES:
            table_results = search_table(table_name, query, limit=limit, in_stock_first=in_stock_first, **kwargs)
            results.extend(table_results)
        return results[:limit]

    per_table = max(20, limit + offset)
    for table_name in ALL_TABLES:
        table_results = search_table(table_name, "", limit=per_table, in_stock_first=in_stock_first, **kwargs)
        results.extend(table_results)
    return results[offset : offset + limit]


def search_all_categories(
    query: str = "",
    limit: int = 20,
    offset: int = 0,
    sort_by: str = "default",
    in_stock_first: bool = False,
    manufacturer: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    filters: Optional[dict] = None,
) -> List[SpecProduct]:
    return search_all_tables(query=query, limit=limit, offset=offset, sort_by=sort_by, in_stock_first=in_stock_first, manufacturer=manufacturer, price_min=price_min, price_max=price_max, filters=filters)


def extract_search_terms(ocr_text: str) -> List[str]:
    if not ocr_text or not ocr_text.strip():
        return []

    text_upper = ocr_text.upper()
    words = re.findall(r"[A-Za-z0-9]+", text_upper)

    meaningful = []
    for w in words:
        w_stripped = w.strip()
        if len(w_stripped) >= 3 and (
            any(c.isdigit() for c in w_stripped) or w_stripped.isalpha()
        ):
            meaningful.append(w_stripped)

    meaningful.sort(key=lambda t: (
        0 if t.isdigit() else (1 if any(c.isdigit() for c in t) else 2),
        -len(t),
    ))

    return meaningful


def _normalize_ocr_text(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    s = re.sub(r"\b[a-z]{1,2}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def search_components_ocr(
    category: str,
    ocr_text: str,
    limit: int = 5,
) -> List[SpecProduct]:
    table_name = CATEGORY_TABLE_MAP.get(category)
    if not table_name:
        print(f"Unknown category: {category}")
        return []

    clean = _normalize_ocr_text(ocr_text)
    results = search_table(table_name, clean, limit=limit)

    if not results:
        terms = extract_search_terms(clean)
        broad = []
        for term in terms:
            broad = search_table(table_name, term, limit=80)
            if broad:
                break

        if broad:
            tl = [t.lower() for t in terms]
            scores = {}
            for p in broad:
                nl = p.name.lower()
                s = sum(1 for t in tl if t in nl)
                if s:
                    scores[p.id] = scores.get(p.id, 0) + s
            broad.sort(key=lambda p: -scores.get(p.id, 0))
            results = broad[:limit]

    if not results:
        results = search_table(table_name, "", limit=limit)

    return results[:limit]


def get_product(table_name: str, product_id: int, client=None) -> Optional[SpecProduct]:
    if table_name not in ALL_TABLES:
        return None
    sb = client or _supabase
    response = sb.table(table_name).select("*").eq("id", product_id).limit(1).execute()
    if not response.data:
        return None
    return row_to_spec_product(table_name, response.data[0])


def validate_cart_stock(cart_items: list, client=None) -> list[dict]:
    failures = []
    for item in cart_items:
        table_name = item.get("_tableName") or item.get("table_name")
        product_id = item.get("_productId") or item.get("product_id")
        quantity = item.get("quantity") or item.get("qty") or 1
        if not table_name or not product_id:
            continue
        product = get_product(table_name, product_id, client=client)
        if product is None:
            failures.append({
                "name": item.get("name", f"{table_name} #{product_id}"),
                "table_name": table_name,
                "product_id": product_id,
                "available": 0,
                "requested": quantity,
                "reason": "Product not found",
            })
            continue
        reserved = product.specs.get("stock_reserved") or 0
        available = max(0, (product.stock or 0) - reserved)
        if available < quantity:
            failures.append({
                "name": product.name,
                "table_name": table_name,
                "product_id": product_id,
                "available": available,
                "requested": quantity,
            })
    return failures


def reserve_stock(table_name: str, product_id: int, quantity: int, client=None) -> bool:
    if table_name not in ALL_TABLES:
        return False
    try:
        result = _supabase.rpc("reserve_product_stock", {"tbl": table_name, "pid": product_id, "qty": quantity}).execute()
        return bool(result.data)
    except Exception:
        return False


def confirm_payment_stock(table_name: str, product_id: int, quantity: int, client=None) -> bool:
    if table_name not in ALL_TABLES:
        return False
    try:
        result = _supabase.rpc("confirm_product_stock", {"tbl": table_name, "pid": product_id, "qty": quantity}).execute()
        return bool(result.data)
    except Exception:
        return False


def release_reservation(table_name: str, product_id: int, quantity: int, client=None) -> bool:
    if table_name not in ALL_TABLES:
        return False
    try:
        result = _supabase.rpc("release_product_stock", {"tbl": table_name, "pid": product_id, "qty": quantity}).execute()
        return bool(result.data)
    except Exception:
        return False


def update_order_status(order_id: str, status: str, client=None) -> bool:
    sb = client or _supabase
    sb.table("orders").update({"status": status}).eq("id", order_id).execute()
    return True


def get_category_info_list() -> List[dict]:
    result = []
    for cat_name, table_name in CATEGORY_TABLE_MAP.items():
        response = _supabase.table(table_name).select("id", count="exact").limit(0).execute()
        result.append(
            {
                "name": cat_name,
                "table_name": table_name,
                "product_count": response.count or 0,
            }
        )
    return result


def update_product(table_name: str, product_id: int, data: dict, client=None) -> Optional[SpecProduct]:
    if table_name not in ALL_TABLES:
        return None
    sb = client or _supabase
    sb.table(table_name).update(data).eq("id", product_id).execute()
    return get_product(table_name, product_id, client=client)


def delete_product(table_name: str, product_id: int, client=None) -> bool:
    if table_name not in ALL_TABLES:
        return False
    sb = client or _supabase
    sb.table(table_name).delete().eq("id", product_id).execute()
    return True


def check_admin_email(token: str) -> tuple:
    """Verifies a JWT token and checks if the user's email is in the admin_users table.
    Returns (is_admin: bool, email: str | None)."""
    try:
        user = _supabase.auth.get_user(token)
        email = user.user.email
        if not email:
            print(f"[admin] No email in token for user {user.user.id}")
            return False, None
        response = _supabase.table("admin_users").select("email").eq("email", email).limit(1).execute()
        is_admin = len(response.data or []) > 0
        if not is_admin:
            print(f"[admin] Email '{email}' not found in admin_users")
        return is_admin, email
    except Exception as e:
        print(f"[admin] check_admin_email error: {e}")
        return False, None


def save_user_build(user_id: str, name: str, components: dict, client=None) -> Optional[dict]:
    sb = client or _supabase
    data = {"user_id": user_id, "name": name, "components": components}
    response = sb.table("user_builds").insert(data).execute()
    return response.data[0] if response.data else None


def list_user_builds(user_id: str, client=None) -> list[dict]:
    sb = client or _supabase
    response = sb.table("user_builds").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return response.data or []


def get_user_build(build_id: str, user_id: str, client=None) -> Optional[dict]:
    sb = client or _supabase
    response = sb.table("user_builds").select("*").eq("id", build_id).eq("user_id", user_id).limit(1).execute()
    return response.data[0] if response.data else None


def delete_user_build(build_id: str, user_id: str, client=None) -> bool:
    sb = client or _supabase
    sb.table("user_builds").delete().eq("id", build_id).eq("user_id", user_id).execute()
    return True


def get_manufacturers() -> Dict[str, List[str]]:
    result = {}
    for cat_name, table_name in CATEGORY_TABLE_MAP.items():
        response = _supabase.table(table_name).select("manufacturer").execute()
        brands = set()
        for r in response.data or []:
            m = (r.get("manufacturer") or "").strip()
            if m:
                brands.add(m)
        result[cat_name] = sorted(brands)
    return result


COLUMN_LABELS = {
    "manufacturer": "Brand",
    "socket": "Socket",
    "microarchitecture": "Microarchitecture",
    "core_family": "Core Family",
    "series": "Series",
    "integrated_graphics": "Integrated Graphics",
    "ecc_support": "ECC Support",
    "includes_cooler": "Includes Cooler",
    "packaging": "Packaging",
    "lithography": "Lithography",
    "smt": "SMT",
    "core_count": "Core Count",
    "thread_count": "Thread Count",
    "core_clock": "Core Clock",
    "boost_clock": "Boost Clock",
    "l2_cache": "L2 Cache",
    "l3_cache": "L3 Cache",
    "tdp": "TDP",
    "max_memory": "Max Memory",
    "chipset": "Chipset",
    "memory_type": "Memory Type",
    "interface": "Interface",
    "frame_sync": "Frame Sync",
    "cooling": "Cooling",
    "external_power": "External Power",
    "memory": "VRAM",
    "length": "Length",
    "slot_width": "Slot Width",
    "displayport_outputs": "DisplayPort",
    "hdmi_outputs": "HDMI",
    "form_factor": "Form Factor",
    "memory_max": "Max Memory",
    "memory_slots": "Memory Slots",
    "memory_speed": "Memory Speed",
    "pcie_x16_slots": "PCIe x16 Slots",
    "m2_slots": "M.2 Slots",
    "sata_ports": "SATA Ports",
    "ethernet": "Ethernet",
    "onboard_video": "Onboard Video",
    "usb_headers": "USB Headers",
    "wifi": "WiFi",
    "raid": "RAID Support",
    "back_connect": "Back Connect",
    "ecc_registered": "ECC / Registered",
    "heat_spreader": "Heat Spreader",
    "timing": "Timing",
    "speed": "Speed",
    "modules": "Modules",
    "price_per_gb": "Price per GB",
    "first_word_latency": "First Word Latency",
    "cas_latency": "CAS Latency",
    "voltage": "Voltage",
    "type": "Type",
    "efficiency": "Efficiency Rating",
    "modular": "Modular",
    "fanless": "Fanless",
    "wattage": "Wattage",
    "atx_4pin": "ATX 4-pin",
    "eps_8pin": "EPS 8-pin",
    "pcie_16pin": "PCIe 16-pin",
    "pcie_8pin": "PCIe 8-pin",
    "pcie_6plus2": "PCIe 6+2",
    "sata": "SATA Connectors",
    "molex": "Molex Connectors",
    "capacity": "Capacity",
    "nvme": "NVMe",
    "full_disk_write_throughput": "Disk Write Throughput",
    "random_read_qd1": "Random Read (QD1)",
    "random_read_qd32": "Random Read (QD32)",
    "random_write_qd1": "Random Write (QD1)",
    "random_write_qd32": "Random Write (QD32)",
    "seq_read_qd1": "Seq Read (QD1)",
    "seq_read_qd4": "Seq Read (QD4)",
    "seq_write_qd1": "Seq Write (QD1)",
    "seq_write_qd4": "Seq Write (QD4)",
    "fan_rpm": "Fan RPM",
    "noise_level": "Noise Level",
    "height": "Height",
    "cpu_socket": "CPU Socket",
    "water_cooled": "Water Cooled",
    "power_supply": "Power Supply",
    "side_panel": "Side Panel",
    "psu_shroud": "PSU Shroud",
    "motherboard_form_factor": "Mobo Form Factor",
    "max_gpu_length": "Max GPU Length",
    "drive_bays": "Drive Bays",
    "expansion_slots": "Expansion Slots",
    "included_fans": "Included Fans",
    "volume": "Volume",
}

CATEGORY_FILTERS = {
    "CPU": {
        "checkbox": ["manufacturer", "socket", "microarchitecture", "core_family", "series", "integrated_graphics", "ecc_support", "includes_cooler", "lithography", "smt"],
        "range": ["core_count", "thread_count", "tdp"],
    },
    "GPU": {
        "checkbox": ["manufacturer", "chipset", "memory_type", "frame_sync", "cooling", "external_power"],
        "range": ["memory", "core_clock", "boost_clock", "length", "tdp", "slot_width"],
    },
    "Motherboard": {
        "checkbox": ["manufacturer", "socket", "form_factor", "chipset", "memory_type", "ethernet", "wifi"],
        "range": ["memory_max", "memory_slots", "m2_slots", "sata_ports", "pcie_x16_slots"],
    },
    "Memory": {
        "checkbox": ["manufacturer", "form_factor", "ecc_registered", "heat_spreader"],
        "range": ["speed", "modules", "cas_latency", "voltage"],
    },
    "PSU": {
        "checkbox": ["manufacturer", "type", "efficiency", "modular", "fanless"],
        "range": ["wattage", "length"],
    },
    "Internal Drive": {
        "checkbox": ["manufacturer", "type", "form_factor", "interface", "nvme"],
        "range": ["capacity"],
    },
    "CPU Cooler": {
        "checkbox": ["manufacturer", "water_cooled", "fanless"],
        "range": ["fan_rpm", "noise_level", "height"],
    },
    "Case": {
        "checkbox": ["manufacturer", "type", "power_supply", "side_panel", "psu_shroud", "motherboard_form_factor"],
        "range": ["max_gpu_length", "expansion_slots", "included_fans", "volume"],
    },
}


def get_filters(category: str) -> dict:
    table_name = CATEGORY_TABLE_MAP.get(category)
    if not table_name:
        return {"checkbox": {}, "range": {}}

    spec = CATEGORY_FILTERS.get(category, {})
    result = {"checkbox": {}, "range": {}}

    for col in spec.get("checkbox", []):
        response = _supabase.table(table_name).select(col).execute()
        values = set()
        for r in response.data or []:
            v = (r.get(col) or "").strip()
            if v and v.lower() not in ("none", "n/a", ""):
                values.add(v)
        if values:
            result["checkbox"][col] = {
                "label": COLUMN_LABELS.get(col, col.replace("_", " ").title()),
                "values": sorted(values),
            }

    for col in spec.get("range", []):
        try:
            min_resp = _supabase.table(table_name).select(col).order(col).limit(1).execute()
            max_resp = _supabase.table(table_name).select(col).order(col, desc=True).limit(1).execute()
            min_val = min_resp.data[0][col] if min_resp.data else None
            max_val = max_resp.data[0][col] if max_resp.data else None
            if min_val is not None and max_val is not None:
                result["range"][col] = {
                    "label": COLUMN_LABELS.get(col, col.replace("_", " ").title()),
                    "min": str(min_val),
                    "max": str(max_val),
                }
        except Exception:
            pass

    return result


def sync_user_profile(user_id: str, email: str, client=None) -> Optional[dict]:
    sb = client or _supabase
    data = {"user_id": user_id, "email": email}
    response = sb.table("users").upsert(data).execute()
    return response.data[0] if response.data else None


def get_user_profile(user_id: str, client=None) -> Optional[dict]:
    sb = client or _supabase
    response = sb.table("users").select("*").eq("user_id", user_id).limit(1).execute()
    return response.data[0] if response.data else None


def upsert_user_profile(user_id: str, data: dict, client=None) -> Optional[dict]:
    sb = client or _supabase
    payload = {**data, "user_id": user_id, "email": data.get("email", ""), "updated_at": "now()"}
    response = sb.table("users").upsert(payload).execute()
    return response.data[0] if response.data else None


def create_order(user_id: str, shipping_address: dict, build_snapshot: dict, total: float, shipping_cost: float = 0, delivery_method: str = 'Express', status: str = 'pending_payment', client=None) -> Optional[dict]:
    sb = client or _supabase
    data = {
        "user_id": user_id,
        "shipping_address": shipping_address,
        "build_snapshot": build_snapshot,
        "total": total,
        "shipping_cost": shipping_cost,
        "delivery_method": delivery_method,
        "status": status,
    }
    response = sb.table("orders").insert(data).execute()
    return response.data[0] if response.data else None


def list_user_orders(user_id: str, client=None) -> list[dict]:
    sb = client or _supabase
    response = sb.table("orders").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return response.data or []


def list_all_orders(client=None) -> list[dict]:
    sb = client or _supabase
    response = sb.rpc("admin_list_all_orders").execute()
    return response.data or []


def get_user_order(order_id: str, user_id: str, client=None) -> Optional[dict]:
    sb = client or _supabase
    response = sb.table("orders").select("*").eq("id", order_id).eq("user_id", user_id).limit(1).execute()
    return response.data[0] if response.data else None


def get_cart(user_id: str, client=None) -> list[dict]:
    sb = client or _supabase
    response = sb.table("user_cart_items").select("*").eq("user_id", user_id).order("added_at").execute()
    return response.data or []


def sync_cart(user_id: str, items: list[dict], client=None) -> bool:
    sb = client or _supabase
    sb.table("user_cart_items").delete().eq("user_id", user_id).execute()
    if items:
        rows = []
        for item in items:
            rows.append({
                "user_id": user_id,
                "table_name": item.get("table_name", ""),
                "product_id": item.get("product_id", 0),
                "quantity": item.get("quantity", 1),
                "from_build": item.get("from_build", False),
                "product_data": item.get("product_data"),
            })
        sb.table("user_cart_items").insert(rows).execute()
    return True


def delete_cart_items(user_id: str, item_ids: list[str], client=None) -> bool:
    sb = client or _supabase
    sb.table("user_cart_items").delete().eq("user_id", user_id).in_("id", item_ids).execute()
    return True


def clear_cart(user_id: str, client=None) -> bool:
    sb = client or _supabase
    sb.table("user_cart_items").delete().eq("user_id", user_id).execute()
    return True


def set_cart_build(user_id: str, items: list[dict], client=None) -> bool:
    sb = client or _supabase
    sb.table("user_cart_items").delete().eq("user_id", user_id).eq("from_build", True).execute()
    if items:
        rows = []
        for item in items:
            rows.append({
                "user_id": user_id,
                "table_name": item.get("table_name", ""),
                "product_id": item.get("product_id", 0),
                "quantity": item.get("quantity", 1),
                "from_build": True,
                "product_data": item.get("product_data"),
            })
        sb.table("user_cart_items").insert(rows).execute()
    return True


def catalog_add_item(user_id: str, table_name: str, product_id: int, quantity: int = 1, product_data: dict = None, client=None) -> Optional[dict]:
    sb = client or _supabase
    existing = sb.table("user_cart_items").select("*").eq("user_id", user_id).eq("table_name", table_name).eq("product_id", product_id).eq("from_build", False).execute()
    if existing.data:
        new_qty = existing.data[0]["quantity"] + quantity
        update = {"quantity": new_qty}
        if product_data:
            update["product_data"] = product_data
        response = sb.table("user_cart_items").update(update).eq("id", existing.data[0]["id"]).execute()
        return response.data[0] if response.data else None
    data = {"user_id": user_id, "table_name": table_name, "product_id": product_id, "quantity": quantity, "from_build": False}
    if product_data:
        data["product_data"] = product_data
    response = sb.table("user_cart_items").insert(data).execute()
    return response.data[0] if response.data else None


def catalog_remove_item(user_id: str, item_id: str, client=None) -> bool:
    sb = client or _supabase
    sb.table("user_cart_items").delete().eq("id", item_id).eq("user_id", user_id).execute()
    return True


MALAYSIA_ZONE_MAP: Dict[str, int] = {
    "1": 1, "2": 2, "3": 3, "4": 2, "5": 1,
    "6": 1, "7": 4, "8": 5, "9": 6,
}


def _malaysia_zip_zone(zip_str: str) -> int:
    first = zip_str.strip()[:1]
    return MALAYSIA_ZONE_MAP.get(first, 0)


ZONE_DISTANCES = {
    (1, 1): 0, (1, 2): 1, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 6,
    (2, 1): 1, (2, 2): 0, (2, 3): 1, (2, 4): 2, (2, 5): 3, (2, 6): 6,
    (3, 1): 2, (3, 2): 1, (3, 3): 0, (3, 4): 2, (3, 5): 3, (3, 6): 6,
    (4, 1): 3, (4, 2): 2, (4, 3): 2, (4, 4): 0, (4, 5): 1, (4, 6): 5,
    (5, 1): 4, (5, 2): 3, (5, 3): 3, (5, 4): 1, (5, 5): 0, (5, 6): 5,
    (6, 1): 6, (6, 2): 6, (6, 3): 6, (6, 4): 5, (6, 5): 5, (6, 6): 0,
}


def _zone_distance(z1: int, z2: int) -> int:
    if z1 == 0 or z2 == 0:
        return 5
    key = (min(z1, z2), max(z1, z2))
    return ZONE_DISTANCES.get(key, 5)


def calculate_shipping(origin_zip: str, dest_zip: str) -> list[dict]:
    oz = _malaysia_zip_zone(origin_zip)
    dz = _malaysia_zip_zone(dest_zip)
    dist = _zone_distance(oz, dz)

    if dist == 0:
        standard = {"service": "Standard", "cost": 0.0, "days": "1 day"}
        express = {"service": "Express", "cost": 44.0, "days": "Same day"}
    elif dist <= 2:
        standard = {"service": "Standard", "cost": 26.36, "days": "2-3 days"}
        express = {"service": "Express", "cost": 83.56, "days": "1 day"}
    elif dist <= 4:
        standard = {"service": "Standard", "cost": 55.00, "days": "3-5 days"}
        express = {"service": "Express", "cost": 131.96, "days": "2-3 days"}
    else:
        standard = {"service": "Standard", "cost": 88.00, "days": "5-7 days"}
        express = {"service": "Express", "cost": 198.00, "days": "3-5 days"}

    return [standard, express]
