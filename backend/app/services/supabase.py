from typing import List

from supabase import create_client, Client

from backend.app.config import get_settings
from backend.app.schemas import ComponentMatch


_supabase_client: Client | None = None


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


def build_search_string(ocr_text: str) -> str:
    # Simple normalization to help searching.
    text = ocr_text.upper()
    return " ".join(text.split())


def search_components(
    category: str,
    ocr_text: str,
    limit: int = 5,
) -> List[ComponentMatch]:
    """
    Query Supabase `components_catalog` table using category + OCR text.
    Strategy:
      1) Exact-ish match on model/aliases with ILIKE.
      2) Fallback to a generic text search on `search_text`.
    """
    sb = get_supabase()
    norm = build_search_string(ocr_text)

    # Step 1: direct match on model or aliases (if you model aliases as text[]).
    query = (
        sb.table("components_catalog")
        .select("*")
        .eq("category", category)
        .ilike("model", f"%{norm}%")
        .limit(limit)
    )
    resp = query.execute()
    rows = resp.data or []

    if not rows:
        # Step 2: fallback to fuzzy/FTS-style search on a denormalized text column.
        # This assumes you created a `search_text` column and maybe an index on it.
        query = (
            sb.table("components_catalog")
            .select("*")
            .eq("category", category)
            .ilike("search_text", f"%{norm}%")
            .limit(limit)
        )
        resp = query.execute()
        rows = resp.data or []

    matches: List[ComponentMatch] = []
    for r in rows:
        matches.append(
            ComponentMatch(
                id=r.get("id"),
                category=r.get("category", category),
                brand=r.get("brand"),
                model=r.get("model"),
                score=None,
                extra={k: v for k, v in r.items() if k not in {"id", "category", "brand", "model"}},
            )
        )
    return matches

