import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "app" / "data"
ALIASES_PATH = DATA_DIR / "product_aliases.json"
DISHES_PATH = DATA_DIR / "dish_recipes.json"


def _load_json_safe(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8-sig").strip()
        if not raw:
            return default
        data = json.loads(raw)
        return data if isinstance(data, dict) else default
    except Exception:
        return default


def load_product_aliases() -> Dict[str, Any]:
    return _load_json_safe(
        ALIASES_PATH,
        {
            "strip_prefixes": [],
            "strip_suffixes": [],
            "strip_keywords": [],
            "unit_conversions": {},
            "aliases": [],
        },
    )


def load_dish_recipes() -> Dict[str, Any]:
    return _load_json_safe(DISHES_PATH, {"dishes": []})


def clean_raw_item_name(raw_name: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or load_product_aliases()
    text = (raw_name or "").strip()
    if not text:
        return ""

    for pat in cfg.get("strip_prefixes") or []:
        try:
            text = re.sub(pat, "", text).strip()
        except re.error:
            pass

    for kw in cfg.get("strip_keywords") or []:
        if kw:
            text = text.replace(str(kw), " ")

    for pat in cfg.get("strip_suffixes") or []:
        try:
            text = re.sub(pat, "", text).strip()
        except re.error:
            pass

    text = re.sub(r"\s+", " ", text).strip()
    return text or raw_name.strip()


def convert_unit_to_factor(quantity: float, unit: str, cfg: Optional[Dict[str, Any]] = None) -> float:
    cfg = cfg or load_product_aliases()
    conv_map: Dict[str, Any] = cfg.get("unit_conversions") or {}
    u_clean = (unit or "").strip()
    u_lower = u_clean.lower()

    factor: Optional[float] = None
    if u_clean in conv_map:
        factor = float(conv_map[u_clean])
    elif u_lower in conv_map:
        factor = float(conv_map[u_lower])
    else:
        for k, v in conv_map.items():
            if k.lower() in u_lower or u_clean in k:
                factor = float(v)
                break

    if factor is None:
        if "公斤" in u_lower or u_lower == "kg":
            factor = 1.0
        elif "公克" in u_lower or u_lower in ("g", "ml", "毫升"):
            factor = 0.001
        elif "公升" in u_lower or u_lower == "l":
            factor = 1.0
        elif "台斤" in u_lower or "斤" in u_lower:
            factor = 0.6
        elif "兩" in u_lower:
            factor = 0.0375
        else:
            factor = 1.0

    return float(quantity) * factor


def normalize_item_query(raw_name: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = cfg or load_product_aliases()
    cleaned_name = clean_raw_item_name(raw_name, cfg)
    lower_target = f"{raw_name} {cleaned_name}".lower()

    best_alias: Optional[Dict[str, Any]] = None
    best_hits = 0

    for entry in cfg.get("aliases") or []:
        keywords = entry.get("keywords") or []
        hits = sum(1 for kw in keywords if kw and str(kw).lower() in lower_target)
        if hits > best_hits:
            best_hits = hits
            best_alias = entry

    if best_alias is not None:
        canonical_name = str(best_alias.get("canonical_name") or cleaned_name)
        search_query = str(best_alias.get("search_query") or f"{canonical_name} {cleaned_name}").strip()
        unit_scale = float(best_alias.get("unit_to_kg_or_l") or 1.0)
        return {
            "raw_name": raw_name,
            "cleaned_name": cleaned_name,
            "canonical_name": canonical_name,
            "search_query": search_query,
            "unit_scale": unit_scale,
            "alias_matched": True,
        }

    return {
        "raw_name": raw_name,
        "cleaned_name": cleaned_name,
        "canonical_name": cleaned_name,
        "search_query": cleaned_name,
        "unit_scale": 1.0,
        "alias_matched": False,
    }


def lookup_dish_recipe(meal_name: str) -> Optional[Dict[str, Any]]:
    data = load_dish_recipes()
    dishes = data.get("dishes") or []
    target = (meal_name or "").strip().lower()
    if not target:
        return None

    for dish in dishes:
        d_name = str(dish.get("dish_name") or "").strip()
        if not d_name:
            continue
        variants = [part.strip().lower() for part in re.split(r"[／/]", d_name) if part.strip()]
        if target == d_name.lower() or target in variants or any(v in target for v in variants):
            return dish

    return None