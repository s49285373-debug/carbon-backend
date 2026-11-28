import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
RAW_JSON_PATH = BASE_DIR / "moenv_raw.json"
SEED_JSON_PATH = BASE_DIR / "moenv_cfp_seed_data.json"
CONFIG_JSON_PATH = BASE_DIR / "carbon_estimation_config.json"
OUT_SQL_PATH = BASE_DIR / "seed_emission_factors.sql"

EXCLUDE_KEYWORDS = (
    "廢水",
    "污水",
    "汙水",
    "處理服務",
    "混凝土",
    "水泥",
    "瀝青",
    "鋼捲",
    "鋼板",
    "型鋼",
    "鋁錠",
    "銅箔",
    "矽晶",
    "樹脂",
    "橡膠",
    "塑膠袋",
    "容器",
    "包裝材",
    "奶瓶",
    "蛋盤",
    "潤滑油",
    "燃料油",
    "裂解油",
    "工業用水",
    "自來水廠",
    "水性水泥漆",
    "水性白膠",
    "水中混凝土",
)

TARGET_KEYWORDS = (
    "米",
    "飯",
    "麵",
    "蛋",
    "雞",
    "豬",
    "牛",
    "魚",
    "蝦",
    "蛤",
    "牡蠣",
    "菜",
    "蔬",
    "果",
    "奶",
    "乳",
    "優格",
    "優酪乳",
    "豆漿",
    "茶",
    "咖啡",
    "飲",
    "果汁",
    "可樂",
    "水",
    "麵包",
    "餅乾",
    "便當",
    "油",
    "糖",
    "衛生紙",
    "洗碗",
    "洗衣",
    "沐浴",
    "洗髮",
    "公車",
    "捷運",
    "高鐵",
)


def pick_field(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        val = lower_map.get(str(k).lower())
        if val is not None:
            return val
    return default


def get_split_ratios() -> Tuple[float, float]:
    if CONFIG_JSON_PATH.exists():
        try:
            cfg = json.loads(CONFIG_JSON_PATH.read_text(encoding="utf-8"))
            split = cfg.get("carbon_split") or {}
            d = float(split.get("direct_ratio", 0.70))
            h = float(split.get("hidden_ratio", 0.30))
            if d + h > 0:
                return d, h
        except Exception:
            pass
    return 0.70, 0.30


def is_valid_daily_item(name: str) -> bool:
    if not name:
        return False
    if any(bad in name for bad in EXCLUDE_KEYWORDS):
        return False
    if name.strip() == "水":
        return False
    return any(k in name for k in TARGET_KEYWORDS)


def map_category(name: str, scope3: str = "") -> Tuple[str, str]:
    s = f"{name} {scope3}".lower()
    if any(k in name for k in ("茶", "咖啡", "飲", "奶", "乳", "優酪乳", "豆漿", "果汁", "可樂", "水")):
        return "beverage", "樂活飲品"
    if any(
        k in name
        for k in (
            "米", "飯", "麵", "蛋", "雞", "豬", "牛", "魚", "蝦", "蛤", "牡蠣",
            "菜", "蔬", "果", "優格", "麵包", "餅乾", "便當", "油", "糖",
        )
    ):
        return "food", "餐飲食物"
    if any(k in s for k in ("運輸", "公車", "捷運", "高鐵", "客運", "物流")):
        return "transport", "運輸交通"
    if any(k in s for k in ("電", "水", "瓦斯", "能源")):
        return "utility", "水電瓦斯"
    return "living", "生活民生"


def parse_unit(unit_raw: str) -> Tuple[str, float, int]:
    u = (unit_raw or "件").strip()
    low = u.lower()
    if "公噸" in u or low in ("mt", "t") or "(mt)" in low:
        return u, 1000000.0, 1
    if "公斤" in u or low == "kg" or "(kg)" in low:
        return u, 1000.0, 1
    if "公克" in u or low == "g" or "(g)" in low:
        return u, 100.0, 1
    if "公升" in u or low == "l" or "(l)" in low:
        return u, 1000.0, 0
    if "毫升" in u or low == "ml" or "(ml)" in low:
        return u, 1.0, 0
    return u, 1.0, 0


def build_keywords(name: str) -> List[str]:
    clean = re.sub(r"[（）()／/、,，\-\d\.]+", " ", name)
    kws = [name.strip()] + [w.strip() for w in clean.split() if len(w.strip()) >= 1]
    out: List[str] = []
    for k in kws:
        if k and k not in out:
            out.append(k)
    return out[:8]


def main() -> None:
    if not RAW_JSON_PATH.exists():
        raise FileNotFoundError(f"找不到 {RAW_JSON_PATH}")

    direct_ratio, hidden_ratio = get_split_ratios()
    raw_data = json.loads(RAW_JSON_PATH.read_text(encoding="utf-8"))
    if isinstance(raw_data, list):
        rows = raw_data
    elif isinstance(raw_data, dict):
        rows = raw_data.get("records") or raw_data.get("data") or []
    else:
        rows = []

    by_name: Dict[str, Dict[str, Any]] = {}
    for idx, r in enumerate(rows, start=1):
        if not isinstance(r, dict):
            continue
        name = str(pick_field(r, "name", "Name", default="")).strip()
        if not is_valid_daily_item(name):
            continue

        coe_raw = str(pick_field(r, "coe", "Coe", default="0"))
        try:
            coe = float(re.sub(r"[^\d\.]", "", coe_raw) or 0)
        except ValueError:
            coe = 0.0
        if coe <= 0:
            continue

        year = str(pick_field(r, "announcementyear", "AnnouncementYear", default="")).strip()
        unit_raw = str(pick_field(r, "unit", "Unit", default="件")).strip()
        scope3_raw = str(pick_field(r, "scpoe3_iso", "Scpoe3_iso", default="")).strip()

        declared_unit, unit_to_base, is_weight = parse_unit(unit_raw)
        cat_key, cat_label = map_category(name, scope3_raw)

        item: Dict[str, Any] = {
            "factor_code": f"MOENV-{idx:04d}",
            "factor_name": name,
            "category_key": cat_key,
            "category_label": cat_label,
            "declared_unit": declared_unit,
            "unit_to_base_gram": unit_to_base,
            "is_weight_unit": is_weight,
            "kg_co2e": round(coe, 6),
            "direct_kg_co2e": round(coe * direct_ratio, 6),
            "hidden_supply_kg_co2e": round(coe * hidden_ratio, 6),
            "keywords": build_keywords(name),
            "source": f"環境部產品碳足跡係數庫（CFP_P_02，{year or '最新公告'}）",
            "announce_year": year or "最新公告",
            "_year": year,
        }

        if name not in by_name or year >= by_name[name]["_year"]:
            by_name[name] = item

    factors: List[Dict[str, Any]] = []
    for v in by_name.values():
        v.pop("_year", None)
        factors.append(v)

    existing_seed: Dict[str, Any] = {}
    if SEED_JSON_PATH.exists():
        try:
            existing_seed = json.loads(SEED_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing_seed = {}

    seed_payload = {
        "source": existing_seed.get("source")
        or {
            "name": "環境部環境資料開放平臺（CFP_P_02）",
            "url": "https://data.moenv.gov.tw/",
            "announce_year": "最新公告",
            "confidence_threshold": 0.75,
        },
        "category_averages": existing_seed.get("category_averages") or {},
        "emission_factors": factors,
    }
    SEED_JSON_PATH.write_text(
        json.dumps(seed_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sql_rows = []
    esc = lambda s: str(s).replace("'", "''")
    for f in factors:
        kw = esc(",".join(f["keywords"]))
        sql_rows.append(
            f"('{esc(f['factor_code'])}', '{esc(f['factor_name'])}', '{esc(f['category_key'])}', "
            f"'{esc(f['category_label'])}', '{esc(f['declared_unit'])}', {f['unit_to_base_gram']}, "
            f"{f['is_weight_unit']}, {f['kg_co2e']}, {f['direct_kg_co2e']}, {f['hidden_supply_kg_co2e']}, '{kw}')"
        )

    sql_text = (
        "SET NAMES utf8mb4;\n"
        "INSERT INTO emission_factors (\n"
        "  factor_code, factor_name, category_key, category_label,\n"
        "  declared_unit, unit_to_base_gram, is_weight_unit,\n"
        "  kg_co2e, direct_kg_co2e, hidden_supply_kg_co2e, keywords\n"
        ") VALUES\n"
        + ",\n".join(sql_rows)
        + "\nON DUPLICATE KEY UPDATE "
        "kg_co2e=VALUES(kg_co2e), direct_kg_co2e=VALUES(direct_kg_co2e), "
        "hidden_supply_kg_co2e=VALUES(hidden_supply_kg_co2e), keywords=VALUES(keywords);\n"
    )
    OUT_SQL_PATH.write_text(sql_text, encoding="utf-8")
    print(f"完成！共整理 {len(factors)} 筆係數 -> {SEED_JSON_PATH.name} 與 {OUT_SQL_PATH.name}")


if __name__ == "__main__":
    main()