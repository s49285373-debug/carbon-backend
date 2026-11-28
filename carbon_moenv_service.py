import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pymysql

router = APIRouter(prefix="/api/carbon", tags=["Carbon MOENV Service"])

BASE_DIR = Path(__file__).resolve().parent
SEED_JSON_PATH = BASE_DIR / "moenv_cfp_seed_data.json"

DEFAULT_CATEGORY_AVERAGES: Dict[str, Dict[str, Any]] = {
    "utility": {"label": "水電瓦斯", "avg_kg_co2e": 0.474, "unit": "度"},
    "food": {"label": "餐飲食物", "avg_kg_co2e": 1.45, "unit": "份"},
    "beverage": {"label": "樂活飲品", "avg_kg_co2e": 0.32, "unit": "瓶/杯"},
    "living": {"label": "生活民生", "avg_kg_co2e": 0.78, "unit": "件"},
    "transport": {"label": "運輸交通", "avg_kg_co2e": 0.12, "unit": "公里"},
    "electronics": {"label": "電子產品", "avg_kg_co2e": 18.5, "unit": "件"},
}


def load_seed_data() -> Dict[str, Any]:
    if SEED_JSON_PATH.exists():
        with SEED_JSON_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "source": {
            "name": "環境部產品碳足跡資訊網",
            "url": "https://cfp.moenv.gov.tw/WebPage/Index.aspx",
            "announce_year": "115年",
            "confidence_threshold": 0.75,
        },
        "category_averages": DEFAULT_CATEGORY_AVERAGES,
        "emission_factors": [],
    }


def _get_mysql_connection() -> Optional[pymysql.connections.Connection]:
    host = os.getenv("MYSQL_HOST") or os.getenv("MYSQLHOST")
    port_raw = os.getenv("MYSQL_PORT") or os.getenv("MYSQLPORT")
    user = os.getenv("MYSQL_USER") or os.getenv("MYSQLUSER")
    password = os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQLPASSWORD")
    database = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQLDATABASE")

    db_url = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL")
    if (not host or not user or not database) and db_url:
        parsed = urlparse(db_url)
        host = host or parsed.hostname
        port_raw = port_raw or str(parsed.port or 3306)
        user = user or parsed.username
        password = password if password is not None else (parsed.password or "")
        database = database or parsed.path.lstrip("/")

    if not host or not user or not database:
        return None

    return pymysql.connect(
        host=host,
        port=int(port_raw or "3306"),
        user=user,
        password=password if password is not None else "",
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def get_all_emission_factors() -> List[Dict[str, Any]]:
    conn = _get_mysql_connection()
    if conn is not None:
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        factor_code,
                        factor_name,
                        category_key,
                        category_label,
                        declared_unit,
                        unit_to_base_gram,
                        is_weight_unit,
                        kg_co2e,
                        keywords,
                        source_name,
                        source_url,
                        announce_year
                    FROM emission_factors
                    ORDER BY factor_code ASC
                    """
                )
                rows = cursor.fetchall()
                if rows:
                    normalized: List[Dict[str, Any]] = []
                    for r in rows:
                        kw_raw = r.get("keywords") or ""
                        kw_list = (
                            [k.strip() for k in kw_raw.split(",") if k.strip()]
                            if isinstance(kw_raw, str)
                            else list(kw_raw)
                        )
                        normalized.append(
                            {
                                **r,
                                "unit_to_base_gram": float(r.get("unit_to_base_gram") or 1.0),
                                "is_weight_unit": int(r.get("is_weight_unit") or 0),
                                "kg_co2e": float(r.get("kg_co2e") or 0.0),
                                "keywords": kw_list,
                            }
                        )
                    return normalized
        except Exception:
            pass
        finally:
            conn.close()

    seed = load_seed_data()
    return list(seed.get("emission_factors", []))


def get_category_averages() -> Dict[str, Dict[str, Any]]:
    seed = load_seed_data()
    averages = seed.get("category_averages")
    if isinstance(averages, dict) and averages:
        return averages
    return DEFAULT_CATEGORY_AVERAGES


class EstimateRequest(BaseModel):
    item_name: str = Field(..., min_length=1, description="品項名稱")
    category_key: str = Field(default="food", description="類別鍵值")
    quantity: float = Field(default=1.0, gt=0, description="數量")
    amount_ml_or_g: Optional[float] = Field(
        default=None, gt=0, description="若為容量(ml)或重量(g)，可傳入單件數值"
    )
    factor_code: Optional[str] = Field(default=None, description="指定排放係數代碼")


@router.get("/factors")
def list_factors(category_key: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    seed = load_seed_data()
    factors = get_all_emission_factors()
    if category_key:
        factors = [f for f in factors if f.get("category_key") == category_key]

    return {
        "source": seed.get("source"),
        "category_averages": get_category_averages(),
        "total": len(factors),
        "factors": factors,
    }


@router.post("/estimate")
def estimate_item_carbon(req: EstimateRequest) -> Dict[str, Any]:
    factors = get_all_emission_factors()
    averages = get_category_averages()

    selected_factor = None
    if req.factor_code:
        for f in factors:
            if f.get("factor_code") == req.factor_code:
                selected_factor = f
                break
        if selected_factor is None:
            raise HTTPException(status_code=404, detail=f"找不到係數代碼：{req.factor_code}")

    if selected_factor is not None:
        base_kg = float(selected_factor["kg_co2e"])
        is_weight = int(selected_factor.get("is_weight_unit") or 0)
        unit_to_base = float(selected_factor.get("unit_to_base_gram") or 1.0)

        if is_weight == 1 and req.amount_ml_or_g and unit_to_base > 0:
            effective_units = (float(req.amount_ml_or_g) / unit_to_base) * float(req.quantity)
        else:
            effective_units = float(req.quantity)

        total_kg_co2e = round(base_kg * effective_units, 4)
        return {
            "item_name": req.item_name,
            "factor_code": selected_factor["factor_code"],
            "factor_name": selected_factor["factor_name"],
            "category_key": selected_factor["category_key"],
            "category_label": selected_factor["category_label"],
            "effective_units": round(effective_units, 4),
            "unit_kg_co2e": base_kg,
            "total_kg_co2e": total_kg_co2e,
            "is_estimated": False,
            "label": "已對齊碳係數",
        }

    cat_info = averages.get(req.category_key) or averages["food"]
    avg_kg = float(cat_info["avg_kg_co2e"])
    total_kg_co2e = round(avg_kg * float(req.quantity), 4)
    return {
        "item_name": req.item_name,
        "factor_code": None,
        "factor_name": f"{cat_info['label']}類別平均係數",
        "category_key": req.category_key if req.category_key in averages else "food",
        "category_label": cat_info["label"],
        "effective_units": float(req.quantity),
        "unit_kg_co2e": avg_kg,
        "total_kg_co2e": total_kg_co2e,
        "is_estimated": True,
        "label": "估算品項",
    }