import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from fastapi import APIRouter
from openai import OpenAI
from pydantic import BaseModel, Field

from carbon_moenv_service import (
    get_all_emission_factors,
    get_carbon_split,
    get_category_averages,
    load_estimation_config,
)
from carbon_normalization import extract_measure_from_text, normalize_item_query

router = APIRouter(prefix="/api/carbon/ai", tags=["AI Carbon Aligner"])

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
FEEDBACK_FILE = BASE_DIR / "manual_corrections.json"
BATCH_SIZE = 32

FOOD_HINTS = (
    "麵",
    "杯麵",
    "泡麵",
    "來一客",
    "飯",
    "御飯糰",
    "便當",
    "蛋",
    "餃",
    "鍋貼",
    "漢堡",
    "麵包",
    "餅乾",
    "吐司",
    "三明治",
    "魚板",
    "鮮蝦",
    "雞腿",
    "排骨",
    "肉",
    "菜",
    "菇",
)
BEVERAGE_HINTS = (
    "茶",
    "紅茶",
    "綠茶",
    "烏龍",
    "奶茶",
    "茉莉茶園",
    "蘋果紅茶",
    "咖啡",
    "拿鐵",
    "美式",
    "鮮奶",
    "鮮乳",
    "豆漿",
    "綠豆沙",
    "可可",
    "飲料",
    "飲品",
    "汽水",
    "果汁",
    "ml",
    "毫升",
)

_CACHED_COLLECTION: Any = None
_CACHED_SIGNATURE: Optional[str] = None


def _get_confidence_threshold() -> float:
    cfg = load_estimation_config()
    try:
        return float(cfg.get("confidence_threshold", 0.75))
    except (TypeError, ValueError):
        return 0.75


def _load_feedback_map() -> Dict[str, Dict[str, Any]]:
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def _save_feedback_map(data: Dict[str, Dict[str, Any]]) -> None:
    with FEEDBACK_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _compute_data_signature(
    factors: List[Dict[str, Any]], feedback: Dict[str, Any], cfg: Dict[str, Any]
) -> str:
    payload = json.dumps(
        {"factors": factors, "feedback": feedback, "config": cfg},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return str(abs(hash(payload)))


def _split_carbon_units(raw: Dict[str, Any]) -> Tuple[float, float, float]:
    split = get_carbon_split()
    direct_ratio = split["direct_ratio"]
    hidden_ratio = split["hidden_ratio"]

    total_kg = float(raw.get("kg_co2e") or 0.0)
    direct_kg = (
        float(raw.get("direct_kg_co2e"))
        if raw.get("direct_kg_co2e") is not None
        else round(total_kg * direct_ratio, 6)
    )
    hidden_kg = (
        float(raw.get("hidden_supply_kg_co2e"))
        if raw.get("hidden_supply_kg_co2e") is not None
        else round(total_kg * hidden_ratio, 6)
    )
    return total_kg, direct_kg, hidden_kg


def _upsert_in_batches(
    collection: Any,
    items: List[Tuple[str, str, Dict[str, Any]]],
    batch_size: int = BATCH_SIZE,
) -> None:
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        ids = [x[0] for x in chunk]
        docs = [x[1] for x in chunk]
        metas = [x[2] for x in chunk]
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        del ids, docs, metas, chunk
        gc.collect()


def _build_chroma_collection(force_rebuild: bool = False) -> Any:
    global _CACHED_COLLECTION, _CACHED_SIGNATURE

    factors = get_all_emission_factors()
    feedback = _load_feedback_map()
    cfg = load_estimation_config()
    sig = _compute_data_signature(factors, feedback, cfg)

    if not force_rebuild and _CACHED_COLLECTION is not None and _CACHED_SIGNATURE == sig:
        return _CACHED_COLLECTION

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = "moenv_emission_factors"

    if force_rebuild:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        gc.collect()

    collection = client.get_or_create_collection(name=collection_name)

    records_to_index: List[Tuple[str, str, Dict[str, Any]]] = []

    for f in factors:
        code = str(f["factor_code"])
        kw_list = f.get("keywords") or []
        kw_text = "、".join(kw_list) if isinstance(kw_list, list) else str(kw_list)
        total_kg, direct_kg, hidden_kg = _split_carbon_units(f)

        doc_text = (
            f"係數代碼:{code} | 品項:{f['factor_name']} | "
            f"類別:{f['category_label']}({f['category_key']}) | "
            f"單位:{f.get('declared_unit') or '件'} | 關鍵字:{kw_text}"
        )
        meta = {
            "factor_code": code,
            "factor_name": str(f["factor_name"]),
            "category_key": str(f["category_key"]),
            "category_label": str(f["category_label"]),
            "kg_co2e": total_kg,
            "direct_kg_co2e": direct_kg,
            "hidden_supply_kg_co2e": hidden_kg,
            "declared_unit": str(f.get("declared_unit") or "件"),
            "unit_to_base_gram": float(f.get("unit_to_base_gram") or 1.0),
            "is_weight_unit": int(f.get("is_weight_unit") or 0),
            "source": str(f.get("source") or "環境部產品碳足跡係數庫（MOENV CFP_P_02）"),
            "is_user_feedback": False,
        }
        records_to_index.append((f"factor::{code}", doc_text, meta))

    for raw_name, fb in feedback.items():
        fb_id = f"feedback::{raw_name}"
        total_kg, direct_kg, hidden_kg = _split_carbon_units(fb)

        doc_text = (
            f"人工校正品名:{raw_name} -> 係數:{fb.get('factor_name')} "
            f"({fb.get('factor_code')}) | 類別:{fb.get('category_label')}"
        )
        meta = {
            "factor_code": str(fb.get("factor_code") or ""),
            "factor_name": str(fb.get("factor_name") or ""),
            "category_key": str(fb.get("category_key") or "food"),
            "category_label": str(fb.get("category_label") or "餐飲食物"),
            "kg_co2e": total_kg,
            "direct_kg_co2e": direct_kg,
            "hidden_supply_kg_co2e": hidden_kg,
            "declared_unit": str(fb.get("declared_unit") or "份"),
            "unit_to_base_gram": float(fb.get("unit_to_base_gram") or 1.0),
            "is_weight_unit": int(fb.get("is_weight_unit") or 0),
            "source": str(fb.get("source") or "人工校正記憶（manual_corrections.json）"),
            "is_user_feedback": True,
        }
        records_to_index.append((fb_id, doc_text, meta))

    if records_to_index:
        _upsert_in_batches(collection, records_to_index, batch_size=BATCH_SIZE)

    del records_to_index
    gc.collect()

    _CACHED_COLLECTION = collection
    _CACHED_SIGNATURE = sig
    return collection


def _compute_effective_units(
    meta: Dict[str, Any],
    quantity: float,
    amount_val: Optional[float],
    amount_unit: Optional[str],
    unit_scale: float = 1.0,
) -> float:
    is_weight = int(meta.get("is_weight_unit") or 0)
    unit_to_base = float(meta.get("unit_to_base_gram") or 1.0)
    declared_unit = str(meta.get("declared_unit") or "").strip().lower()

    if amount_val is not None and amount_val > 0 and unit_to_base > 0:
        if declared_unit in ("l", "公升") and amount_unit in ("ml", "g"):
            return (amount_val / 1000.0) * float(quantity)
        if is_weight == 1 and amount_unit in ("g", "ml"):
            return (amount_val / unit_to_base) * float(quantity)

    return float(quantity) * (float(unit_scale) if unit_scale > 0 else 1.0)


def _compute_fallback_units(
    quantity: float,
    amount_val: Optional[float],
    amount_unit: Optional[str],
    cat_key: str,
) -> float:
    """低信心退回類別平均時，避免把公克/毫升數值直接當份數放大。"""
    q = float(quantity)
    if amount_val is not None and amount_val > 0:
        if amount_unit == "g":
            # 蛋類以單顆約 50g 估算；一般食材/食物以 100g 為一份估算
            return round(amount_val / 100.0, 4) * q
        if amount_unit == "ml":
            # 飲品以 500ml 為一杯基準估算
            base_ml = 500.0 if cat_key == "beverage" else 250.0
            return round(amount_val / base_ml, 4) * q

    # 若未在字串寫出單位，但 quantity 明顯是公克/毫升數值（例如誤傳 150）
    if q >= 50.0 and cat_key in ("food", "beverage"):
        return round(q / 100.0, 4)

    return q


def _infer_category_from_name(item_name: str, default_cat: str) -> str:
    lower = item_name.lower()
    has_bev = any(h in item_name or h in lower for h in BEVERAGE_HINTS)
    has_food = any(h in item_name or h in lower for h in FOOD_HINTS)
    if has_bev and not has_food:
        return "beverage"
    if has_food and not has_bev:
        return "food"
    if "紅茶" in item_name or "綠茶" in item_name or "茶" in item_name:
        return "beverage"
    return default_cat


def _is_category_conflict(inferred_cat: str, matched_cat: str) -> bool:
    if inferred_cat == "food" and matched_cat == "beverage":
        return True
    if inferred_cat == "beverage" and matched_cat == "food":
        return True
    return False


def _llm_rerank_and_score(
    item_name: str, candidates: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    client = _get_openai_client()
    if client is None or not candidates:
        return None

    model_name = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
    prompt_payload = {
        "item_name": item_name,
        "candidates": [
            {
                "factor_code": c["factor_code"],
                "factor_name": c["factor_name"],
                "category_key": c["category_key"],
                "category_label": c["category_label"],
                "vector_similarity": c["similarity"],
            }
            for c in candidates
        ],
    }

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是台灣環境部產品碳足跡語意對齊助手。請根據輸入的發票或消費品項名稱，"
                        "從候選排放係數中選出最貼切且同類別的一筆，並給出 0.0 到 1.0 的信心分數 confidence。"
                        "若候選皆不貼切或跨大類錯誤，請給較低信心分數（< 0.75）。"
                        '僅回傳 JSON：{"factor_code": "...", "confidence": 0.0, "reason": "..."}'
                    ),
                },
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        chosen_code = str(parsed.get("factor_code") or "")
        conf = float(parsed.get("confidence") or 0.0)
        reason = str(parsed.get("reason") or "")

        for c in candidates:
            if c["factor_code"] == chosen_code:
                return {
                    **c,
                    "confidence": max(0.0, min(1.0, conf)),
                    "reason": reason or "LLM 語意重排判定",
                }
    except Exception:
        return None

    return None


def _keyword_fallback_candidates(
    query_text: str, factors_all: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    lower_name = query_text.lower()
    inferred_cat = _infer_category_from_name(query_text, "")
    matched: List[Dict[str, Any]] = []

    for f in factors_all:
        kw_list = f.get("keywords") or []
        hit = any(kw and str(kw).lower() in lower_name for kw in kw_list)
        name_hit = str(f.get("factor_name") or "").lower() in lower_name
        if hit or name_hit:
            sim = 0.88 if hit else 0.78
            if _is_category_conflict(inferred_cat, str(f.get("category_key") or "")):
                sim -= 0.35
            total_kg, direct_kg, hidden_kg = _split_carbon_units(f)

            matched.append(
                {
                    "factor_code": str(f.get("factor_code") or ""),
                    "factor_name": str(f.get("factor_name") or ""),
                    "category_key": str(f.get("category_key") or "food"),
                    "category_label": str(f.get("category_label") or "餐飲食物"),
                    "kg_co2e": total_kg,
                    "direct_kg_co2e": direct_kg,
                    "hidden_supply_kg_co2e": hidden_kg,
                    "declared_unit": str(f.get("declared_unit") or "件"),
                    "unit_to_base_gram": float(f.get("unit_to_base_gram") or 1.0),
                    "is_weight_unit": int(f.get("is_weight_unit") or 0),
                    "source": str(f.get("source") or "環境部產品碳足跡係數庫（MOENV CFP_P_02）"),
                    "similarity": round(max(0.0, min(1.0, sim)), 4),
                }
            )

    matched.sort(key=lambda x: x["similarity"], reverse=True)
    return matched[:3]


class AlignItemRequest(BaseModel):
    item_name: str = Field(..., min_length=1, description="發票或消費品項名稱")
    quantity: float = Field(default=1.0, gt=0, description="數量")
    fallback_category_key: str = Field(
        default="food", description="低信心退回大類平均時的預設類別鍵值"
    )


class ManualCorrectionRequest(BaseModel):
    item_name: str = Field(..., min_length=1, description="原始發票／品項名稱")
    factor_code: Optional[str] = Field(default=None, description="校正後的係數代碼")
    category_key: str = Field(default="food", description="校正後的類別鍵值")
    custom_kg_co2e: Optional[float] = Field(
        default=None, gt=0, description="若無固定係數代碼，可指定單位碳排(kg CO2e)"
    )


@router.post("/rebuild-index")
def rebuild_index() -> Dict[str, Any]:
    col = _build_chroma_collection(force_rebuild=True)
    return {"status": "ok", "count": col.count()}


@router.post("/align-item")
def align_item(req: AlignItemRequest) -> Dict[str, Any]:
    threshold = _get_confidence_threshold()
    split = get_carbon_split()
    cfg = load_estimation_config()
    cfg_fallback = str(cfg.get("fallback_category_key") or "food")

    norm_info = normalize_item_query(req.item_name)
    query_text = norm_info["search_query"]
    unit_scale = float(norm_info.get("unit_scale") or 1.0)

    feedback = _load_feedback_map()
    averages = get_category_averages()
    factors_all = get_all_emission_factors()
    amount_val, amount_unit = extract_measure_from_text(req.item_name)
    default_cat = req.fallback_category_key if req.fallback_category_key in averages else cfg_fallback
    inferred_cat = _infer_category_from_name(
        f"{req.item_name} {norm_info['cleaned_name']} {norm_info['canonical_name']}",
        default_cat,
    )

    fb_key = (
        req.item_name
        if req.item_name in feedback
        else (norm_info["cleaned_name"] if norm_info["cleaned_name"] in feedback else None)
    )
    if fb_key is not None:
        fb = feedback[fb_key]
        base_kg, direct_unit, hidden_unit = _split_carbon_units(fb)
        effective_units = _compute_effective_units(
            fb, float(req.quantity), amount_val, amount_unit, unit_scale=1.0
        )
        total_kg = round(base_kg * effective_units, 4)
        fb_source = str(fb.get("source") or "人工校正記憶（manual_corrections.json）")
        return {
            "item_name": req.item_name,
            "canonical_name": norm_info["canonical_name"],
            "matched": True,
            "match_mode": "manual_correction",
            "is_estimated": False,
            "label": "已對齊碳係數（人工校正記憶）",
            "confidence": 0.99,
            "threshold": threshold,
            "factor_code": fb["factor_code"],
            "factor_name": fb["factor_name"],
            "category_key": fb["category_key"],
            "category_label": fb["category_label"],
            "unit_kg_co2e": base_kg,
            "effective_units": round(effective_units, 4),
            "total_kg_co2e": total_kg,
            "direct_kg_co2e": round(direct_unit * effective_units, 4),
            "hidden_carbon_cost_kg": round(hidden_unit * effective_units, 4),
            "source": fb_source,
            "factor_source": fb_source,
            "basis": f"依先前人工校正紀錄對齊 {fb['factor_name']}（{base_kg} kg CO2e/{fb.get('declared_unit', '件')}）",
            "top3_candidates": [],
        }

    top3: List[Dict[str, Any]] = []
    chroma_ok = False
    try:
        collection = _build_chroma_collection(force_rebuild=False)
        query_res = collection.query(query_texts=[query_text], n_results=3)

        metadatas_list = (query_res.get("metadatas") or [[]])[0]
        distances_list = (query_res.get("distances") or [[]])[0]

        for meta, dist in zip(metadatas_list, distances_list):
            if not meta:
                continue
            d = max(0.0, float(dist))
            similarity = round(max(0.0, min(1.0, math.exp(-d * 0.65))), 4)
            cat_k = str(meta.get("category_key") or "food")
            if _is_category_conflict(inferred_cat, cat_k):
                similarity = round(max(0.0, similarity - 0.35), 4)

            total_kg, direct_kg, hidden_kg = _split_carbon_units(meta)
            top3.append(
                {
                    "factor_code": str(meta.get("factor_code") or ""),
                    "factor_name": str(meta.get("factor_name") or ""),
                    "category_key": cat_k,
                    "category_label": str(meta.get("category_label") or "餐飲食物"),
                    "kg_co2e": total_kg,
                    "direct_kg_co2e": direct_kg,
                    "hidden_supply_kg_co2e": hidden_kg,
                    "declared_unit": str(meta.get("declared_unit") or "件"),
                    "unit_to_base_gram": float(meta.get("unit_to_base_gram") or 1.0),
                    "is_weight_unit": int(meta.get("is_weight_unit") or 0),
                    "source": str(meta.get("source") or "環境部產品碳足跡係數庫（MOENV CFP_P_02）"),
                    "similarity": similarity,
                }
            )
        top3.sort(key=lambda x: x["similarity"], reverse=True)
        chroma_ok = True
    except Exception:
        top3 = _keyword_fallback_candidates(query_text, factors_all)

    reranked = _llm_rerank_and_score(f"{norm_info['canonical_name']} ({req.item_name})", top3)
    if reranked is not None:
        best = reranked
        confidence = float(reranked["confidence"])
    elif top3:
        best = dict(top3[0])
        kw_hit = norm_info["alias_matched"]
        if not kw_hit:
            for f in factors_all:
                if f.get("factor_code") == best["factor_code"]:
                    for kw in f.get("keywords") or []:
                        if kw and str(kw).lower() in query_text.lower():
                            kw_hit = True
                            break
        base_conf = best["similarity"] if chroma_ok else max(best["similarity"], 0.82)
        confidence = round(min(0.95, max(base_conf, 0.86 if kw_hit else base_conf)), 4)
        best["confidence"] = confidence
    else:
        best = None
        confidence = 0.0

    category_conflict = bool(
        best and _is_category_conflict(inferred_cat, str(best.get("category_key") or ""))
    )

    if best is None or confidence < threshold or category_conflict:
        cat_key = (
            inferred_cat
            if inferred_cat in averages
            else (
                best["category_key"]
                if (best and best.get("category_key") in averages and not category_conflict)
                else default_cat
            )
        )
        if cat_key not in averages:
            cat_key = next(iter(averages.keys()))

        cat_info = averages[cat_key]
        avg_kg = float(cat_info.get("avg_kg_co2e") or cat_info.get("kg_co2e") or 0.0)
        eff_units = _compute_fallback_units(float(req.quantity), amount_val, amount_unit, cat_key)
        est_total = round(avg_kg * eff_units, 4)
        est_source = "系統估算設定檔（carbon_estimation_config.json）"

        conflict_reason = (
            f"候選係數類別（{best.get('category_label')}）與推斷類別（{cat_info.get('label', cat_key)}）不符，改走類別平均估算"
            if category_conflict and best
            else (
                cat_info.get("basis")
                or f"信心分數 {confidence} 低於門檻 {threshold}，依 {cat_info.get('label', cat_key)} 類別平均係數（{avg_kg} kg CO2e）× 有效單位 {eff_units} 估算"
            )
        )

        return {
            "item_name": req.item_name,
            "canonical_name": norm_info["canonical_name"],
            "matched": False,
            "match_mode": "category_average",
            "is_estimated": True,
            "label": "估算品項",
            "confidence": confidence,
            "threshold": threshold,
            "factor_code": None,
            "factor_name": f"{cat_info.get('label', cat_key)}大類平均係數",
            "category_key": cat_key,
            "category_label": cat_info.get("label", cat_key),
            "unit_kg_co2e": avg_kg,
            "effective_units": eff_units,
            "total_kg_co2e": est_total,
            "direct_kg_co2e": round(est_total * split["direct_ratio"], 4),
            "hidden_carbon_cost_kg": round(est_total * split["hidden_ratio"], 4),
            "source": est_source,
            "factor_source": est_source,
            "basis": conflict_reason,
            "top3_candidates": top3,
        }

    base_kg, direct_unit, hidden_unit = _split_carbon_units(best)
    effective_units = _compute_effective_units(
        best,
        float(req.quantity),
        amount_val,
        amount_unit,
        unit_scale=unit_scale if amount_val is None else 1.0,
    )
    total_kg_co2e = round(base_kg * effective_units, 4)
    matched_source = str(best.get("source") or "環境部產品碳足跡係數庫（MOENV CFP_P_02）")

    return {
        "item_name": req.item_name,
        "canonical_name": norm_info["canonical_name"],
        "matched": True,
        "match_mode": "matched",
        "is_estimated": False,
        "label": "已對齊碳係數",
        "confidence": confidence,
        "threshold": threshold,
        "factor_code": best["factor_code"],
        "factor_name": best["factor_name"],
        "category_key": best["category_key"],
        "category_label": best["category_label"],
        "unit_kg_co2e": base_kg,
        "effective_units": round(effective_units, 4),
        "total_kg_co2e": total_kg_co2e,
        "direct_kg_co2e": round(direct_unit * effective_units, 4),
        "hidden_carbon_cost_kg": round(hidden_unit * effective_units, 4),
        "source": matched_source,
        "factor_source": matched_source,
        "basis": f"信心分數 {confidence} ≥ 門檻 {threshold}，對齊係數 {best['factor_name']}（{base_kg} kg CO2e/{best['declared_unit']}）× 有效單位 {round(effective_units, 4)}",
        "top3_candidates": top3,
    }


@router.post("/correct")
def save_manual_correction(req: ManualCorrectionRequest) -> Dict[str, Any]:
    split = get_carbon_split()
    factors = get_all_emission_factors()
    averages = get_category_averages()

    chosen_factor = None
    if req.factor_code:
        for f in factors:
            if f.get("factor_code") == req.factor_code:
                chosen_factor = f
                break

    cat_key = req.category_key if req.category_key in averages else "food"
    cat_info = averages[cat_key]
    cat_avg_kg = float(cat_info.get("avg_kg_co2e") or cat_info.get("kg_co2e") or 0.0)

    if chosen_factor is not None:
        total_kg = float(req.custom_kg_co2e or chosen_factor["kg_co2e"])
        record = {
            "factor_code": chosen_factor["factor_code"],
            "factor_name": chosen_factor["factor_name"],
            "category_key": chosen_factor["category_key"],
            "category_label": chosen_factor["category_label"],
            "kg_co2e": total_kg,
            "direct_kg_co2e": float(
                chosen_factor.get("direct_kg_co2e")
                if chosen_factor.get("direct_kg_co2e") is not None
                else round(total_kg * split["direct_ratio"], 6)
            ),
            "hidden_supply_kg_co2e": float(
                chosen_factor.get("hidden_supply_kg_co2e")
                if chosen_factor.get("hidden_supply_kg_co2e") is not None
                else round(total_kg * split["hidden_ratio"], 6)
            ),
            "declared_unit": chosen_factor.get("declared_unit", "件"),
            "unit_to_base_gram": float(chosen_factor.get("unit_to_base_gram", 1.0)),
            "is_weight_unit": int(chosen_factor.get("is_weight_unit", 0)),
            "source": str(chosen_factor.get("source") or "環境部產品碳足跡係數庫（MOENV CFP_P_02）"),
        }
    else:
        total_kg = float(req.custom_kg_co2e or cat_avg_kg)
        record = {
            "factor_code": "MANUAL-CUSTOM",
            "factor_name": f"{cat_info.get('label', cat_key)}手動校正係數",
            "category_key": cat_key,
            "category_label": cat_info.get("label", cat_key),
            "kg_co2e": total_kg,
            "direct_kg_co2e": round(total_kg * split["direct_ratio"], 6),
            "hidden_supply_kg_co2e": round(total_kg * split["hidden_ratio"], 6),
            "declared_unit": cat_info.get("unit", "份"),
            "unit_to_base_gram": 1.0,
            "is_weight_unit": 0,
            "source": "使用者手動校正（manual_corrections.json）",
        }

    feedback = _load_feedback_map()
    feedback[req.item_name] = record
    _save_feedback_map(feedback)
    try:
        _build_chroma_collection(force_rebuild=True)
    except Exception:
        pass

    return {
        "status": "ok",
        "message": "已儲存人工校正並更新向量庫",
        "item_name": req.item_name,
        "correction": record,
    }