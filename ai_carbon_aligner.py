import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from fastapi import APIRouter
from openai import OpenAI
from pydantic import BaseModel, Field

from carbon_moenv_service import (
    get_all_emission_factors,
    get_category_averages,
    load_seed_data,
)

router = APIRouter(prefix="/api/carbon/ai", tags=["AI Carbon Aligner"])

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
FEEDBACK_FILE = BASE_DIR / "manual_corrections.json"

seed_meta = load_seed_data().get("source", {})
CONFIDENCE_THRESHOLD = float(seed_meta.get("confidence_threshold", 0.75))


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
    try:
        with FEEDBACK_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _build_chroma_collection() -> Any:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="moenv_emission_factors")

    factors = get_all_emission_factors()
    feedback = _load_feedback_map()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for f in factors:
        code = str(f["factor_code"])
        kw_list = f.get("keywords") or []
        kw_text = "、".join(kw_list) if isinstance(kw_list, list) else str(kw_list)
        doc_text = (
            f"係數代碼:{code} | 品項:{f['factor_name']} | "
            f"類別:{f['category_label']}({f['category_key']}) | 關鍵字:{kw_text}"
        )
        ids.append(f"factor::{code}")
        documents.append(doc_text)
        metadatas.append(
            {
                "factor_code": code,
                "factor_name": str(f["factor_name"]),
                "category_key": str(f["category_key"]),
                "category_label": str(f["category_label"]),
                "kg_co2e": float(f["kg_co2e"]),
                "declared_unit": str(f.get("declared_unit") or "件"),
                "unit_to_base_gram": float(f.get("unit_to_base_gram") or 1.0),
                "is_weight_unit": int(f.get("is_weight_unit") or 0),
                "is_user_feedback": False,
            }
        )

    for raw_name, fb in feedback.items():
        fb_id = f"feedback::{raw_name}"
        doc_text = (
            f"人工校正品名:{raw_name} -> 係數:{fb.get('factor_name')} "
            f"({fb.get('factor_code')}) | 類別:{fb.get('category_label')}"
        )
        ids.append(fb_id)
        documents.append(doc_text)
        metadatas.append(
            {
                "factor_code": str(fb.get("factor_code") or ""),
                "factor_name": str(fb.get("factor_name") or ""),
                "category_key": str(fb.get("category_key") or "food"),
                "category_label": str(fb.get("category_label") or "餐飲食物"),
                "kg_co2e": float(fb.get("kg_co2e") or 1.45),
                "declared_unit": str(fb.get("declared_unit") or "份"),
                "unit_to_base_gram": float(fb.get("unit_to_base_gram") or 1.0),
                "is_weight_unit": int(fb.get("is_weight_unit") or 0),
                "is_user_feedback": True,
            }
        )

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def _extract_volume_or_weight(item_name: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ml|mL|ML|g|G)\b", item_name)
    if match:
        return float(match.group(1))
    tail_match = re.search(r"(\d{3,4})$", item_name.strip())
    if tail_match:
        val = float(tail_match.group(1))
        if 100.0 <= val <= 2000.0:
            return val
    return None


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
                        "你是碳足跡品項語意對齊助手。請根據輸入的發票或消費品項名稱，"
                        "從候選排放係數中選出最貼切的一筆，並給出 0.0 到 1.0 的信心分數 confidence。"
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
    item_name: str, factors_all: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    lower_name = item_name.lower()
    matched: List[Dict[str, Any]] = []
    for f in factors_all:
        kw_list = f.get("keywords") or []
        hit = any(kw and str(kw).lower() in lower_name for kw in kw_list)
        if hit or str(f.get("factor_name") or "").lower() in lower_name:
            matched.append(
                {
                    "factor_code": str(f.get("factor_code") or ""),
                    "factor_name": str(f.get("factor_name") or ""),
                    "category_key": str(f.get("category_key") or "food"),
                    "category_label": str(f.get("category_label") or "餐飲食物"),
                    "kg_co2e": float(f.get("kg_co2e") or 0.0),
                    "declared_unit": str(f.get("declared_unit") or "件"),
                    "unit_to_base_gram": float(f.get("unit_to_base_gram") or 1.0),
                    "is_weight_unit": int(f.get("is_weight_unit") or 0),
                    "similarity": 0.86 if hit else 0.78,
                }
            )
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


@router.post("/align-item")
def align_item(req: AlignItemRequest) -> Dict[str, Any]:
    feedback = _load_feedback_map()
    averages = get_category_averages()
    factors_all = get_all_emission_factors()
    amount_val = _extract_volume_or_weight(req.item_name)

    if req.item_name in feedback:
        fb = feedback[req.item_name]
        base_kg = float(fb["kg_co2e"])
        is_weight = int(fb.get("is_weight_unit") or 0)
        unit_to_base = float(fb.get("unit_to_base_gram") or 1.0)
        effective_units = (
            (amount_val / unit_to_base) * float(req.quantity)
            if (is_weight == 1 and amount_val and unit_to_base > 0)
            else float(req.quantity)
        )
        return {
            "item_name": req.item_name,
            "matched": True,
            "is_estimated": False,
            "label": "已對齊碳係數（人工校正記憶）",
            "confidence": 0.99,
            "threshold": CONFIDENCE_THRESHOLD,
            "factor_code": fb["factor_code"],
            "factor_name": fb["factor_name"],
            "category_key": fb["category_key"],
            "category_label": fb["category_label"],
            "effective_units": round(effective_units, 4),
            "total_kg_co2e": round(base_kg * effective_units, 4),
            "top3_candidates": [],
        }

    top3: List[Dict[str, Any]] = []
    chroma_ok = False
    try:
        collection = _build_chroma_collection()
        query_res = collection.query(query_texts=[req.item_name], n_results=3)

        metadatas_list = (query_res.get("metadatas") or [[]])[0]
        distances_list = (query_res.get("distances") or [[]])[0]

        for meta, dist in zip(metadatas_list, distances_list):
            if not meta:
                continue
            similarity = round(max(0.0, min(1.0, 1.0 - float(dist) / 2.0)), 4)
            if float(dist) <= 1.0:
                similarity = round(max(0.0, min(1.0, math.exp(-float(dist)))), 4)

            top3.append(
                {
                    "factor_code": str(meta.get("factor_code") or ""),
                    "factor_name": str(meta.get("factor_name") or ""),
                    "category_key": str(meta.get("category_key") or "food"),
                    "category_label": str(meta.get("category_label") or "餐飲食物"),
                    "kg_co2e": float(meta.get("kg_co2e") or 0.0),
                    "declared_unit": str(meta.get("declared_unit") or "件"),
                    "unit_to_base_gram": float(meta.get("unit_to_base_gram") or 1.0),
                    "is_weight_unit": int(meta.get("is_weight_unit") or 0),
                    "similarity": similarity,
                }
            )
        chroma_ok = True
    except Exception:
        top3 = _keyword_fallback_candidates(req.item_name, factors_all)

    reranked = _llm_rerank_and_score(req.item_name, top3)
    if reranked is not None:
        best = reranked
        confidence = float(reranked["confidence"])
    elif top3:
        best = dict(top3[0])
        kw_hit = False
        for f in factors_all:
            if f.get("factor_code") == best["factor_code"]:
                for kw in f.get("keywords") or []:
                    if kw and str(kw).lower() in req.item_name.lower():
                        kw_hit = True
                        break
        base_conf = best["similarity"] if chroma_ok else max(best["similarity"], 0.82)
        confidence = round(min(0.95, max(base_conf, 0.86 if kw_hit else base_conf)), 4)
        best["confidence"] = confidence
    else:
        best = None
        confidence = 0.0

    if best is None or confidence < CONFIDENCE_THRESHOLD:
        cat_key = (
            best["category_key"]
            if (best and best.get("category_key") in averages)
            else (req.fallback_category_key if req.fallback_category_key in averages else "food")
        )
        cat_info = averages[cat_key]
        avg_kg = float(cat_info["avg_kg_co2e"])
        total_kg_co2e = round(avg_kg * float(req.quantity), 4)
        return {
            "item_name": req.item_name,
            "matched": False,
            "is_estimated": True,
            "label": "估算品項",
            "confidence": confidence,
            "threshold": CONFIDENCE_THRESHOLD,
            "factor_code": None,
            "factor_name": f"{cat_info['label']}大類平均係數",
            "category_key": cat_key,
            "category_label": cat_info["label"],
            "effective_units": float(req.quantity),
            "total_kg_co2e": total_kg_co2e,
            "top3_candidates": top3,
        }

    base_kg = float(best["kg_co2e"])
    is_weight = int(best.get("is_weight_unit") or 0)
    unit_to_base = float(best.get("unit_to_base_gram") or 1.0)
    effective_units = (
        (amount_val / unit_to_base) * float(req.quantity)
        if (is_weight == 1 and amount_val and unit_to_base > 0)
        else float(req.quantity)
    )
    total_kg_co2e = round(base_kg * effective_units, 4)

    return {
        "item_name": req.item_name,
        "matched": True,
        "is_estimated": False,
        "label": "已對齊碳係數",
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "factor_code": best["factor_code"],
        "factor_name": best["factor_name"],
        "category_key": best["category_key"],
        "category_label": best["category_label"],
        "effective_units": round(effective_units, 4),
        "total_kg_co2e": total_kg_co2e,
        "top3_candidates": top3,
    }


@router.post("/correct")
def save_manual_correction(req: ManualCorrectionRequest) -> Dict[str, Any]:
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

    if chosen_factor is not None:
        record = {
            "factor_code": chosen_factor["factor_code"],
            "factor_name": chosen_factor["factor_name"],
            "category_key": chosen_factor["category_key"],
            "category_label": chosen_factor["category_label"],
            "kg_co2e": float(req.custom_kg_co2e or chosen_factor["kg_co2e"]),
            "declared_unit": chosen_factor.get("declared_unit", "件"),
            "unit_to_base_gram": float(chosen_factor.get("unit_to_base_gram", 1.0)),
            "is_weight_unit": int(chosen_factor.get("is_weight_unit", 0)),
        }
    else:
        record = {
            "factor_code": "MANUAL-CUSTOM",
            "factor_name": f"{cat_info['label']}手動校正係數",
            "category_key": cat_key,
            "category_label": cat_info["label"],
            "kg_co2e": float(req.custom_kg_co2e or cat_info["avg_kg_co2e"]),
            "declared_unit": cat_info.get("unit", "份"),
            "unit_to_base_gram": 1.0,
            "is_weight_unit": 0,
        }

    feedback = _load_feedback_map()
    feedback[req.item_name] = record
    _save_feedback_map(feedback)
    try:
        _build_chroma_collection()
    except Exception:
        pass

    return {
        "status": "ok",
        "message": "已儲存人工校正並更新向量庫",
        "item_name": req.item_name,
        "correction": record,
    }