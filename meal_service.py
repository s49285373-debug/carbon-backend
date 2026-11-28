from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ai_carbon_aligner import AlignItemRequest, align_item

router = APIRouter(prefix="/api/meals", tags=["Meal Carbon Records"])

HIDDEN_CARBON_COST_PER_KG = 3.2


def _format_meal_date(date_str: Optional[str]) -> str:
    if not date_str:
        return datetime.now().strftime("%Y/%m/%d")
    cleaned = date_str.strip().replace("-", "/")
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", cleaned):
        y, m, d = cleaned.split("/")
        return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"
    return cleaned


def _convert_to_grams(quantity: float, unit: str) -> float:
    u = (unit or "").strip().lower()
    if "公斤" in u or u == "kg":
        return quantity * 1000.0
    if "毫克" in u or u == "mg":
        return quantity / 1000.0
    if "台斤" in u or "斤" in u:
        return quantity * 600.0
    if "兩" in u:
        return quantity * 37.5
    return quantity


class MealIngredientInput(BaseModel):
    ingredient_name: str = Field(..., min_length=1, description="食材名稱")
    quantity: float = Field(default=150.0, gt=0, description="份量數值")
    unit: str = Field(default="公克 (g)", description="單位，例如 公克 (g)")


class MealCreateRequest(BaseModel):
    phone: Optional[str] = Field(default=None, description="會員手機號碼")
    meal_name: str = Field(..., min_length=1, description="餐點名稱")
    meal_date: Optional[str] = Field(default=None, description="用餐日期 YYYY/MM/DD")
    meal_time_slot: str = Field(default="正餐", description="餐別")
    amount: float = Field(default=0.0, ge=0, description="餐點金額（元）")
    category_key: str = Field(default="food", description="預設類別鍵值")
    note: Optional[str] = Field(default=None, description="備註")
    items: List[MealIngredientInput] = Field(default_factory=list, description="食材品項與份量清單")


@router.post("")
def create_meal_record(req: MealCreateRequest) -> Dict[str, Any]:
    from main import get_db_connection

    phone_clean = (req.phone or "default").strip()
    meal_name_clean = req.meal_name.strip()
    formatted_date = _format_meal_date(req.meal_date)

    if not meal_name_clean:
        raise HTTPException(status_code=400, detail="請輸入餐點名稱")

    aligned_items: List[Dict[str, Any]] = []
    total_carbon_kg = 0.0
    confidences: List[float] = []
    any_estimated = False

    if req.items:
        for ing in req.items:
            name_clean = ing.ingredient_name.strip()
            if not name_clean:
                continue
            qty_val = float(ing.quantity)
            unit_str = (ing.unit or "公克 (g)").strip()

            grams_val = _convert_to_grams(qty_val, unit_str)
            is_weight_g = "g" in unit_str.lower() or "公克" in unit_str
            query_item_str = (
                f"{name_clean} {int(grams_val) if grams_val.is_integer() else grams_val}g"
                if is_weight_g
                else name_clean
            )

            res = align_item(
                AlignItemRequest(
                    item_name=query_item_str,
                    quantity=1.0 if is_weight_g else qty_val,
                    fallback_category_key=req.category_key or "food",
                )
            )

            item_carbon = round(float(res.get("total_kg_co2e") or 0.0), 4)
            item_conf = float(res.get("confidence") or 0.0)
            item_est = bool(res.get("is_estimated"))

            total_carbon_kg += item_carbon
            confidences.append(item_conf)
            if item_est:
                any_estimated = True

            aligned_items.append(
                {
                    "ingredient_name": name_clean,
                    "quantity": qty_val,
                    "unit": unit_str,
                    "factor_code": res.get("factor_code"),
                    "factor_name": res.get("factor_name"),
                    "confidence": item_conf,
                    "is_estimated": item_est,
                    "label": res.get("label"),
                    "carbon_kg": item_carbon,
                }
            )
    else:
        res = align_item(
            AlignItemRequest(
                item_name=meal_name_clean,
                quantity=1.0,
                fallback_category_key=req.category_key or "food",
            )
        )
        item_carbon = round(float(res.get("total_kg_co2e") or 0.0), 4)
        item_conf = float(res.get("confidence") or 0.0)
        item_est = bool(res.get("is_estimated"))

        total_carbon_kg = item_carbon
        confidences.append(item_conf)
        any_estimated = item_est

    total_carbon_kg = round(total_carbon_kg, 4)
    hidden_carbon_cost = round(total_carbon_kg * HIDDEN_CARBON_COST_PER_KG, 2)
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    is_estimated_flag = 1 if any_estimated else 0

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql_meal = """
                INSERT INTO meal_records (
                    phone, meal_name, meal_date, meal_time_slot,
                    amount, category_key, category_label, confidence, is_estimated,
                    carbon_kg, hidden_carbon_cost, note
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                sql_meal,
                (
                    phone_clean,
                    meal_name_clean,
                    formatted_date,
                    req.meal_time_slot.strip() or "正餐",
                    float(req.amount),
                    req.category_key or "food",
                    "餐飲食物",
                    avg_confidence,
                    is_estimated_flag,
                    total_carbon_kg,
                    hidden_carbon_cost,
                    req.note,
                ),
            )
            meal_id = cursor.lastrowid

            if aligned_items:
                sql_item = """
                    INSERT INTO meal_items (
                        meal_id, ingredient_name, quantity, unit,
                        factor_code, factor_name, confidence, is_estimated, carbon_kg
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                rows = [
                    (
                        meal_id,
                        it["ingredient_name"],
                        it["quantity"],
                        it["unit"],
                        it["factor_code"],
                        it["factor_name"],
                        it["confidence"],
                        1 if it["is_estimated"] else 0,
                        it["carbon_kg"],
                    )
                    for it in aligned_items
                ]
                cursor.executemany(sql_item, rows)

            conn.commit()

        return {
            "status": "success",
            "message": "餐點與食材碳排已儲存",
            "data": {
                "id": meal_id,
                "phone": phone_clean,
                "meal_name": meal_name_clean,
                "meal_date": formatted_date,
                "meal_time_slot": req.meal_time_slot.strip() or "正餐",
                "amount": float(req.amount),
                "category_key": req.category_key or "food",
                "category_label": "餐飲食物",
                "confidence": avg_confidence,
                "is_estimated": bool(is_estimated_flag),
                "label": "估算品項" if is_estimated_flag else "已對齊碳係數",
                "carbon_kg": total_carbon_kg,
                "hidden_carbon_cost": hidden_carbon_cost,
                "items": aligned_items,
            },
        }
    finally:
        conn.close()


@router.get("")
def list_meal_records(
    phone: str = Query(..., description="會員手機號碼"),
    start_date: Optional[str] = Query(default=None, description="起始日期 YYYY/MM/DD"),
    end_date: Optional[str] = Query(default=None, description="結束日期 YYYY/MM/DD"),
) -> Dict[str, Any]:
    from main import get_db_connection

    phone_clean = phone.strip()
    if not phone_clean:
        raise HTTPException(status_code=400, detail="請提供手機號碼")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM meal_records WHERE phone = %s"
            params: List[Any] = [phone_clean]

            if start_date:
                sql += " AND meal_date >= %s"
                params.append(_format_meal_date(start_date))
            if end_date:
                sql += " AND meal_date <= %s"
                params.append(_format_meal_date(end_date))

            sql += " ORDER BY meal_date DESC, id DESC"
            cursor.execute(sql, tuple(params))
            meals = cursor.fetchall()

            if not meals:
                return {"status": "success", "data": []}

            meal_ids = [m["id"] for m in meals]
            placeholders = ",".join(["%s"] * len(meal_ids))
            cursor.execute(
                f"SELECT * FROM meal_items WHERE meal_id IN ({placeholders}) ORDER BY id ASC",
                tuple(meal_ids),
            )
            all_items = cursor.fetchall()

        items_by_meal: Dict[int, List[Dict[str, Any]]] = {mid: [] for mid in meal_ids}
        for it in all_items:
            est = bool(it.get("is_estimated"))
            items_by_meal[it["meal_id"]].append(
                {
                    "id": it["id"],
                    "ingredient_name": it["ingredient_name"],
                    "quantity": float(it.get("quantity") or 1.0),
                    "unit": it.get("unit") or "公克 (g)",
                    "factor_code": it.get("factor_code"),
                    "factor_name": it.get("factor_name"),
                    "confidence": float(it.get("confidence") or 0.0),
                    "is_estimated": est,
                    "label": "估算品項" if est else "已對齊碳係數",
                    "carbon_kg": float(it.get("carbon_kg") or 0.0),
                }
            )

        result: List[Dict[str, Any]] = []
        for m in meals:
            is_est = bool(m.get("is_estimated"))
            result.append(
                {
                    "id": m["id"],
                    "phone": m["phone"],
                    "meal_name": m["meal_name"],
                    "meal_date": _format_meal_date(m.get("meal_date")),
                    "meal_time_slot": m.get("meal_time_slot") or "正餐",
                    "amount": float(m.get("amount") or 0.0),
                    "category_key": m.get("category_key") or "food",
                    "category_label": m.get("category_label") or "餐飲食物",
                    "confidence": float(m.get("confidence") or 0.0),
                    "is_estimated": is_est,
                    "label": "估算品項" if is_est else "已對齊碳係數",
                    "carbon_kg": float(m.get("carbon_kg") or 0.0),
                    "hidden_carbon_cost": float(m.get("hidden_carbon_cost") or 0.0),
                    "note": m.get("note"),
                    "source_type": "meal",
                    "items": items_by_meal.get(m["id"], []),
                }
            )

        return {"status": "success", "data": result}
    finally:
        conn.close()


@router.delete("/{meal_id}")
def delete_meal_record(
    meal_id: int, phone: str = Query(..., description="會員手機號碼")
) -> Dict[str, Any]:
    from main import get_db_connection

    phone_clean = phone.strip()
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM meal_records WHERE id = %s AND phone = %s",
                (meal_id, phone_clean),
            )
            affected = cursor.rowcount
            conn.commit()

        if affected == 0:
            raise HTTPException(status_code=404, detail="找不到該筆餐點紀錄或無權刪除")

        return {"status": "success", "message": "餐點紀錄已刪除"}
    finally:
        conn.close()