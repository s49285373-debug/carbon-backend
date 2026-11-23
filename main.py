import os
import base64
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import bcrypt
import pymysql

app = FastAPI(title="日常發票載具之個人碳足跡與隱形碳成本智慧盤查推估系統")

# 1. 設定 CORS (允許來自 Vercel 雲端前端與本機的跨網域呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 連線 MySQL 資料庫的函式 (自動適配 Railway 雲端環境變數或本機環境)
def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", os.getenv("MYSQLHOST", "localhost")),
        user=os.getenv("DB_USER", os.getenv("MYSQLUSER", "root")),
        password=os.getenv("DB_PASSWORD", os.getenv("MYSQLPASSWORD", "")),
        database=os.getenv("DB_NAME", os.getenv("MYSQLDATABASE", "carbon_db")),
        port=int(os.getenv("DB_PORT", os.getenv("MYSQLPORT", 3306))),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

# 3. 定義前端傳入的註冊、登入與發票解析資料格式
class RegisterRequest(BaseModel):
    phone: str = Field(..., description="手機號碼")
    password: str = Field(..., description="密碼")
    nickname: str = Field(..., description="暱稱")
    role: str = Field(..., description="身分")

class LoginRequest(BaseModel):
    phone: str = Field(..., description="手機號碼")
    password: str = Field(..., description="密碼")

class InvoiceExtractRequest(BaseModel):
    phone: str | None = Field(None, description="會員手機號碼")
    invoice_type: str = Field("electronic", description="發票類型：electronic 或 traditional")
    source_type: str = Field("qr", description="來源類型：qr、qr_simulated 或 manual")
    qr_raw_data: str = Field(..., description="發票 QR Code 原始字串或手動輸入 JSON")

# 4. 註冊 API 端點
@app.post("/api/register")
def register_user(user: RegisterRequest):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql_check = "SELECT * FROM users WHERE phone = %s"
            cursor.execute(sql_check, (user.phone,))
            existing_user = cursor.fetchone()

            if existing_user:
                raise HTTPException(status_code=400, detail="此手機號碼已被註冊過！")

            hashed_password = bcrypt.hashpw(
                user.password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            sql_insert = (
                "INSERT INTO users (phone, password, nickname, role) VALUES (%s, %s, %s, %s)"
            )
            cursor.execute(sql_insert, (user.phone, hashed_password, user.nickname, user.role))
            connection.commit()

        return {
            "status": "success",
            "message": "註冊成功，已安全存入 MySQL 資料庫！",
            "data": {
                "phone": user.phone,
                "nickname": user.nickname,
                "role": user.role,
            },
        }
    finally:
        connection.close()

# 5. 登入 API 端點
@app.post("/api/login")
def login_user(user: LoginRequest):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql_check = "SELECT * FROM users WHERE phone = %s"
            cursor.execute(sql_check, (user.phone,))
            db_user = cursor.fetchone()

            if not db_user:
                raise HTTPException(status_code=400, detail="此手機號碼尚未註冊！")

            stored_password = db_user["password"].encode("utf-8")
            if not bcrypt.checkpw(user.password.encode("utf-8"), stored_password):
                raise HTTPException(status_code=400, detail="密碼錯誤！")

            return {
                "status": "success",
                "message": "登入成功！",
                "nickname": db_user["nickname"],
                "role": db_user["role"],
            }
    finally:
        connection.close()

# 6. 碳足跡係數、店家統編對照與類別推估設定
ITEM_CARBON_FACTORS = {
    "牛肉": 0.0270,
    "豬肉": 0.0072,
    "雞": 0.0057,
    "炸雞": 0.0065,
    "年糕": 0.0022,
    "薯條": 0.0024,
    "奶": 0.0032,
    "茶": 0.0012,
    "咖啡": 0.0020,
    "飯糰": 0.0018,
    "便當": 0.0025,
}

CATEGORY_INTENSITY = {
    "餐飲食品": 0.0022,
    "便利商店": 0.0018,
    "一般消費": 0.0015,
}

MERCHANT_TAX_MAP: dict[str, tuple[str, str]] = {
    # 可依常掃發票補上賣方統編 -> (店家名稱, 消費類別)
    # "56801904": ("蝦皮購物", "一般消費"),
    # "22099131": ("統一超商 (7-ELEVEN)", "便利商店"),
    # "22555003": ("全家便利商店", "便利商店"),
}

TRACK_CATEGORY_MAP: dict[str, str] = {
    "FT": "一般消費",
    "FG": "餐飲消費",
}

HIDDEN_CARBON_COST_PER_KG = 3.2  # 每公斤 CO2e 隱形碳成本（元/kg CO2e）

def estimate_item_carbon(item_name: str, amount: float) -> float:
    for kw, factor in ITEM_CARBON_FACTORS.items():
        if kw in item_name:
            return round(amount * factor, 4)
    return round(amount * CATEGORY_INTENSITY["一般消費"], 4)

def lookup_merchant_category(
    seller_tax_id: str | None,
    invoice_number: str = "",
    connection=None,
) -> tuple[str | None, str]:
    """查詢資料庫、統編對照表或字軌預設類別，回傳 (店家名稱或None, 消費類別)"""
    prefix = invoice_number[:2].strip().upper() if len(invoice_number) >= 2 else ""
    default_category = TRACK_CATEGORY_MAP.get(prefix, "一般消費")

    if seller_tax_id:
        tax_key = seller_tax_id.strip()
        if connection is not None:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT merchant_name, category_name FROM merchant_categories WHERE seller_tax_id = %s LIMIT 1",
                        (tax_key,),
                    )
                    row = cursor.fetchone()
                    if row and row.get("merchant_name"):
                        return row["merchant_name"], row.get("category_name") or default_category
            except Exception:
                pass

        if tax_key in MERCHANT_TAX_MAP:
            mapped_name, mapped_cat = MERCHANT_TAX_MAP[tax_key]
            return mapped_name, mapped_cat or default_category

    return None, default_category

# 7. 發票內容解析工具與 API 端點
def decode_text_safely(text: str) -> str:
    """嘗試還原可能經過 Base64 或 Big5/UTF-8 編碼的發票文字"""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if len(cleaned) >= 8 and len(cleaned) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
        try:
            decoded_bytes = base64.b64decode(cleaned, validate=True)
            for enc in ("utf-8", "big5", "cp950"):
                try:
                    decoded_str = decoded_bytes.decode(enc).strip()
                    if decoded_str and all(ord(ch) >= 32 or ch in "\r\n\t" for ch in decoded_str):
                        return decoded_str
                except UnicodeDecodeError:
                    continue
        except Exception:
            pass

    return cleaned

def parse_roc_date_to_iso(roc_7digits: str) -> str | None:
    """將民國 7 碼日期（如 1150920）轉為西元日期（如 2026-09-20）"""
    if len(roc_7digits) == 7 and roc_7digits.isdigit():
        roc_year = int(roc_7digits[0:3])
        month = int(roc_7digits[3:5])
        day = int(roc_7digits[5:7])
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{roc_year + 1911:04d}-{month:02d}-{day:02d}"
    return None

def parse_qr_header_amount(header: str) -> float:
    """從 77 碼表頭第 29~37 碼（總計金額）或第 21~29 碼（銷售額）嘗試解析金額"""
    if len(header) >= 37:
        total_seg = header[29:37].strip()
        if total_seg.isdigit():
            val = float(int(total_seg))
            if val > 0:
                return val
        try:
            val_hex = float(int(total_seg, 16))
            if val_hex > 0:
                return val_hex
        except ValueError:
            pass

    if len(header) >= 29:
        sales_seg = header[21:29].strip()
        if sales_seg.isdigit():
            val = float(int(sales_seg))
            if val > 0:
                return val
    return 0.0

def is_valid_item_name(name: str) -> bool:
    """過濾遮罩字元（如 ********）、空白或純數字品名"""
    cleaned = name.strip()
    if not cleaned:
        return False
    if re.fullmatch(r"\*+", cleaned):
        return False
    if cleaned.isdigit():
        return False
    return True

def extract_invoice_content(raw_data: str, source_type: str = "qr", db_conn=None) -> dict:
    """
    解析紙本電子發票 QR Code 或手動輸入內容：
    - 有逐項商品明細時：逐項計算碳足跡（保留負值折扣品項）
    - 無逐項商品明細時：不產生假商品，改依總金額與預設類別係數推估碳足跡與隱形碳成本
    """
    raw = decode_text_safely(raw_data)
    if not raw:
        raise ValueError("發票內容為空，無法解析")

    # 手動輸入模式（前端傳入 JSON: invoiceType, invoiceDate, invoiceNumber, totalAmount, randomCode）
    if source_type == "manual":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invoiceNumber": raw[:10]}

        invoice_number = str(
            payload.get("invoiceNumber") or payload.get("invoice_number") or ""
        ).strip().upper()
        purchase_date = str(
            payload.get("invoiceDate") or payload.get("purchase_date") or ""
        ).strip() or None
        random_code = str(
            payload.get("randomCode") or payload.get("random_code") or ""
        ).strip() or None
        seller_tax_id = str(payload.get("seller_tax_id") or "").strip() or None

        raw_amount = payload.get("totalAmount")
        if raw_amount is None:
            raw_amount = payload.get("total_amount", 0)
        total_amount = round(float(raw_amount or 0), 2)

        merchant_raw, category_name = lookup_merchant_category(
            seller_tax_id, invoice_number=invoice_number, connection=db_conn
        )
        store_label = (
            merchant_raw
            if merchant_raw
            else (f"統編 {seller_tax_id}" if seller_tax_id else f"{invoice_number[:2] or '手動'}字軌")
        )
        merchant_name = f"{store_label} · {category_name}"

        factor = CATEGORY_INTENSITY.get(category_name, CATEGORY_INTENSITY["一般消費"])
        carbon_kg = round(max(total_amount, 0.0) * factor, 2)
        hidden_carbon_cost = round(carbon_kg * HIDDEN_CARBON_COST_PER_KG, 2)

        return {
            "invoice_number": invoice_number or "手動輸入",
            "purchase_date": purchase_date,
            "random_code": random_code,
            "seller_tax_id": seller_tax_id,
            "merchant_name": merchant_name,
            "category_name": category_name,
            "total_amount": total_amount,
            "carbon_kg": carbon_kg,
            "hidden_carbon_cost": hidden_carbon_cost,
            "calc_label": f"無逐項明細，套用預設「{category_name}」係數推估",
            "items": [],
        }

    # 紙本電子發票 QR Code 模式：前 77 碼為固定表頭，之後為冒號分隔明細
    header = raw[:77]
    detail_section = raw[77:] if len(raw) > 77 else ""

    invoice_number = header[0:10].strip().upper() if len(header) >= 10 else ""
    roc_date = header[10:17].strip() if len(header) >= 17 else ""
    purchase_date = parse_roc_date_to_iso(roc_date)
    random_code = header[17:21].strip() if len(header) >= 21 else ""
    seller_tax_id = header[45:53].strip() if len(header) >= 53 else ""
    header_amount = parse_qr_header_amount(header)

    tokens = [decode_text_safely(t) for t in detail_section.split(":")]
    filtered_tokens = [t for t in tokens if t != "" and not t.startswith("**")]

    items = []
    idx = 0
    while idx + 2 < len(filtered_tokens):
        name_candidate = filtered_tokens[idx].strip()
        qty_candidate = filtered_tokens[idx + 1].strip()
        price_candidate = filtered_tokens[idx + 2].strip()

        try:
            qty = float(qty_candidate)
            price = float(price_candidate)
            if not is_valid_item_name(name_candidate) or qty == 0:
                idx += 1
                continue

            amount = round(qty * price, 2)
            item_carbon = estimate_item_carbon(name_candidate, amount)
            items.append(
                {
                    "item_name": name_candidate,
                    "quantity": qty,
                    "unit_price": price,
                    "amount": amount,
                    "carbon_kg": item_carbon,
                }
            )
            idx += 3
        except ValueError:
            idx += 1

    merchant_raw, category_name = lookup_merchant_category(
        seller_tax_id, invoice_number=invoice_number, connection=db_conn
    )
    store_label = (
        merchant_raw
        if merchant_raw
        else (f"統編 {seller_tax_id}" if seller_tax_id else f"{invoice_number[:2] or 'QR'}字軌")
    )
    merchant_name = f"{store_label} · {category_name}"

    if items:
        items_sum = round(sum(item["amount"] for item in items), 2)
        total_amount = header_amount if header_amount > 0 else items_sum
        carbon_kg = round(max(sum(item["carbon_kg"] for item in items), 0.0), 2)
        hidden_carbon_cost = round(carbon_kg * HIDDEN_CARBON_COST_PER_KG, 2)
        calc_label = "逐項商品明細推估"
    else:
        total_amount = header_amount
        factor = CATEGORY_INTENSITY.get(category_name, CATEGORY_INTENSITY["一般消費"])
        carbon_kg = round(max(total_amount, 0.0) * factor, 2)
        hidden_carbon_cost = round(carbon_kg * HIDDEN_CARBON_COST_PER_KG, 2)
        calc_label = f"無逐項明細，套用預設「{category_name}」係數推估"

    return {
        "invoice_number": invoice_number or "未知字軌",
        "purchase_date": purchase_date,
        "random_code": random_code or None,
        "seller_tax_id": seller_tax_id or None,
        "merchant_name": merchant_name,
        "category_name": category_name,
        "total_amount": total_amount,
        "carbon_kg": carbon_kg,
        "hidden_carbon_cost": hidden_carbon_cost,
        "calc_label": calc_label,
        "items": items,
    }

@app.post("/api/invoices/extract")
def extract_invoice(req: InvoiceExtractRequest):
    conn = None
    try:
        try:
            conn = get_db_connection()
        except Exception:
            conn = None

        parsed = extract_invoice_content(req.qr_raw_data, req.source_type, db_conn=conn)
        return {
            "status": "success",
            "phone": req.phone,
            "invoice_type": req.invoice_type,
            "source_type": req.source_type,
            "data": parsed,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if conn is not None:
            conn.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)