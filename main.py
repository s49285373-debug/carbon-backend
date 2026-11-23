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
    allow_origins=["*"],  # 專題上線測試期間允許所有來源
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
        cursorclass=pymysql.cursors.DictCursor
    )

# 3. 定義前端傳入的註冊與登入資料格式
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
            
            hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            sql_insert = "INSERT INTO users (phone, password, nickname, role) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql_insert, (user.phone, hashed_password, user.nickname, user.role))
            connection.commit()
            
        return {
            "status": "success",
            "message": "註冊成功，已安全存入 MySQL 資料庫！",
            "data": {
                "phone": user.phone,
                "nickname": user.nickname,
                "role": user.role
            }
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
            
            stored_password = db_user["password"].encode('utf-8')
            if not bcrypt.checkpw(user.password.encode('utf-8'), stored_password):
                raise HTTPException(status_code=400, detail="密碼錯誤！")
            
            return {
                "status": "success",
                "message": "登入成功！",
                "nickname": db_user["nickname"],
                "role": db_user["role"]
            }
    finally:
        connection.close()

# 6. 發票內容解析工具與 API 端點 (先擷取發票表頭與明細，暫不寫入資料庫)
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

def extract_invoice_content(raw_data: str, source_type: str = "qr") -> dict:
    """解析紙本電子發票 QR Code 或手動輸入內容，回傳結構化發票與品項明細"""
    raw = decode_text_safely(raw_data)
    if not raw:
        raise ValueError("發票內容為空，無法解析")

    # 手動輸入模式
    if source_type == "manual":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invoice_number": raw[:10], "item_name": raw}

        invoice_number = str(payload.get("invoice_number") or "").strip().upper()
        purchase_date = str(payload.get("purchase_date") or "").strip() or None
        item_name = str(payload.get("item_name") or "未命名品項").strip()
        quantity = float(payload.get("quantity") or 1)
        unit_price = float(payload.get("unit_price") or payload.get("amount") or 0)
        amount = float(payload.get("amount") if payload.get("amount") is not None else quantity * unit_price)

        return {
            "invoice_number": invoice_number or "手動輸入",
            "purchase_date": purchase_date,
            "random_code": None,
            "seller_tax_id": None,
            "total_amount": amount,
            "items": [
                {
                    "item_name": item_name,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "amount": amount,
                }
            ],
        }

    # 紙本電子發票 QR Code 模式：前 77 碼為固定表頭，之後為冒號分隔明細
    header = raw[:77]
    detail_section = raw[77:] if len(raw) > 77 else ""

    invoice_number = header[0:10].strip().upper() if len(header) >= 10 else ""
    roc_date = header[10:17].strip() if len(header) >= 17 else ""
    purchase_date = parse_roc_date_to_iso(roc_date)
    random_code = header[17:21].strip() if len(header) >= 21 else ""
    seller_tax_id = header[45:53].strip() if len(header) >= 53 else ""

    tokens = [decode_text_safely(t) for t in detail_section.split(":")]
    filtered_tokens = [t for t in tokens if t != "" and not t.startswith("**")]

    items = []
    idx = 0
    while idx + 2 < len(filtered_tokens):
        name_candidate = filtered_tokens[idx]
        qty_candidate = filtered_tokens[idx + 1]
        price_candidate = filtered_tokens[idx + 2]

        try:
            qty = float(qty_candidate)
            price = float(price_candidate)
            amount = round(qty * price, 2)
            items.append(
                {
                    "item_name": name_candidate,
                    "quantity": qty,
                    "unit_price": price,
                    "amount": amount,
                }
            )
            idx += 3
        except ValueError:
            idx += 1

    if not items and detail_section.strip():
        items.append(
            {
                "item_name": detail_section.strip(),
                "quantity": 1.0,
                "unit_price": 0.0,
                "amount": 0.0,
            }
        )

    total_amount = round(sum(item["amount"] for item in items), 2)

    return {
        "invoice_number": invoice_number or "未知字軌",
        "purchase_date": purchase_date,
        "random_code": random_code or None,
        "seller_tax_id": seller_tax_id or None,
        "total_amount": total_amount,
        "items": items,
    }

@app.post("/api/invoices/extract")
def extract_invoice(req: InvoiceExtractRequest):
    try:
        parsed = extract_invoice_content(req.qr_raw_data, req.source_type)
        return {
            "status": "success",
            "phone": req.phone,
            "invoice_type": req.invoice_type,
            "source_type": req.source_type,
            "data": parsed,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)