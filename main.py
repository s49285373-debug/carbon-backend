import os
import base64
import json
import re
import urllib.request
import urllib.parse
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import bcrypt
import pymysql

from carbon_moenv_service import router as carbon_router
from ai_carbon_aligner import router as ai_carbon_router
from meal_service import router as meal_router

app = FastAPI(title="日常發票載具之個人碳足跡與隱形碳成本智慧盤查推估系統")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(carbon_router)
app.include_router(ai_carbon_router)
app.include_router(meal_router)


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


class InvoiceItemInput(BaseModel):
    item_name: str
    quantity: float = 1.0
    unit_price: float = 0.0
    amount: float = 0.0
    carbon_kg: float = 0.0


class InvoiceCreateRequest(BaseModel):
    phone: str = Field(..., description="會員手機號碼")
    invoice_number: str = Field(..., description="發票字軌號碼")
    purchase_date: str | None = Field(None, description="購買日期 YYYY/MM/DD")
    random_code: str | None = Field(None, description="隨機碼")
    seller_tax_id: str | None = Field(None, description="賣方統一編號")
    merchant_name: str = Field(..., description="店家與類別顯示字串")
    store_name: str | None = Field(None, description="店家名稱")
    category_name: str = Field("一般消費", description="消費類別")
    total_amount: float = Field(0.0, description="發票總金額")
    carbon_kg: float = Field(0.0, description="碳足跡 kg CO2e")
    hidden_carbon_cost: float = Field(0.0, description="隱形碳成本 元")
    calc_label: str | None = Field(None, description="計算方式說明")
    source_type: str = Field("qr", description="來源類型")
    items: list[InvoiceItemInput] = Field(default_factory=list, description="逐項商品明細")


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


ITEM_CARBON_FACTORS = {
    "牛肉": 0.0270,
    "豬肉": 0.0072,
    "雞": 0.0057,
    "炸雞": 0.0065,
    "套餐": 0.0028,
    "年糕": 0.0022,
    "薯條": 0.0024,
    "奶": 0.0032,
    "茶": 0.0012,
    "咖啡": 0.0020,
    "杯麵": 0.0021,
    "麵": 0.0019,
    "飯糰": 0.0018,
    "便當": 0.0025,
}

NON_CARBON_KEYWORDS = ("代收", "代繳", "手續費", "跨行", "物流費", "運費")

CATEGORY_INTENSITY = {
    "餐飲食品": 0.0022,
    "餐飲消費": 0.0022,
    "便利商店": 0.0018,
    "一般消費": 0.0015,
}

TRACK_CATEGORY_MAP: dict[str, str] = {
    "FT": "一般消費",
    "FG": "餐飲消費",
}

KNOWN_MERCHANT_MAP: dict[str, tuple[str, str]] = {
    "80344914": ("統一超商", "便利商店"),
    "22618874": ("全家便利商店", "便利商店"),
    "53152396": ("萊爾富", "便利商店"),
    "22099131": ("OK超商", "便利商店"),
}

HIDDEN_CARBON_COST_PER_KG = 3.2


def format_date_slash(date_str: str | None) -> str | None:
    if not date_str:
        return None
    cleaned = date_str.strip().replace("-", "/")
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", cleaned):
        parts = cleaned.split("/")
        return f"{int(parts[0]):04d}/{int(parts[1]):02d}/{int(parts[2]):02d}"
    return cleaned


def estimate_item_carbon(item_name: str, amount: float) -> float:
    if amount <= 0:
        return 0.0
    if any(kw in item_name for kw in NON_CARBON_KEYWORDS):
        return 0.0
    for kw, factor in ITEM_CARBON_FACTORS.items():
        if kw in item_name:
            return round(amount * factor, 4)
    return round(amount * CATEGORY_INTENSITY["一般消費"], 4)


def fetch_merchant_from_public_api(seller_tax_id: str) -> tuple[str | None, str | None]:
    if not re.fullmatch(r"\d{8}", seller_tax_id):
        return None, None

    url = (
        "https://data.gcis.nat.gov.tw/od/data/api/5F64D864-61CB-4D0D-8AD9-492047CC1EA6"
        f"?$format=json&$filter=Business_Accounting_NO%20eq%20{urllib.parse.quote(seller_tax_id)}&$skip=0&$top=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, list) and len(payload) > 0:
                row = payload[0]
                name = (
                    row.get("Business_Name")
                    or row.get("Company_Name")
                    or ""
                ).strip()
                industry = str(row.get("Industry") or row.get("Business_Item") or "")
                category = (
                    "餐飲消費"
                    if any(k in industry for k in ("餐飲", "食品", "飲料", "咖啡"))
                    else "一般消費"
                )
                if name:
                    return name, category
    except Exception:
        pass
    return None, None


def lookup_merchant_category(
    seller_tax_id: str | None,
    invoice_number: str = "",
    connection=None,
) -> tuple[str | None, str]:
    prefix = invoice_number[:2].strip().upper() if len(invoice_number) >= 2 else ""
    default_category = TRACK_CATEGORY_MAP.get(prefix, "一般消費")

    if not seller_tax_id:
        return None, default_category

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

    fetched_name, fetched_cat = fetch_merchant_from_public_api(tax_key)
    if not fetched_name and tax_key in KNOWN_MERCHANT_MAP:
        fetched_name, fetched_cat = KNOWN_MERCHANT_MAP[tax_key]

    if fetched_name:
        final_cat = fetched_cat or default_category
        if connection is not None:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO merchant_categories (seller_tax_id, merchant_name, category_name)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            merchant_name = VALUES(merchant_name),
                            category_name = VALUES(category_name)
                        """,
                        (tax_key, fetched_name, final_cat),
                    )
                connection.commit()
            except Exception:
                pass
        return fetched_name, final_cat

    return None, default_category


def decode_text_safely(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if len(cleaned) >= 4 and len(cleaned) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", cleaned):
        try:
            decoded_bytes = base64.b64decode(cleaned, validate=True)
            for enc in ("utf-8", "big5", "cp950"):
                try:
                    decoded_str = decoded_bytes.decode(enc).strip()
                    if (
                        decoded_str
                        and "\ufffd" not in decoded_str
                        and all(ord(ch) >= 32 or ch in "\r\n\t" for ch in decoded_str)
                    ):
                        return decoded_str
                except UnicodeDecodeError:
                    continue
        except Exception:
            pass

    return cleaned


def parse_roc_date_to_slash(roc_7digits: str) -> str | None:
    if len(roc_7digits) == 7 and roc_7digits.isdigit():
        roc_year = int(roc_7digits[0:3])
        month = int(roc_7digits[3:5])
        day = int(roc_7digits[5:7])
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{roc_year + 1911:04d}/{month:02d}/{day:02d}"
    return None


def parse_qr_header_amount(header: str) -> float:
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
    cleaned = name.strip()
    if not cleaned:
        return False
    if re.fullmatch(r"\*+", cleaned):
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        return False
    return True


def extract_invoice_content(raw_data: str, source_type: str = "qr", db_conn=None) -> dict:
    raw = decode_text_safely(raw_data)
    if not raw:
        raise ValueError("發票內容為空，無法解析")

    if source_type == "manual":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invoiceNumber": raw[:10]}

        invoice_number = str(
            payload.get("invoiceNumber") or payload.get("invoice_number") or ""
        ).strip().upper()
        raw_date = str(
            payload.get("invoiceDate") or payload.get("purchase_date") or ""
        ).strip() or None
        purchase_date = format_date_slash(raw_date)
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
            "store_name": merchant_raw,
            "category_name": category_name,
            "total_amount": total_amount,
            "carbon_kg": carbon_kg,
            "hidden_carbon_cost": hidden_carbon_cost,
            "calc_label": f"無逐項明細，套用預設「{category_name}」係數推估",
            "items": [],
        }

    header = raw[:77]
    detail_section = raw[77:] if len(raw) > 77 else ""

    invoice_number = header[0:10].strip().upper() if len(header) >= 10 else ""
    roc_date = header[10:17].strip() if len(header) >= 17 else ""
    purchase_date = parse_roc_date_to_slash(roc_date)
    random_code = header[17:21].strip() if len(header) >= 21 else ""
    seller_tax_id = header[45:53].strip() if len(header) >= 53 else ""
    header_amount = parse_qr_header_amount(header)

    decoded_detail = decode_text_safely(detail_section)
    raw_tokens = [decode_text_safely(t).strip() for t in decoded_detail.split(":")]
    filtered_tokens = [t for t in raw_tokens if t != "" and not re.fullmatch(r"\*+", t)]

    if len(filtered_tokens) >= 2 and filtered_tokens[0].isdigit() and filtered_tokens[1] in ("0", "1", "2"):
        filtered_tokens = filtered_tokens[2:]

    items = []
    idx = 0
    while idx + 2 < len(filtered_tokens):
        name_candidate = decode_text_safely(filtered_tokens[idx]).strip()
        qty_candidate = filtered_tokens[idx + 1].strip()
        price_candidate = filtered_tokens[idx + 2].strip()

        try:
            qty = float(qty_candidate)
            price = float(price_candidate)
            if not is_valid_item_name(name_candidate) or qty == 0 or price == 0:
                idx += 1
                continue

            amount = round(qty * price, 2)
            item_carbon = estimate_item_carbon(name_candidate, amount)
            items.append(
                {
                    "item_name": name_candidate,
                    "quantity": abs(qty),
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
        raw_items_sum = round(sum(item["amount"] for item in items), 2)
        total_amount = header_amount if header_amount > 0 else raw_items_sum
        if total_amount > 0 and abs(total_amount - raw_items_sum) >= 1:
            diff = round(total_amount - raw_items_sum, 2)
            items.append(
                {
                    "item_name": "促銷折扣／優惠折讓" if diff < 0 else "其他加項／差額",
                    "quantity": 1.0,
                    "unit_price": diff,
                    "amount": diff,
                    "carbon_kg": 0.0,
                }
            )
        carbon_kg = round(max(sum(max(item["carbon_kg"], 0.0) for item in items), 0.0), 2)
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
        "store_name": merchant_raw,
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


@app.post("/api/invoices")
def create_invoice(req: InvoiceCreateRequest):
    phone_clean = req.phone.strip()
    inv_no_clean = req.invoice_number.strip().upper()
    if not phone_clean or not inv_no_clean:
        raise HTTPException(status_code=400, detail="缺少手機號碼或發票號碼")

    formatted_date = format_date_slash(req.purchase_date) or ""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql_insert_inv = """
                INSERT INTO invoices (
                    phone, invoice_number, purchase_date, random_code,
                    seller_tax_id, merchant_name, store_name, category_name,
                    total_amount, carbon_kg, hidden_carbon_cost, calc_label, source_type
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    id = LAST_INSERT_ID(id),
                    purchase_date = VALUES(purchase_date),
                    random_code = VALUES(random_code),
                    seller_tax_id = VALUES(seller_tax_id),
                    merchant_name = VALUES(merchant_name),
                    store_name = VALUES(store_name),
                    category_name = VALUES(category_name),
                    total_amount = VALUES(total_amount),
                    carbon_kg = VALUES(carbon_kg),
                    hidden_carbon_cost = VALUES(hidden_carbon_cost),
                    calc_label = VALUES(calc_label),
                    source_type = VALUES(source_type)
            """
            cursor.execute(
                sql_insert_inv,
                (
                    phone_clean,
                    inv_no_clean,
                    formatted_date,
                    req.random_code,
                    req.seller_tax_id,
                    req.merchant_name,
                    req.store_name,
                    req.category_name,
                    req.total_amount,
                    req.carbon_kg,
                    req.hidden_carbon_cost,
                    req.calc_label,
                    req.source_type,
                ),
            )
            invoice_id = cursor.lastrowid

            cursor.execute("DELETE FROM invoice_items WHERE invoice_id = %s", (invoice_id,))
            if req.items:
                sql_insert_item = """
                    INSERT INTO invoice_items (
                        invoice_id, item_name, quantity, unit_price, amount, carbon_kg
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                item_rows = [
                    (
                        invoice_id,
                        decode_text_safely(it.item_name).strip(),
                        it.quantity,
                        it.unit_price,
                        it.amount,
                        it.carbon_kg,
                    )
                    for it in req.items
                ]
                cursor.executemany(sql_insert_item, item_rows)

            conn.commit()

        return {
            "status": "success",
            "message": "發票已儲存至雲端資料庫",
            "data": {
                "id": invoice_id,
                "phone": phone_clean,
                "invoice_number": inv_no_clean,
                "purchase_date": formatted_date,
            },
        }
    finally:
        conn.close()


@app.get("/api/invoices")
def list_invoices(phone: str = Query(..., description="會員手機號碼")):
    phone_clean = phone.strip()
    if not phone_clean:
        raise HTTPException(status_code=400, detail="請提供手機號碼")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM invoices
                WHERE phone = %s
                ORDER BY id DESC
                """,
                (phone_clean,),
            )
            invoices = cursor.fetchall()

            if not invoices:
                return {"status": "success", "data": []}

            invoice_ids = [inv["id"] for inv in invoices]
            placeholders = ",".join(["%s"] * len(invoice_ids))
            cursor.execute(
                f"""
                SELECT id, invoice_id, item_name, quantity, unit_price, amount, carbon_kg
                FROM invoice_items
                WHERE invoice_id IN ({placeholders})
                ORDER BY id ASC
                """,
                invoice_ids,
            )
            all_items = cursor.fetchall()

        items_by_inv: dict[int, list[dict]] = {inv_id: [] for inv_id in invoice_ids}
        for item in all_items:
            items_by_inv[item["invoice_id"]].append(
                {
                    "item_name": decode_text_safely(item["item_name"]).strip(),
                    "quantity": float(item["quantity"]),
                    "unit_price": float(item["unit_price"]),
                    "amount": float(item["amount"]),
                    "carbon_kg": float(item["carbon_kg"]),
                }
            )

        result = []
        for inv in invoices:
            result.append(
                {
                    "id": inv["id"],
                    "phone": inv["phone"],
                    "invoice_number": inv["invoice_number"],
                    "purchase_date": format_date_slash(inv.get("purchase_date")),
                    "random_code": inv.get("random_code"),
                    "seller_tax_id": inv.get("seller_tax_id"),
                    "merchant_name": inv["merchant_name"],
                    "store_name": inv.get("store_name"),
                    "category_name": inv["category_name"],
                    "total_amount": float(inv["total_amount"]),
                    "carbon_kg": float(inv["carbon_kg"]),
                    "hidden_carbon_cost": float(inv["hidden_carbon_cost"]),
                    "calc_label": inv.get("calc_label"),
                    "source_type": inv.get("source_type", "qr"),
                    "items": items_by_inv.get(inv["id"], []),
                }
            )

        return {"status": "success", "data": result}
    finally:
        conn.close()


@app.delete("/api/invoices/{invoice_id}")
def delete_invoice(invoice_id: int, phone: str = Query(..., description="會員手機號碼")):
    phone_clean = phone.strip()
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM invoices WHERE id = %s AND phone = %s",
                (invoice_id, phone_clean),
            )
            affected = cursor.rowcount
            conn.commit()

        if affected == 0:
            raise HTTPException(status_code=404, detail="找不到該筆發票或無權刪除")

        return {"status": "success", "message": "發票已刪除"}
    finally:
        conn.close()


@app.get("/api/ledger")
def get_combined_ledger(
    phone: str = Query(..., description="會員手機號碼"),
    start_date: str | None = Query(default=None, description="起始日期 YYYY/MM/DD"),
    end_date: str | None = Query(default=None, description="結束日期 YYYY/MM/DD"),
):
    from meal_service import list_meal_records

    phone_clean = phone.strip()
    if not phone_clean:
        raise HTTPException(status_code=400, detail="請提供手機號碼")

    meals_res = list_meal_records(phone=phone_clean, start_date=start_date, end_date=end_date)
    meal_records = meals_res.get("data") or []

    conn = get_db_connection()
    invoice_records = []
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM invoices WHERE phone = %s"
            params: list = [phone_clean]
            if start_date:
                sql += " AND purchase_date >= %s"
                params.append(format_date_slash(start_date))
            if end_date:
                sql += " AND purchase_date <= %s"
                params.append(format_date_slash(end_date))
            sql += " ORDER BY id DESC"
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            for r in rows:
                carbon_val = float(r.get("carbon_kg") or 0.0)
                invoice_records.append(
                    {
                        "id": r["id"],
                        "phone": r["phone"],
                        "title": r.get("merchant_name") or r.get("invoice_number") or "發票紀錄",
                        "date": format_date_slash(r.get("purchase_date")) or "",
                        "amount": float(r.get("total_amount") or 0.0),
                        "carbon_kg": carbon_val,
                        "hidden_carbon_cost": float(
                            r.get("hidden_carbon_cost") or round(carbon_val * HIDDEN_CARBON_COST_PER_KG, 2)
                        ),
                        "source_type": "invoice",
                    }
                )
    finally:
        conn.close()

    total_meal_carbon = round(sum(float(m.get("carbon_kg") or 0.0) for m in meal_records), 4)
    total_invoice_carbon = round(sum(float(inv.get("carbon_kg") or 0.0) for inv in invoice_records), 4)
    total_carbon_kg = round(total_meal_carbon + total_invoice_carbon, 4)
    total_hidden_cost = round(total_carbon_kg * HIDDEN_CARBON_COST_PER_KG, 2)

    return {
        "status": "success",
        "data": {
            "summary": {
                "total_carbon_kg": total_carbon_kg,
                "total_meal_carbon_kg": total_meal_carbon,
                "total_invoice_carbon_kg": total_invoice_carbon,
                "total_hidden_carbon_cost": total_hidden_cost,
                "meal_count": len(meal_records),
                "invoice_count": len(invoice_records),
            },
            "meals": meal_records,
            "invoices": invoice_records,
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)