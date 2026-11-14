import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import bcrypt
import pymysql

app = FastAPI(title="日常發票載具之個人碳足跡系統 API")

# 1. 設定 CORS (允許來自 Vercel 雲端前端與本機的跨網域呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 專題上線測試期間可先設為 ["*"] 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 連線 MySQL 資料庫的函式 (自動適配本機或雲端環境變數)
def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),      # 本機預設空密碼，雲端則讀取環境變數
        database=os.getenv("DB_NAME", "carbon_db"),
        port=int(os.getenv("DB_PORT", 3306)),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# 3. 定義前端傳入的註冊資料格式
class RegisterRequest(BaseModel):
    phone: str = Field(..., description="手機號碼")
    password: str = Field(..., description="密碼")
    nickname: str = Field(..., description="暱稱")
    role: str = Field(..., description="身分")

# 4. 註冊 API 端點
@app.post("/api/register")
def register_user(user: RegisterRequest):
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 檢查手機號碼是否已被註冊
            sql_check = "SELECT * FROM users WHERE phone = %s"
            cursor.execute(sql_check, (user.phone,))
            existing_user = cursor.fetchone()
            
            if existing_user:
                raise HTTPException(status_code=400, detail="此手機號碼已被註冊過！")
            
            # 密碼進行 Hash 加密 (符合專題資安規格)
            hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # 寫入 MySQL 資料庫的 users 資料表
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)