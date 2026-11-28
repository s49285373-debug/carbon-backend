import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pymysql

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEED_FILES = [
    BASE_DIR / "moenv_cfp_seed_data.json",
    BASE_DIR / "moenv_cfp_extra_items.json",
]


def load_json_file(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_emission_factors(
    files: Optional[List[Path]] = None,
    include_extra_glob: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_files: List[Path] = list(files) if files else list(DEFAULT_SEED_FILES)

    if include_extra_glob:
        for extra_file in sorted(BASE_DIR.glob("moenv_cfp_extra*.json")):
            if extra_file.resolve() not in {p.resolve() for p in target_files}:
                target_files.append(extra_file)

    merged_by_code: Dict[str, Dict[str, Any]] = {}
    source_info: Dict[str, Any] = {
        "name": "環境部產品碳足跡資訊網",
        "url": "https://cfp.moenv.gov.tw/WebPage/Index.aspx",
        "announce_year": "115年",
    }

    loaded_files_count = 0
    for path in target_files:
        if not path.exists():
            continue
        data = load_json_file(path)
        loaded_files_count += 1

        if isinstance(data.get("source"), dict):
            source_info.update({k: v for k, v in data["source"].items() if v is not None})

        rows = data.get("emission_factors", [])
        if isinstance(rows, list):
            for item in rows:
                code = str(item.get("factor_code", "")).strip()
                if not code:
                    continue
                merged_by_code[code] = {
                    **item,
                    "_source_file": path.name,
                    "_source_name": item.get("source_name") or source_info.get("name"),
                    "_source_url": item.get("source_url") or source_info.get("url"),
                    "_announce_year": item.get("announce_year") or source_info.get("announce_year"),
                }

    print(f"已掃描並載入 {loaded_files_count} 份係數 JSON 檔案，合併後共 {len(merged_by_code)} 筆品項。")
    return list(merged_by_code.values()), source_info


def get_mysql_connection() -> pymysql.connections.Connection:
    host = os.getenv("MYSQL_HOST") or os.getenv("MYSQLHOST") or os.getenv("DB_HOST")
    port_raw = os.getenv("MYSQL_PORT") or os.getenv("MYSQLPORT") or os.getenv("DB_PORT")
    user = os.getenv("MYSQL_USER") or os.getenv("MYSQLUSER") or os.getenv("DB_USER")
    password = (
        os.getenv("MYSQL_PASSWORD")
        or os.getenv("MYSQLPASSWORD")
        or os.getenv("DB_PASSWORD")
    )
    database = (
        os.getenv("MYSQL_DATABASE")
        or os.getenv("MYSQLDATABASE")
        or os.getenv("DB_NAME")
    )

    db_url = os.getenv("MYSQL_URL") or os.getenv("MYSQL_PUBLIC_URL") or os.getenv("DATABASE_URL")
    if (not host or not user or not database) and db_url:
        parsed = urlparse(db_url)
        host = host or parsed.hostname or "127.0.0.1"
        port_raw = port_raw or str(parsed.port or 3306)
        user = user or (parsed.username or "root")
        password = password if password is not None else (parsed.password or "")
        database = database or parsed.path.lstrip("/")

    return pymysql.connect(
        host=host or "127.0.0.1",
        port=int(port_raw or "3306"),
        user=user or "root",
        password=password if password is not None else "",
        database=database or "carbon_db",
        charset="utf8mb4",
        autocommit=False,
    )


def seed_emission_factors(rows: List[Dict[str, Any]], source_info: Dict[str, Any]) -> int:
    if not rows:
        print("沒有可寫入的 emission_factors 資料。")
        return 0

    sql = """
    INSERT INTO emission_factors (
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
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON DUPLICATE KEY UPDATE
        factor_name = VALUES(factor_name),
        category_key = VALUES(category_key),
        category_label = VALUES(category_label),
        declared_unit = VALUES(declared_unit),
        unit_to_base_gram = VALUES(unit_to_base_gram),
        is_weight_unit = VALUES(is_weight_unit),
        kg_co2e = VALUES(kg_co2e),
        keywords = VALUES(keywords),
        source_name = VALUES(source_name),
        source_url = VALUES(source_url),
        announce_year = VALUES(announce_year)
    """

    default_source_name = source_info.get("name", "環境部產品碳足跡資訊網")
    default_source_url = source_info.get("url", "https://cfp.moenv.gov.tw/WebPage/Index.aspx")
    default_announce_year = source_info.get("announce_year", "115年")

    params_list = []
    for item in rows:
        raw_keywords = item.get("keywords", [])
        if isinstance(raw_keywords, list):
            keywords_str = ",".join(str(k).strip() for k in raw_keywords if str(k).strip())
        else:
            keywords_str = str(raw_keywords or "").strip()

        if item.get("kg_co2e") is not None:
            total_kg_co2e = float(item["kg_co2e"])
        else:
            direct_val = float(item.get("direct_kg_co2e") or 0.0)
            hidden_val = float(item.get("hidden_supply_kg_co2e") or 0.0)
            total_kg_co2e = round(direct_val + hidden_val, 6)

        params_list.append(
            (
                str(item["factor_code"]).strip(),
                str(item["factor_name"]).strip(),
                str(item["category_key"]).strip(),
                str(item["category_label"]).strip(),
                str(item["declared_unit"]).strip(),
                float(item.get("unit_to_base_gram", 1000.0)),
                int(item.get("is_weight_unit", 1)),
                total_kg_co2e,
                keywords_str,
                str(item.get("_source_name") or default_source_name),
                str(item.get("_source_url") or default_source_url),
                str(item.get("_announce_year") or default_announce_year),
            )
        )

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.executemany(sql, params_list)
        conn.commit()
        return len(params_list)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批次匯入環境部碳足跡係數種子與擴充資料至 MySQL"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="指定要匯入的 JSON 檔案路徑（可傳多個，以空白分隔）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    specified_files = [Path(p) for p in args.files] if args.files else None
    all_rows, src_info = collect_emission_factors(files=specified_files)
    count = seed_emission_factors(all_rows, src_info)
    print(f"已成功批次寫入／更新 {count} 筆環境部碳足跡係數資料至 MySQL。")