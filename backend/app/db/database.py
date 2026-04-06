"""
RECALIX Database Layer — Production Grade
==========================================
Added: file_hash column for deduplication
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.config import DB_PATH

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS media (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT    UNIQUE NOT NULL,
    filename        TEXT    NOT NULL,
    file_type       TEXT    NOT NULL CHECK(file_type IN ('image','video')),
    folder          TEXT    NOT NULL,
    width           INTEGER,
    height          INTEGER,
    file_size       INTEGER,
    file_hash       TEXT,
    created_at      TEXT,
    modified_at     TEXT,
    thumbnail_path  TEXT,
    embedding_index INTEGER UNIQUE,
    ocr_text        TEXT,
    filename_tokens TEXT,
    is_document     INTEGER,
    is_screenshot   INTEGER,
    pixel_area      INTEGER,
    indexed_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);


        CREATE INDEX IF NOT EXISTS idx_media_folder     ON media(folder);
        CREATE INDEX IF NOT EXISTS idx_media_file_type  ON media(file_type);
        CREATE INDEX IF NOT EXISTS idx_media_modified   ON media(modified_at);
        CREATE INDEX IF NOT EXISTS idx_media_emb_index  ON media(embedding_index);
        CREATE INDEX IF NOT EXISTS idx_media_hash       ON media(file_hash);

        CREATE TABLE IF NOT EXISTS indexing_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()


def upsert_media(record: Dict[str, Any]) -> int:
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO media
            (file_path, filename, file_type, folder, width, height,
             file_size, file_hash, created_at, modified_at, thumbnail_path, embedding_index,
             ocr_text, filename_tokens, is_document, is_screenshot, pixel_area)
        VALUES
            (:file_path, :filename, :file_type, :folder, :width, :height,
             :file_size, :file_hash, :created_at, :modified_at, :thumbnail_path, :embedding_index,
             :ocr_text, :filename_tokens, :is_document, :is_screenshot, :pixel_area)
        ON CONFLICT(file_path) DO UPDATE SET
            filename        = excluded.filename,
            file_type       = excluded.file_type,
            folder          = excluded.folder,
            width           = excluded.width,
            height          = excluded.height,
            file_size       = excluded.file_size,
            file_hash       = excluded.file_hash,
            created_at      = excluded.created_at,
            modified_at     = excluded.modified_at,
            thumbnail_path  = excluded.thumbnail_path,
            embedding_index = excluded.embedding_index,
            ocr_text        = excluded.ocr_text,
            filename_tokens = excluded.filename_tokens,
            is_document     = excluded.is_document,
            is_screenshot   = excluded.is_screenshot,
            pixel_area      = excluded.pixel_area,
            indexed_at      = datetime('now')
        RETURNING id
    """, record)
    row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def get_media_by_embedding_indices(indices: List[int]) -> List[Dict]:
    if not indices:
        return []
    conn = get_conn()
    placeholders = ",".join("?" * len(indices))
    rows = conn.execute(
        f"SELECT * FROM media WHERE embedding_index IN ({placeholders})", indices
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_embedding_indices_with_filters(
    file_type: Optional[str] = None,
    folder: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[int]:
    conn = get_conn()
    query = "SELECT embedding_index FROM media WHERE embedding_index IS NOT NULL"
    params = []
    if file_type:
        query += " AND file_type = ?"
        params.append(file_type)
    if folder:
        query += " AND folder LIKE ?"
        params.append(f"{folder}%")
    if date_from:
        query += " AND modified_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND modified_at <= ?"
        params.append(date_to)
    rows = conn.execute(query, params).fetchall()
    return [r["embedding_index"] for r in rows]


def get_all_folders() -> List[Dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT folder, COUNT(*) as count
        FROM media GROUP BY folder ORDER BY folder
    """).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN file_type='image' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN file_type='video' THEN 1 ELSE 0 END) as videos,
            MAX(indexed_at) as last_indexed
        FROM media
    """).fetchone()
    return dict(row) if row else {}


def get_state(key: str) -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT value FROM indexing_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(key: str, value: Any):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO indexing_state(key, value) VALUES (?, ?)",
        (key, json.dumps(value) if not isinstance(value, str) else value)
    )
    conn.commit()


def clear_all():
    conn = get_conn()
    conn.execute("DELETE FROM media")
    conn.execute("DELETE FROM indexing_state")
    conn.commit()
