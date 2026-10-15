"""
RECALIX Database Layer — Production Grade
==========================================
Added: file_hash column for deduplication
"""

import sqlite3
import json
import threading
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

        CREATE TABLE IF NOT EXISTS search_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            folder      TEXT,
            date_from   TEXT,
            date_to     TEXT,
            extensions  TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at);
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
    extensions: Optional[List[str]] = None,
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
    if extensions:
        ext_parts = []
        for ext in extensions:
            ext = ext.lower().strip()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            ext_parts.append("LOWER(file_path) LIKE ?")
            params.append(f"%{ext}")
        if ext_parts:
            query += " AND (" + " OR ".join(ext_parts) + ")"
    rows = conn.execute(query, params).fetchall()
    return [r["embedding_index"] for r in rows]


def get_all_folders() -> List[Dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT folder, COUNT(*) as count
        FROM media GROUP BY folder ORDER BY folder
    """).fetchall()
    return [dict(r) for r in rows]


def get_media_gallery(
    folder: Optional[str] = None,
    file_type: Optional[str] = "image",
    page: int = 1,
    page_size: int = 48,
    extensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Paged media browser for gallery views."""
    conn = get_conn()
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    where = ["embedding_index IS NOT NULL"]
    params: List[Any] = []
    if folder:
        where.append("folder = ?")
        params.append(folder)
    if file_type in ("image", "video"):
        where.append("file_type = ?")
        params.append(file_type)
    if extensions:
        ext_parts = []
        for ext in extensions:
            ext = ext.lower().strip()
            if not ext:
                continue
            if not ext.startswith("."):
                ext = f".{ext}"
            ext_parts.append("LOWER(file_path) LIKE ?")
            params.append(f"%{ext}")
        if ext_parts:
            where.append("(" + " OR ".join(ext_parts) + ")")

    where_sql = " AND ".join(where)
    total = conn.execute(
        f"SELECT COUNT(*) AS count FROM media WHERE {where_sql}", params
    ).fetchone()["count"]
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""
        SELECT *
        FROM media
        WHERE {where_sql}
        ORDER BY COALESCE(modified_at, indexed_at) DESC, id DESC
        LIMIT ? OFFSET ?
        """,
        [*params, page_size, offset],
    ).fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}


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


def add_search_history(
    query: str,
    folder: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    extensions: Optional[str] = None,
    retention_days: int = 10,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO search_history (query, folder, date_from, date_to, extensions)
        VALUES (?, ?, ?, ?, ?)
        """,
        (query, folder, date_from, date_to, extensions),
    )
    conn.execute(
        "DELETE FROM search_history WHERE created_at < datetime('now', ?)",
        (f"-{retention_days} days",),
    )
    conn.commit()
    return cur.lastrowid


def get_search_history(limit: int = 3) -> List[Dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, query, folder, date_from, date_to, extensions, created_at
        FROM search_history
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def clear_search_history() -> int:
    conn = get_conn()
    cur = conn.execute("DELETE FROM search_history")
    conn.commit()
    return cur.rowcount


def delete_search_history_item(item_id: int) -> int:
    conn = get_conn()
    cur = conn.execute("DELETE FROM search_history WHERE id = ?", (item_id,))
    conn.commit()
    return cur.rowcount


def delete_media_by_folder(folder: str, include_subfolders: bool = True) -> Dict[str, Any]:
    conn = get_conn()
    if include_subfolders:
        where_sql = "folder LIKE ?"
        params = (f"{folder}%",)
    else:
        where_sql = "folder = ?"
        params = (folder,)

    rows = conn.execute(
        f"SELECT id, embedding_index, thumbnail_path FROM media WHERE {where_sql}",
        params,
    ).fetchall()
    if not rows:
        return {"deleted": 0, "embedding_indices": [], "thumbnails": []}

    conn.execute(f"DELETE FROM media WHERE {where_sql}", params)
    conn.commit()

    embedding_indices = [r["embedding_index"] for r in rows if r["embedding_index"] is not None]
    thumbnails = [r["thumbnail_path"] for r in rows if r["thumbnail_path"]]
    return {"deleted": len(rows), "embedding_indices": embedding_indices, "thumbnails": thumbnails}
