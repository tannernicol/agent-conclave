"""RAG + NAS indexing integration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import fnmatch
import json
import os
import sqlite3
import time
import httpx


@dataclass
class RagClient:
    base_url: str

    def search(self, query: str, collection: Optional[str] = None, limit: int = 20, semantic: bool | None = None) -> list[dict]:
        params = {"q": query, "limit": limit}
        if collection:
            params["collection"] = collection
        if semantic is not None:
            params["semantic"] = str(semantic).lower()
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/rag/search", params=params)
                resp.raise_for_status()
                return resp.json().get("results", [])
        except Exception:
            return []

    def collections(self) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/rag/collections")
                resp.raise_for_status()
                return resp.json().get("collections", [])
        except Exception:
            return []

    def search_files(self, query: str, limit: int = 20, extension: Optional[str] = None) -> list[dict]:
        params = {"q": query, "limit": limit}
        if extension:
            params["extension"] = extension
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/search/files", params=params)
                resp.raise_for_status()
                return resp.json().get("results", [])
        except Exception:
            return []


class NasIndex:
    def __init__(
        self,
        data_dir: Path,
        allowlist: list[str],
        exclude_patterns: list[str],
        max_file_mb: int = 2,
    ) -> None:
        self.data_dir = data_dir
        self.allowlist = [Path(p) for p in allowlist]
        self.exclude_patterns = exclude_patterns
        self.max_bytes = max_file_mb * 1024 * 1024
        self.db_path = data_dir / "index" / "nas_index.db"

    def _excluded(self, path: Path) -> bool:
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def _init_db(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS files USING fts5(
                path, filename, content, extension, size,
                content='files_content', content_rowid='rowid'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files_content(
                rowid INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                filename TEXT,
                content TEXT,
                extension TEXT,
                size INTEGER,
                mtime REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON files_content(path)")

    def index(self) -> int:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        self._init_db(conn)
        indexed = 0
        for base in self.allowlist:
            if not base.exists():
                continue
            for root, dirs, files in os.walk(base):
                root_path = Path(root)
                if self._excluded(root_path):
                    dirs[:] = []
                    continue
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith('.') and not self._excluded(root_path / d)
                ]
                for name in files:
                    if name.startswith('.'):
                        continue
                    path = Path(root) / name
                    if self._excluded(path):
                        continue
                    ext = path.suffix.lower()
                    if ext not in {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".toml", ".pdf"}:
                        continue
                    try:
                        stat = path.stat()
                    except Exception:
                        continue
                    if stat.st_size > self.max_bytes:
                        continue
                    cursor = conn.execute("SELECT mtime FROM files_content WHERE path = ?", (str(path),))
                    row = cursor.fetchone()
                    if row and row[0] >= stat.st_mtime:
                        continue
                    try:
                        content = path.read_text(errors="ignore")[:50000]
                    except Exception:
                        content = ""
                    conn.execute(
                        """
                        INSERT INTO files_content (path, filename, content, extension, size, mtime)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(path) DO UPDATE SET
                            filename=excluded.filename, content=excluded.content,
                            extension=excluded.extension, size=excluded.size, mtime=excluded.mtime
                        """,
                        (str(path), name, content, ext, stat.st_size, stat.st_mtime),
                    )
                    indexed += 1
        conn.execute("INSERT INTO files(files) VALUES('rebuild')")
        conn.commit()
        conn.close()
        return indexed

    def search(self, query: str, limit: int = 20, extension: Optional[str] = None) -> list[dict]:
        if not self.db_path.exists():
            self.index()
        if not self.db_path.exists():
            return []
        results = []
        safe_query = self._fts_query(query)
        if not safe_query:
            return results
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        sql = """
            SELECT path, filename, snippet(files, 2, '<mark>', '</mark>', '...', 30) as snippet,
                   extension, size
            FROM files
            WHERE files MATCH ?
        """
        params = [safe_query]
        if extension:
            sql += " AND extension = ?"
            params.append(extension)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                results.append({
                    "type": "document",
                    "source": "nas",
                    "title": row["filename"],
                    "file_path": row["path"],
                    "snippet": row["snippet"],
                    "metadata": {
                        "extension": row["extension"],
                        "size": row["size"],
                    },
                })
        finally:
            conn.close()
        return results

    def _fts_query(self, query: str) -> str:
        import re
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        return " ".join(tokens)

    def stats(self) -> dict:
        if not self.db_path.exists():
            return {"total_files": 0}
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM files_content")
            total = cursor.fetchone()[0]
            return {"total_files": total}
        finally:
            conn.close()
