"""RAG + local file indexing integration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import fnmatch
import json
import os
import sqlite3
import time
import httpx
import logging

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".conclave" / "cache"


@dataclass
class RagClient:
    base_url: str
    timeout: float = 15.0  # Increased from 10s
    collections_cache_ttl: float = 3600.0  # 1 hour
    cache_dir: Path | None = None
    errors: list[dict] = field(default_factory=list)
    _collections_cache: list[dict] = field(default_factory=list)
    _collections_cache_time: float = 0.0

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = DEFAULT_CACHE_DIR
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If cache dir isn't writable, operate in-memory only.
            self.cache_dir = None

    def _record_error(self, action: str, exc: Exception) -> None:
        self.errors.append({"action": action, "error": str(exc), "time": time.time()})

    def drain_errors(self) -> list[dict]:
        errors = list(self.errors)
        self.errors.clear()
        return errors

    def _load_collections_cache(self, stale_ok: bool = False) -> list[dict] | None:
        """Load collections from disk cache if valid."""
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / "rag_collections.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            cache_time = data.get("time", 0)
            if stale_ok or (time.time() - cache_time < self.collections_cache_ttl):
                return data.get("collections", [])
        except Exception:
            pass
        return None

    def _save_collections_cache(self, collections: list[dict]) -> None:
        """Save collections to disk cache."""
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / "rag_collections.json"
        try:
            cache_file.write_text(json.dumps({
                "time": time.time(),
                "collections": collections,
            }))
        except Exception as e:
            logger.debug(f"Failed to save collections cache: {e}")

    def search(self, query: str, collection: Optional[str] = None, limit: int = 20, semantic: bool | None = None) -> list[dict]:
        params = {"q": query, "limit": limit}
        if collection:
            params["collection"] = collection
        if semantic is not None:
            params["semantic"] = str(semantic).lower()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/api/rag/search", params=params)
                resp.raise_for_status()
                return resp.json().get("results", [])
        except httpx.TimeoutException as exc:
            self._record_error(f"search:{collection or 'all'}", exc)
            logger.warning(f"RAG search timed out after {self.timeout}s for collection={collection}")
            return []
        except Exception as exc:
            self._record_error(f"search:{collection or 'all'}", exc)
            logger.warning("RAG search failed", exc_info=True)
            return []

    def collections(self) -> list[dict]:
        # Check memory cache first
        now = time.time()
        if self._collections_cache_time and (now - self._collections_cache_time) < self.collections_cache_ttl:
            return self._collections_cache

        # Check disk cache
        disk_cache = self._load_collections_cache()
        if disk_cache:
            self._collections_cache = disk_cache
            self._collections_cache_time = now
            return disk_cache

        # Fetch from server
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/rag/collections")
                resp.raise_for_status()
                collections = resp.json().get("collections", [])
                self._collections_cache = collections
                self._collections_cache_time = now
                self._save_collections_cache(collections)
                return collections
        except Exception as exc:
            self._record_error("collections", exc)
            logger.warning("RAG collections fetch failed", exc_info=True)
            # Return memory cache if available, even if stale
            if self._collections_cache_time:
                return self._collections_cache
            # Try disk cache even if expired
            disk_cache = self._load_collections_cache(stale_ok=True)
            if disk_cache:
                return disk_cache
            return []

    def search_files(self, query: str, limit: int = 20, extension: Optional[str] = None) -> list[dict]:
        params = {"q": query, "limit": limit}
        if extension:
            params["extension"] = extension
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/api/search/files", params=params)
                resp.raise_for_status()
                return resp.json().get("results", [])
        except httpx.TimeoutException as exc:
            self._record_error("search_files", exc)
            logger.warning(f"RAG file search timed out after {self.timeout}s")
            return []
        except Exception as exc:
            self._record_error("search_files", exc)
            logger.warning("RAG file search failed", exc_info=True)
            return []

    def health_check(self) -> bool:
        """Check if RAG server is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False


class FileIndex:
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
        self.db_path = data_dir / "index" / "file_index.db"

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
                    "source": "file_index",
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
