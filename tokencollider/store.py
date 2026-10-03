"""SQLite embedding cache, keyed by content hash of (model, layer, pooling, text)."""

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    hash       TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    layer      TEXT NOT NULL,
    pooling    TEXT NOT NULL,
    template   TEXT NOT NULL,
    text       TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_slice
    ON embeddings(model, layer, pooling, template);
CREATE TABLE IF NOT EXISTS images (
    sha        TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    name       TEXT NOT NULL,
    thumb      BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conditionings (
    hash       TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    layer      TEXT NOT NULL,
    template   TEXT NOT NULL,
    text       TEXT NOT NULL,
    seq_len    INTEGER NOT NULL,
    dim        INTEGER NOT NULL,
    data       BLOB NOT NULL,
    created_at REAL NOT NULL
);

-- Which weights filled this model's rows. The cache key names the model by
-- path only, so a different checkpoint swapped in at the same path would
-- otherwise be served the old one's vectors. See Embedder.check_weights.
CREATE TABLE IF NOT EXISTS model_weights (
    model       TEXT PRIMARY KEY,
    identity    TEXT NOT NULL,
    recorded_at REAL NOT NULL
);
"""

FORGET_CHUNK = 500


def content_hash(model: str, layer: str, pooling: str, template: str, text: str) -> str:
    key = f"{model}|{layer}|{pooling}|{template}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class EmbeddingStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Shared across ThreadingHTTPServer worker threads; the lock serializes
        # all access since sqlite3 connections are not thread-safe themselves.
        # Reentrant so bulk() can hold it across the puts it wraps.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self._in_bulk = False
        self.conn.executescript(SCHEMA)

    def _commit(self) -> None:
        if not self._in_bulk:
            self.conn.commit()

    @contextmanager
    def bulk(self):
        """Group many puts into one transaction (one fsync instead of one per
        put — a full warm is 38 puts, and fsync latency dominates warm-time
        bookkeeping). Reentrant: nested bulks join the outermost transaction."""
        with self.lock:
            if self._in_bulk:
                yield
                return
            self._in_bulk = True
            try:
                yield
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise
            finally:
                self._in_bulk = False

    def get(self, model: str, layer: str, pooling: str, template: str, text: str) -> np.ndarray | None:
        h = content_hash(model, layer, pooling, template, text)
        with self.lock:
            row = self.conn.execute(
                "SELECT dim, vec FROM embeddings WHERE hash = ?", (h,)
            ).fetchone()
        if row is None:
            return None
        dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()

    def put(self, model: str, layer: str, pooling: str, template: str, text: str,
            vec: np.ndarray) -> None:
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        h = content_hash(model, layer, pooling, template, text)
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (h, model, layer, pooling, template, text, vec.shape[0],
                 vec.tobytes(), time.time()),
            )
            self._commit()

    def get_conditioning(self, model: str, layer: str, template: str, text: str) -> np.ndarray | None:
        h = content_hash(model, layer, "tokens", template, text)
        with self.lock:
            row = self.conn.execute(
                "SELECT seq_len, dim, data FROM conditionings WHERE hash = ?", (h,)
            ).fetchone()
        if row is None:
            return None
        seq_len, dim, blob = row
        return np.frombuffer(blob, dtype=np.float32).reshape(seq_len, dim).copy()

    def put_conditioning(self, model: str, layer: str, template: str, text: str,
                         tensor: np.ndarray) -> None:
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        h = content_hash(model, layer, "tokens", template, text)
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO conditionings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (h, model, layer, template, text, tensor.shape[0], tensor.shape[1],
                 tensor.tobytes(), time.time()),
            )
            self._commit()

    def count_for(self, model: str, layer: str, pooling: str, template: str) -> int:
        """Row count for one (model, layer, pooling, template) slice — the
        cheap invalidation probe for in-memory copies of vectors_for()."""
        with self.lock:
            (n,) = self.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = ? AND layer = ? "
                "AND pooling = ? AND template = ?",
                (model, layer, pooling, template),
            ).fetchone()
        return n

    def vectors_for(self, model: str, layer: str, pooling: str,
                    template: str) -> tuple[list[str], np.ndarray | None]:
        """Every cached (text, vector) under one config slice, stacked —
        the interrogator's vocabulary corpus. Returns ([], None) when empty."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT text, dim, vec FROM embeddings WHERE model = ? AND "
                "layer = ? AND pooling = ? AND template = ? ORDER BY text",
                (model, layer, pooling, template),
            ).fetchall()
        if not rows:
            return [], None
        texts = [r[0] for r in rows]
        mat = np.stack([
            np.frombuffer(blob, dtype=np.float32).reshape(dim) for _t, dim, blob in rows
        ])
        return texts, mat

    def texts_for(self, model: str) -> list[str]:
        """Every distinct cached phrase under one model — the sweep surface
        for pattern-based hygiene (tokencollider forget --pattern)."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT DISTINCT text FROM embeddings WHERE model = ?", (model,)
            ).fetchall()
        return [r[0] for r in rows]

    def forget(self, model: str, texts: list[str]) -> tuple[int, int]:
        """Delete every cached row (all layers/poolings/templates) for these
        exact phrases under one model. Returns (embedding_rows, conditioning_
        rows) removed. Note: DELETE alone doesn't scrub bytes from the file —
        run vacuum() afterwards to actually reclaim and overwrite."""
        texts = list(dict.fromkeys(texts))
        e = c = 0
        # Chunked: SQLite caps bound parameters per statement (999 on older
        # builds), and a pattern sweep can match the whole vocabulary.
        with self.lock:
            for i in range(0, len(texts), FORGET_CHUNK):
                chunk = texts[i:i + FORGET_CHUNK]
                marks = ",".join("?" * len(chunk))
                e += self.conn.execute(
                    f"DELETE FROM embeddings WHERE model = ? AND text IN ({marks})",
                    (model, *chunk)).rowcount
                c += self.conn.execute(
                    f"DELETE FROM conditionings WHERE model = ? AND text IN ({marks})",
                    (model, *chunk)).rowcount
            self._commit()
        return e, c

    def weights_identity(self, model: str) -> str | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT identity FROM model_weights WHERE model = ?", (model,)).fetchone()
        return row[0] if row else None

    def set_weights_identity(self, model: str, identity: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO model_weights (model, identity, recorded_at) "
                "VALUES (?, ?, ?)", (model, identity, time.time()))
            self._commit()

    def forget_model(self, model: str) -> tuple[int, int]:
        """Delete every cached row for one model, and its recorded weights.
        Returns (embedding_rows, conditioning_rows) removed."""
        with self.lock:
            e = self.conn.execute("DELETE FROM embeddings WHERE model = ?", (model,)).rowcount
            c = self.conn.execute("DELETE FROM conditionings WHERE model = ?", (model,)).rowcount
            self.conn.execute("DELETE FROM model_weights WHERE model = ?", (model,))
            self._commit()
        return e, c

    def vacuum(self) -> None:
        with self.lock:
            self.conn.execute("VACUUM")

    def put_image(self, sha: str, path: str, name: str, thumb: bytes) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?)",
                (sha, path, name, thumb, time.time()))
            self._commit()

    def get_image(self, sha: str) -> tuple[str, str, bytes] | None:
        """(path, name, thumbnail jpeg bytes) for a registered image."""
        with self.lock:
            row = self.conn.execute(
                "SELECT path, name, thumb FROM images WHERE sha = ?", (sha,)
            ).fetchone()
        return None if row is None else (row[0], row[1], bytes(row[2]))

    def layers_cached(self, model: str) -> list[str]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT DISTINCT layer FROM embeddings WHERE model = ?", (model,)
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        with self.lock:
            self.conn.close()
