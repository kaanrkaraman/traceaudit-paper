"""SQLite-backed prompt cache — the determinism contract per D21 / D23.

A cache hit returns a previously-captured response byte-for-byte; a cache
miss in `STRICT_REPLAY` mode raises `ReplayCacheMissError` (D23) and is
never silently re-queried. The cache stores the structural fields needed
to reconstruct a `GenerationEvent`'s response — including `tool_calls`,
which are part of the v0.1.2 `outputs_hash` payload (D25).

Per D26 the cache lands before the replayer (step 5d before 5c); it is
unit-testable in isolation against a `tmp_path / *.sqlite` DB.

Request-key derivation
----------------------

The cache key is the SHA256 of canonical-JSON of
``(prompt_hash, model_id, model_provider, model_version, temperature,
top_p, seed, max_tokens)``. Both ``model_id`` and ``model_provider`` are
pinned because the same nominal model under different providers — Azure
OpenAI ``gpt-4o-mini`` vs OpenAI ``gpt-4o-mini`` — is a different
deployment that can drift in behavior (B-adjustment to D25's request-key
spec).

Modes
-----

- ``CacheMode.RECORD``: miss returns ``None``; the caller does the real
  API call and writes back via ``put``. Used during initial trace capture.
- ``CacheMode.STRICT_REPLAY``: miss raises ``ReplayCacheMissError`` with
  the prompt hash and the full request payload attached. Used during
  replay. No silent fallback — D21 forbids it.

SQLite tuning
-------------

- WAL journal mode — concurrent readers and one writer; readers do not
  block on writes in flight.
- ``PRAGMA synchronous=NORMAL`` — fsync at WAL checkpoint, not every
  commit. Durable across process crashes that are not power-loss.
- ``sqlite3.connect(..., timeout=5.0)`` — a second writer on the same
  database waits up to five seconds for the first to commit rather than
  failing with ``SQLITE_BUSY`` immediately. Applied at the ``connect()``
  call so it covers the early ``PRAGMA journal_mode=WAL`` that flips a
  fresh DB's journal mode and would otherwise contend unprotected.
- ``cache_schema_version`` table seeded on first open. Mismatch on
  subsequent open raises ``CacheSchemaVersionMismatchError``; migrations
  are explicit, never automatic.
"""

from __future__ import annotations

import hashlib
import sqlite3
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from traceaudit.trace.exceptions import ReplayCacheMissError
from traceaudit.trace.schema import (
    GenerationEvent,
    ModelProvider,
    canonical_json,
    utc_now,
)

CACHE_SCHEMA_VERSION: str = "0.1.0"
"""Cache-schema version. Bumped on incompatible table-shape changes; mismatch on open raises."""


class CacheMode(str, Enum):
    """Cache lookup mode — controls miss behavior."""

    RECORD = "record"
    STRICT_REPLAY = "strict_replay"


class CacheSchemaVersionMismatchError(RuntimeError):
    """Raised on opening a cache DB whose schema version disagrees with the code."""

    def __init__(self, *, found: str, expected: str, db_path: Path) -> None:
        self.found = found
        self.expected = expected
        self.db_path = db_path
        super().__init__(
            f"cache schema version mismatch at {db_path}: found {found!r}, "
            f"expected {expected!r}. Migrations are explicit; delete the cache "
            "file or migrate manually before continuing."
        )


class _Frozen(BaseModel):
    """Local immutable Pydantic base — mirrors the schema module's pattern."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
    )


class CacheRequest(_Frozen):
    """Structural identity of a model request, hashed to a cache key.

    Mirrors the determinism-relevant fields of `GenerationEvent`: prompt
    identity, model identity (id + provider + version), and decode params.
    Two requests with the same `CacheRequest` are guaranteed to share a
    `request_key()` and hit the same cache row.
    """

    prompt_hash: str
    model_id: str
    model_provider: ModelProvider
    model_version: str | None = None
    temperature: float
    top_p: float
    seed: int | None = None
    max_tokens: int = Field(gt=0)

    @classmethod
    def from_event(cls, ev: GenerationEvent) -> "CacheRequest":
        """Build a `CacheRequest` from a fully-formed `GenerationEvent`.

        Useful symmetrically: the recorder calls this to write into the
        cache as it captures, the replayer calls it to look up the same
        event during a deterministic replay.
        """
        return cls(
            prompt_hash=ev.prompt_hash,
            model_id=ev.model_id,
            model_provider=ev.model_provider,
            model_version=ev.model_version,
            temperature=ev.temperature,
            top_p=ev.top_p,
            seed=ev.seed,
            max_tokens=ev.max_tokens,
        )

    def request_key(self) -> str:
        """SHA256 of the canonical-JSON of this request. The cache primary key."""
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Hashable / replayable dict form. Enum-typed provider becomes its value."""
        return {
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "model_provider": self.model_provider.value,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }


class CachedResponse(_Frozen):
    """The byte-stable response payload returned on a cache hit."""

    response_text: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    finish_reason: str
    logprobs_json: str | None = None
    tool_calls_json: str | None = None


_VERSION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS cache_schema_version (
    version TEXT PRIMARY KEY
)
"""

_CACHE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS prompt_cache (
    request_key TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_version TEXT,
    temperature REAL NOT NULL,
    top_p REAL NOT NULL,
    seed INTEGER,
    max_tokens INTEGER NOT NULL,
    response_text TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    finish_reason TEXT NOT NULL,
    logprobs_json TEXT,
    tool_calls_json TEXT,
    created_at TEXT NOT NULL
)
"""

_PROMPT_HASH_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_prompt_hash ON prompt_cache (prompt_hash)
"""


class PromptCache:
    """SQLite prompt cache — the closed-model determinism contract.

    Stateful: holds one ``sqlite3.Connection`` for the lifetime of the
    instance. Use as a context manager (``with PromptCache(...) as cache:``)
    or call ``close()`` explicitly. Multiple instances may target the
    same ``db_path`` concurrently: WAL serializes writers; readers do
    not block. Each instance is bound to its constructing thread
    (sqlite3's default ``check_same_thread=True``); cross-thread reuse
    requires a fresh instance per thread.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # `timeout=5.0` on connect() is the busy_timeout in effect from the
        # first SQL byte — including the `PRAGMA journal_mode=WAL` that flips
        # a fresh DB's journal mode. Setting busy_timeout via PRAGMA later
        # would leave the early PRAGMAs unprotected against lock contention.
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(db_path), timeout=5.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL").fetchall()
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._hits = 0
        self._misses = 0
        self._ensure_schema()

    # ---- Public API ----------------------------------------------------------

    def get(
        self,
        request: CacheRequest,
        *,
        mode: CacheMode,
    ) -> CachedResponse | None:
        """Look up `request`. Hit → stored response; miss behavior depends on `mode`.

        - `CacheMode.RECORD` miss returns `None`.
        - `CacheMode.STRICT_REPLAY` miss raises `ReplayCacheMissError`
          with `request.prompt_hash` and `request.as_dict()` attached.
        """
        conn = self._require_open()
        request_key = request.request_key()
        row = conn.execute(
            "SELECT response_text, prompt_tokens, completion_tokens, cost_usd, "
            "latency_ms, finish_reason, logprobs_json, tool_calls_json "
            "FROM prompt_cache WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        if row is None:
            self._misses += 1
            if mode == CacheMode.STRICT_REPLAY:
                raise ReplayCacheMissError(
                    prompt_hash=request.prompt_hash,
                    request_payload=request.as_dict(),
                )
            return None
        self._hits += 1
        return CachedResponse(
            response_text=row[0],
            prompt_tokens=row[1],
            completion_tokens=row[2],
            cost_usd=row[3],
            latency_ms=row[4],
            finish_reason=row[5],
            logprobs_json=row[6],
            tool_calls_json=row[7],
        )

    def put(self, request: CacheRequest, response: CachedResponse) -> None:
        """Insert-or-replace the cached response for `request`.

        Replacement is safe under the determinism contract: `request_key`
        depends on every input that should change the response, so a
        replacement under the same key with the same model semantics
        should be byte-identical. Re-running RECORD mode against a
        non-empty cache is therefore idempotent.
        """
        conn = self._require_open()
        conn.execute(
            "INSERT OR REPLACE INTO prompt_cache ("
            "request_key, prompt_hash, model_id, model_provider, model_version, "
            "temperature, top_p, seed, max_tokens, "
            "response_text, prompt_tokens, completion_tokens, "
            "cost_usd, latency_ms, finish_reason, logprobs_json, tool_calls_json, "
            "created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.request_key(),
                request.prompt_hash,
                request.model_id,
                request.model_provider.value,
                request.model_version,
                request.temperature,
                request.top_p,
                request.seed,
                request.max_tokens,
                response.response_text,
                response.prompt_tokens,
                response.completion_tokens,
                response.cost_usd,
                response.latency_ms,
                response.finish_reason,
                response.logprobs_json,
                response.tool_calls_json,
                utc_now().isoformat(),
            ),
        )
        conn.commit()

    def stats(self) -> dict[str, float]:
        """Per-instance hit/miss accounting. Counters are not persisted to disk.

        Returned dict keys: ``hits`` (int), ``misses`` (int), ``total`` (int),
        ``hit_rate`` (float in [0, 1]; 0.0 if no lookups). Read at the end
        of a replay session to evaluate H0.3's ≥95% target.
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
        }

    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "PromptCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- Internals -----------------------------------------------------------

    def _require_open(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                f"PromptCache at {self.db_path} has been closed; "
                "open a new instance to continue"
            )
        return self._conn

    def _ensure_schema(self) -> None:
        assert self._conn is not None  # init-time invariant
        self._conn.execute(_VERSION_TABLE_DDL)
        rows = self._conn.execute(
            "SELECT version FROM cache_schema_version"
        ).fetchall()
        if not rows:
            self._conn.execute(
                "INSERT INTO cache_schema_version (version) VALUES (?)",
                (CACHE_SCHEMA_VERSION,),
            )
        elif rows[0][0] != CACHE_SCHEMA_VERSION:
            found = rows[0][0]
            self._conn.close()
            self._conn = None
            raise CacheSchemaVersionMismatchError(
                found=found,
                expected=CACHE_SCHEMA_VERSION,
                db_path=self.db_path,
            )
        self._conn.execute(_CACHE_TABLE_DDL)
        self._conn.execute(_PROMPT_HASH_INDEX_DDL)
        self._conn.commit()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheMode",
    "CacheRequest",
    "CachedResponse",
    "CacheSchemaVersionMismatchError",
    "PromptCache",
]
