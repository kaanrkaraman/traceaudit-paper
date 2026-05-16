"""Six-table Parquet I/O for traces.

Layout per `docs/storage_layout.md` (updated for schema v0.1.1 to include
`tool_calls_json` on generation_events and `token_count` on chunks)::

    data/
      traces/system_id=<sys>/dataset_id=<ds>/
        traces.parquet
        steps.parquet
        generation_events.parquet
        retrieval_events.parquet
        rerank_events.parquet
        chunk_appearances.parquet
      chunks/
        chunks.parquet           (global, deduplicated by content_hash)

Atomic write contract
---------------------

Each of the seven tables is written via `<file>.parquet.partial`, then
`os.replace`'d into place. POSIX `rename` is atomic on the same filesystem,
so on a crash between writing the partial and the rename the committed file
is untouched. Readers never see the `.partial` because read paths never
glob for it; they open `<file>.parquet` directly.

Because Parquet has no safe append-mode (concurrent writers corrupt the
file; even a single writer reopening for append can leave the footer
inconsistent), each write performs a *read-combine-rewrite*: the existing
table is read into memory, the new rows are appended, and the full table
is rewritten to the partial. At Phase 0 scale (HotpotQA-100, a few
thousand rows per table) this is fast and simple. We will revisit when
the audit scales beyond ~10 000 traces per partition.

Read-side validation
--------------------

`read_trace` reconstructs the `Trace` object through Pydantic's validators,
so every invariant in `schema.py` fires after the read — including the
`content_hash == sha256(text)` check on chunks, the `prompt_hash ==
sha256(rendered_prompt)` check on generation events, and the
`total_cost_usd == sum(event.cost_usd)` reconciliation on the trace root.
A tampered Parquet file is rejected on read with a clear
`pydantic.ValidationError`.

Cache misses
------------

This module does not interact with the prompt cache directly; cache wiring
lives in `replayer.py` and the (forthcoming) `cache/` subpackage. Per D23,
strict-replay cache misses raise `ReplayCacheMissError`
(`traceaudit.trace.exceptions`). The exception is defined adjacent to this
module so the contract is settled before the replayer or cache lands.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from traceaudit.trace.schema import (
    AgentDecision,
    Chunk,
    ChunkAppearance,
    ChunkAppearanceRole,
    DecisionType,
    GenerationEvent,
    ModelProvider,
    RerankEvent,
    RetrievalEvent,
    Step,
    StepIntent,
    ToolCall,
    Trace,
    canonical_json,
)

# -----------------------------------------------------------------------------
# PyArrow schemas (single source of truth for column types).
# -----------------------------------------------------------------------------

TS_UTC = pa.timestamp("us", tz="UTC")

TRACES_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("system_id", pa.string(), nullable=False),
        pa.field("system_version", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("original_query", pa.string(), nullable=False),
        pa.field("gold_answers", pa.list_(pa.string()), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("config_hash", pa.string(), nullable=False),
        pa.field("config_json", pa.string(), nullable=False),
        pa.field("final_answer", pa.string(), nullable=False),
        pa.field("final_answer_tokens", pa.int32(), nullable=False),
        pa.field("started_at", TS_UTC, nullable=False),
        pa.field("ended_at", TS_UTC, nullable=False),
        pa.field("total_cost_usd", pa.float64(), nullable=False),
        pa.field("inputs_hash", pa.string(), nullable=False),
        pa.field("outputs_hash", pa.string(), nullable=False),
        pa.field("git_commit_sha", pa.string(), nullable=False),
        pa.field("package_versions_json", pa.string(), nullable=False),
    ]
)

STEPS_SCHEMA = pa.schema(
    [
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("parent_step_id", pa.string(), nullable=True),
        pa.field("step_index", pa.int32(), nullable=False),
        pa.field("intent", pa.string(), nullable=False),
        pa.field("query", pa.string(), nullable=False),
        pa.field("intermediate_answer", pa.string(), nullable=True),
        pa.field("decision_type", pa.string(), nullable=False),
        pa.field("decision_rationale", pa.string(), nullable=True),
        pa.field("decision_next_query", pa.string(), nullable=True),
        pa.field("decision_confidence", pa.float64(), nullable=True),
        pa.field("decision_raw_signal_json", pa.string(), nullable=False),
        pa.field("started_at", TS_UTC, nullable=False),
        pa.field("ended_at", TS_UTC, nullable=False),
    ]
)

GENERATION_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("event_index_in_step", pa.int16(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_provider", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=True),
        pa.field("prompt_template_id", pa.string(), nullable=False),
        pa.field("prompt_hash", pa.string(), nullable=False),
        pa.field("rendered_prompt", pa.string(), nullable=False),
        pa.field("temperature", pa.float64(), nullable=False),
        pa.field("top_p", pa.float64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=True),
        pa.field("max_tokens", pa.int32(), nullable=False),
        pa.field("response_text", pa.string(), nullable=False),
        pa.field("prompt_tokens", pa.int32(), nullable=False),
        pa.field("completion_tokens", pa.int32(), nullable=False),
        pa.field("cached", pa.bool_(), nullable=False),
        pa.field("finish_reason", pa.string(), nullable=False),
        pa.field("logprobs_json", pa.string(), nullable=True),
        pa.field("tool_calls_json", pa.string(), nullable=True),
        pa.field("cost_usd", pa.float64(), nullable=False),
        pa.field("latency_ms", pa.float64(), nullable=False),
        pa.field("started_at", TS_UTC, nullable=False),
    ]
)

RETRIEVAL_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("retriever_id", pa.string(), nullable=False),
        pa.field("retriever_version", pa.string(), nullable=True),
        pa.field("retriever_config_json", pa.string(), nullable=False),
        pa.field("pre_rewrite_query", pa.string(), nullable=True),
        pa.field("query", pa.string(), nullable=False),
        pa.field("top_k", pa.int32(), nullable=False),
        pa.field("latency_ms", pa.float64(), nullable=False),
        pa.field("started_at", TS_UTC, nullable=False),
    ]
)

RERANK_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("reranker_id", pa.string(), nullable=False),
        pa.field("reranker_version", pa.string(), nullable=True),
        pa.field("latency_ms", pa.float64(), nullable=False),
        pa.field("started_at", TS_UTC, nullable=False),
    ]
)

CHUNK_APPEARANCES_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("step_id", pa.string(), nullable=False),
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
    ]
)

CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("content_hash", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_offset_start", pa.int64(), nullable=True),
        pa.field("source_offset_end", pa.int64(), nullable=True),
        pa.field("metadata_json", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=True),
        pa.field("first_seen_at", TS_UTC, nullable=False),
    ]
)


# -----------------------------------------------------------------------------
# TraceStore
# -----------------------------------------------------------------------------


class TraceStore:
    """Read and write `Trace` objects against the six-table Parquet layout.

    A `TraceStore` is rooted at a directory; all paths are computed relative
    to that root. The store is stateless apart from the root path — multiple
    instances pointing at the same root are safe to use *serially*, but
    Parquet's append semantics forbid concurrent writers to the same
    partition (see module docstring).
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.traces_root = self.root / "traces"
        self.chunks_root = self.root / "chunks"

    # ---- Public API ----------------------------------------------------------

    def write_trace(self, trace: Trace, chunks: dict[str, Chunk]) -> None:
        """Write `trace` and its referenced chunks atomically.

        `chunks` must map exactly the set of `content_hash` values that appear
        in `trace.steps[*].chunk_appearances`. Missing or extra entries raise
        `ValueError` before any file is touched.
        """
        referenced = {
            app.content_hash for step in trace.steps for app in step.chunk_appearances
        }
        provided = set(chunks.keys())
        missing = referenced - provided
        if missing:
            raise ValueError(
                f"chunks dict is missing {len(missing)} referenced content_hash(es); "
                f"first few: {sorted(missing)[:3]}"
            )
        extra = provided - referenced
        if extra:
            raise ValueError(
                f"chunks dict contains {len(extra)} unreferenced content_hash(es); "
                f"first few: {sorted(extra)[:3]}"
            )

        partition = self._partition_dir(trace.system_id, trace.dataset_id)
        partition.mkdir(parents=True, exist_ok=True)

        # Build all rows in memory first; any serialization failure aborts the
        # write before any file is touched, leaving the partition unchanged.
        trace_row = _serialize_trace(trace)
        step_rows = [_serialize_step(s, trace.trace_id) for s in trace.steps]
        gen_rows: list[dict[str, Any]] = []
        ret_rows: list[dict[str, Any]] = []
        rer_rows: list[dict[str, Any]] = []
        app_rows: list[dict[str, Any]] = []
        for step in trace.steps:
            for ev in step.generation_events:
                gen_rows.append(
                    _serialize_generation_event(ev, trace.trace_id, step.step_id)
                )
            if step.retrieval_event is not None:
                ret_rows.append(
                    _serialize_retrieval_event(
                        step.retrieval_event, trace.trace_id, step.step_id
                    )
                )
            if step.rerank_event is not None:
                rer_rows.append(
                    _serialize_rerank_event(
                        step.rerank_event, trace.trace_id, step.step_id
                    )
                )
            for app in step.chunk_appearances:
                app_rows.append(
                    _serialize_chunk_appearance(app, trace.trace_id, step.step_id)
                )

        # Append rows table-by-table. Each append is atomic w.r.t. its own
        # file; the cross-table sequence is not transactional, but every
        # individual file is either old-or-new, never partial.
        _append_and_rename(partition / "traces.parquet", TRACES_SCHEMA, [trace_row])
        _append_and_rename(partition / "steps.parquet", STEPS_SCHEMA, step_rows)
        if gen_rows:
            _append_and_rename(
                partition / "generation_events.parquet",
                GENERATION_EVENTS_SCHEMA,
                gen_rows,
            )
        if ret_rows:
            _append_and_rename(
                partition / "retrieval_events.parquet",
                RETRIEVAL_EVENTS_SCHEMA,
                ret_rows,
            )
        if rer_rows:
            _append_and_rename(
                partition / "rerank_events.parquet", RERANK_EVENTS_SCHEMA, rer_rows
            )
        if app_rows:
            _append_and_rename(
                partition / "chunk_appearances.parquet",
                CHUNK_APPEARANCES_SCHEMA,
                app_rows,
            )

        # Global chunks table: dedup by content_hash. Only new chunks are
        # appended; first_seen_at on existing rows is preserved.
        self.chunks_root.mkdir(parents=True, exist_ok=True)
        chunks_file = self.chunks_root / "chunks.parquet"
        existing_hashes = self._read_chunk_hashes(chunks_file)
        new_chunk_rows = [
            _serialize_chunk(chunks[h]) for h in referenced if h not in existing_hashes
        ]
        if new_chunk_rows:
            _append_and_rename(chunks_file, CHUNKS_SCHEMA, new_chunk_rows)

    def read_trace(self, trace_id: str, *, system_id: str, dataset_id: str) -> Trace:
        """Read and re-validate the trace with `trace_id` from the given partition.

        Raises `FileNotFoundError` if the partition has never been written.
        Raises `KeyError` if the partition exists but does not contain
        `trace_id`. Raises `pydantic.ValidationError` if the on-disk data
        violates any schema invariant — including a tampered cost reconciliation.
        """
        partition = self._partition_dir(system_id, dataset_id)
        traces_file = partition / "traces.parquet"
        if not traces_file.exists():
            raise FileNotFoundError(
                f"no partition for (system_id={system_id!r}, dataset_id={dataset_id!r}); "
                f"expected {traces_file}"
            )

        trace_rows = [
            r
            for r in _read_parquet_file(traces_file).to_pylist()
            if r["trace_id"] == trace_id
        ]
        if not trace_rows:
            raise KeyError(
                f"trace_id={trace_id!r} not found in partition "
                f"(system_id={system_id!r}, dataset_id={dataset_id!r})"
            )
        trace_row = trace_rows[0]

        step_rows = sorted(
            (
                r
                for r in _read_parquet_file(partition / "steps.parquet").to_pylist()
                if r["trace_id"] == trace_id
            ),
            key=lambda r: r["step_index"],
        )

        gen_rows = self._read_rows_for_trace(
            partition / "generation_events.parquet", trace_id
        )
        ret_rows = self._read_rows_for_trace(
            partition / "retrieval_events.parquet", trace_id
        )
        rer_rows = self._read_rows_for_trace(
            partition / "rerank_events.parquet", trace_id
        )
        app_rows = self._read_rows_for_trace(
            partition / "chunk_appearances.parquet", trace_id
        )

        steps: list[Step] = []
        for sr in step_rows:
            sid = sr["step_id"]
            step_gens = sorted(
                (
                    _deserialize_generation_event(r)
                    for r in gen_rows
                    if r["step_id"] == sid
                ),
                key=lambda ev: ev.event_index_in_step,
            )
            step_ret = next(
                (
                    _deserialize_retrieval_event(r)
                    for r in ret_rows
                    if r["step_id"] == sid
                ),
                None,
            )
            step_rer = next(
                (_deserialize_rerank_event(r) for r in rer_rows if r["step_id"] == sid),
                None,
            )
            step_apps = tuple(
                sorted(
                    (
                        _deserialize_chunk_appearance(r)
                        for r in app_rows
                        if r["step_id"] == sid
                    ),
                    key=lambda a: (a.role.value, a.position),
                )
            )
            steps.append(
                _deserialize_step(sr, step_gens, step_ret, step_rer, step_apps)
            )

        return _deserialize_trace(trace_row, steps)

    def read_chunks(self, content_hashes: set[str]) -> dict[str, Chunk]:
        """Return the subset of `content_hashes` present in the global chunks table."""
        chunks_file = self.chunks_root / "chunks.parquet"
        if not chunks_file.exists():
            return {}
        rows = _read_parquet_file(chunks_file).to_pylist()
        return {
            r["content_hash"]: _deserialize_chunk(r)
            for r in rows
            if r["content_hash"] in content_hashes
        }

    # ---- Internals -----------------------------------------------------------

    def _partition_dir(self, system_id: str, dataset_id: str) -> Path:
        return self.traces_root / f"system_id={system_id}" / f"dataset_id={dataset_id}"

    @staticmethod
    def _read_chunk_hashes(chunks_file: Path) -> set[str]:
        if not chunks_file.exists():
            return set()
        return set(
            _read_parquet_file(chunks_file, columns=["content_hash"])
            .column("content_hash")
            .to_pylist()
        )

    @staticmethod
    def _read_rows_for_trace(path: Path, trace_id: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            r for r in _read_parquet_file(path).to_pylist() if r["trace_id"] == trace_id
        ]


# -----------------------------------------------------------------------------
# Atomic append helper.
# -----------------------------------------------------------------------------


def _read_parquet_file(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read a single Parquet file as one table, bypassing partition discovery.

    `pq.read_table(path)` would otherwise see `system_id=X/dataset_id=Y/` in
    the parent directory path, infer those as Hive-style partition columns,
    and refuse to merge them with the explicit `system_id` / `dataset_id`
    string columns we store inside every file. Reading via `ParquetFile.read`
    skips dataset construction entirely and gives us back exactly the file's
    own schema, which is what every read in this module wants.
    """
    return pq.ParquetFile(str(path)).read(columns=columns)


def _append_and_rename(
    path: Path, schema: pa.Schema, new_rows: list[dict[str, Any]]
) -> None:
    """Read existing parquet, combine with `new_rows`, write `.partial`, rename.

    The rename uses `os.replace`, which is atomic on the same filesystem on
    POSIX and on Windows (since Python 3.3). A crash between writing the
    partial and the rename leaves the committed file untouched; the orphan
    `.partial` is overwritten on the next write to the same path.
    """
    if not new_rows:
        return
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        existing_rows = _read_parquet_file(path).to_pylist()
    combined = existing_rows + new_rows
    partial = path.with_name(path.name + ".partial")
    pq.write_table(pa.Table.from_pylist(combined, schema=schema), partial)
    os.replace(partial, path)


# -----------------------------------------------------------------------------
# Serializers — Pydantic model -> flat row dict (one per table row).
# -----------------------------------------------------------------------------


def _serialize_trace(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "schema_version": trace.schema_version,
        "system_id": trace.system_id,
        "system_version": trace.system_version,
        "dataset_id": trace.dataset_id,
        "query_id": trace.query_id,
        "original_query": trace.original_query,
        "gold_answers": list(trace.gold_answers),
        "seed": trace.seed,
        "config_hash": trace.config_hash,
        "config_json": trace.config_json,
        "final_answer": trace.final_answer,
        "final_answer_tokens": trace.final_answer_tokens,
        "started_at": trace.started_at,
        "ended_at": trace.ended_at,
        "total_cost_usd": trace.total_cost_usd,
        "inputs_hash": trace.inputs_hash,
        "outputs_hash": trace.outputs_hash,
        "git_commit_sha": trace.git_commit_sha,
        "package_versions_json": trace.package_versions_json,
    }


def _serialize_step(step: Step, trace_id: str) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "trace_id": trace_id,
        "parent_step_id": step.parent_step_id,
        "step_index": step.step_index,
        "intent": step.intent.value,
        "query": step.query,
        "intermediate_answer": step.intermediate_answer,
        "decision_type": step.decision.decision_type.value,
        "decision_rationale": step.decision.rationale,
        "decision_next_query": step.decision.next_query,
        "decision_confidence": step.decision.confidence,
        "decision_raw_signal_json": canonical_json(step.decision.raw_signal),
        "started_at": step.started_at,
        "ended_at": step.ended_at,
    }


def _serialize_generation_event(
    ev: GenerationEvent, trace_id: str, step_id: str
) -> dict[str, Any]:
    tool_calls_json: str | None = None
    if ev.tool_calls is not None:
        tool_calls_json = canonical_json(
            [
                {
                    "name": tc.name,
                    "arguments_json": tc.arguments_json,
                    "result_json": tc.result_json,
                    "call_id": tc.call_id,
                }
                for tc in ev.tool_calls
            ]
        )
    return {
        "event_id": ev.event_id,
        "trace_id": trace_id,
        "step_id": step_id,
        "event_index_in_step": ev.event_index_in_step,
        "model_id": ev.model_id,
        "model_provider": ev.model_provider.value,
        "model_version": ev.model_version,
        "prompt_template_id": ev.prompt_template_id,
        "prompt_hash": ev.prompt_hash,
        "rendered_prompt": ev.rendered_prompt,
        "temperature": ev.temperature,
        "top_p": ev.top_p,
        "seed": ev.seed,
        "max_tokens": ev.max_tokens,
        "response_text": ev.response_text,
        "prompt_tokens": ev.prompt_tokens,
        "completion_tokens": ev.completion_tokens,
        "cached": ev.cached,
        "finish_reason": ev.finish_reason,
        "logprobs_json": ev.logprobs_json,
        "tool_calls_json": tool_calls_json,
        "cost_usd": ev.cost_usd,
        "latency_ms": ev.latency_ms,
        "started_at": ev.started_at,
    }


def _serialize_retrieval_event(
    ev: RetrievalEvent, trace_id: str, step_id: str
) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "trace_id": trace_id,
        "step_id": step_id,
        "retriever_id": ev.retriever_id,
        "retriever_version": ev.retriever_version,
        "retriever_config_json": ev.retriever_config_json,
        "pre_rewrite_query": ev.pre_rewrite_query,
        "query": ev.query,
        "top_k": ev.top_k,
        "latency_ms": ev.latency_ms,
        "started_at": ev.started_at,
    }


def _serialize_rerank_event(
    ev: RerankEvent, trace_id: str, step_id: str
) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "trace_id": trace_id,
        "step_id": step_id,
        "reranker_id": ev.reranker_id,
        "reranker_version": ev.reranker_version,
        "latency_ms": ev.latency_ms,
        "started_at": ev.started_at,
    }


def _serialize_chunk_appearance(
    app: ChunkAppearance, trace_id: str, step_id: str
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "step_id": step_id,
        "content_hash": app.content_hash,
        "role": app.role.value,
        "position": app.position,
        "score": app.score,
    }


def _serialize_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "content_hash": chunk.content_hash,
        "text": chunk.text,
        "source_id": chunk.source_id,
        "source_offset_start": chunk.source_offset_start,
        "source_offset_end": chunk.source_offset_end,
        "metadata_json": canonical_json(chunk.metadata),
        "token_count": chunk.token_count,
        "first_seen_at": datetime.now(tz=timezone.utc),
    }


# -----------------------------------------------------------------------------
# Deserializers — flat row dict -> Pydantic model. Validators fire here.
# -----------------------------------------------------------------------------


def _deserialize_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        content_hash=row["content_hash"],
        text=row["text"],
        source_id=row["source_id"],
        source_offset_start=row["source_offset_start"],
        source_offset_end=row["source_offset_end"],
        metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        token_count=row["token_count"],
    )


def _deserialize_chunk_appearance(row: dict[str, Any]) -> ChunkAppearance:
    return ChunkAppearance(
        content_hash=row["content_hash"],
        role=ChunkAppearanceRole(row["role"]),
        position=row["position"],
        score=row["score"],
    )


def _deserialize_generation_event(row: dict[str, Any]) -> GenerationEvent:
    tool_calls: tuple[ToolCall, ...] | None = None
    if row["tool_calls_json"]:
        calls = json.loads(row["tool_calls_json"])
        tool_calls = tuple(
            ToolCall(
                name=c["name"],
                arguments_json=c["arguments_json"],
                result_json=c["result_json"],
                call_id=c["call_id"],
            )
            for c in calls
        )
    return GenerationEvent(
        event_id=row["event_id"],
        event_index_in_step=row["event_index_in_step"],
        model_id=row["model_id"],
        model_provider=ModelProvider(row["model_provider"]),
        model_version=row["model_version"],
        prompt_template_id=row["prompt_template_id"],
        prompt_hash=row["prompt_hash"],
        rendered_prompt=row["rendered_prompt"],
        temperature=row["temperature"],
        top_p=row["top_p"],
        seed=row["seed"],
        max_tokens=row["max_tokens"],
        response_text=row["response_text"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        cached=row["cached"],
        finish_reason=row["finish_reason"],
        logprobs_json=row["logprobs_json"],
        tool_calls=tool_calls,
        cost_usd=row["cost_usd"],
        latency_ms=row["latency_ms"],
        started_at=row["started_at"],
    )


def _deserialize_retrieval_event(row: dict[str, Any]) -> RetrievalEvent:
    return RetrievalEvent(
        event_id=row["event_id"],
        retriever_id=row["retriever_id"],
        retriever_version=row["retriever_version"],
        retriever_config_json=row["retriever_config_json"],
        pre_rewrite_query=row["pre_rewrite_query"],
        query=row["query"],
        top_k=row["top_k"],
        latency_ms=row["latency_ms"],
        started_at=row["started_at"],
    )


def _deserialize_rerank_event(row: dict[str, Any]) -> RerankEvent:
    return RerankEvent(
        event_id=row["event_id"],
        reranker_id=row["reranker_id"],
        reranker_version=row["reranker_version"],
        latency_ms=row["latency_ms"],
        started_at=row["started_at"],
    )


def _deserialize_step(
    row: dict[str, Any],
    gen_events: list[GenerationEvent],
    retrieval: RetrievalEvent | None,
    rerank: RerankEvent | None,
    appearances: tuple[ChunkAppearance, ...],
) -> Step:
    return Step(
        step_id=row["step_id"],
        parent_step_id=row["parent_step_id"],
        step_index=row["step_index"],
        intent=StepIntent(row["intent"]),
        query=row["query"],
        generation_events=tuple(gen_events),
        retrieval_event=retrieval,
        rerank_event=rerank,
        chunk_appearances=appearances,
        intermediate_answer=row["intermediate_answer"],
        decision=AgentDecision(
            decision_type=DecisionType(row["decision_type"]),
            rationale=row["decision_rationale"],
            next_query=row["decision_next_query"],
            confidence=row["decision_confidence"],
            raw_signal=json.loads(row["decision_raw_signal_json"])
            if row["decision_raw_signal_json"]
            else {},
        ),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _deserialize_trace(row: dict[str, Any], steps: list[Step]) -> Trace:
    return Trace(
        trace_id=row["trace_id"],
        schema_version=row["schema_version"],
        system_id=row["system_id"],
        system_version=row["system_version"],
        dataset_id=row["dataset_id"],
        query_id=row["query_id"],
        original_query=row["original_query"],
        gold_answers=tuple(row["gold_answers"]),
        seed=row["seed"],
        config_hash=row["config_hash"],
        config_json=row["config_json"],
        git_commit_sha=row["git_commit_sha"],
        package_versions_json=row["package_versions_json"],
        steps=tuple(steps),
        final_answer=row["final_answer"],
        final_answer_tokens=row["final_answer_tokens"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        total_cost_usd=row["total_cost_usd"],
        inputs_hash=row["inputs_hash"],
        outputs_hash=row["outputs_hash"],
    )


__all__ = [
    "TraceStore",
    "TRACES_SCHEMA",
    "STEPS_SCHEMA",
    "GENERATION_EVENTS_SCHEMA",
    "RETRIEVAL_EVENTS_SCHEMA",
    "RERANK_EVENTS_SCHEMA",
    "CHUNK_APPEARANCES_SCHEMA",
    "CHUNKS_SCHEMA",
]
