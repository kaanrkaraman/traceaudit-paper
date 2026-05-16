# Trace Storage Layout — Proposal (awaiting approval)

**Status:** Proposed 2026-05-14, awaiting Phase 0 milestone approval.
**Resolves:** `open_questions.md` Q6.
**Companion:** `src/traceaudit/trace/schema.py`.

## TL;DR

Normalize the trace model into six Parquet tables joined on stable IDs, partitioned by `system_id` and `dataset_id`, with chunks deduplicated by `content_hash` across all traces. PyArrow datasets handle the joins; no row-level database. A self-contained nested-Parquet *export* format is deferred to the paper-supplementary stage.

## Why normalized over one-file-per-trace

| Concern | Nested (one file / trace) | Normalized (this proposal) |
|---|---|---|
| Chunk dedup across traces | None — chunk text duplicated everywhere it was retrieved | Single row per unique `content_hash` |
| Intervention-sweep cost | Must walk every trace file to compile a chunk's appearances | One query against `chunk_appearances` |
| Set statistics (URR distributions, per-chunk effect sizes) | Hard | Native columnar aggregation |
| Portability of a single trace | Trivial (one file) | Requires the join — solved by a separate export step |
| Disk footprint | Larger (duplication) | Smaller |
| Schema evolution | Per-file versioning | Per-table versioning, easier additive changes |

The normalized layout pays a small ergonomic tax (a trace lives across six tables) for substantial wins on the operations we'll perform thousands of times.

## Directory layout

```
data/
  traces/
    system_id=crag/
      dataset_id=hotpotqa-dev/
        traces.parquet
        steps.parquet
        generation_events.parquet
        retrieval_events.parquet
        rerank_events.parquet
        chunk_appearances.parquet
    system_id=ircot/
      ...
  chunks/
    chunks.parquet   # global, deduplicated across all systems and datasets
  interventions/
    mode=A/
      operator=remove/
        system_id=crag/dataset_id=hotpotqa-dev/
          results.parquet      # one row per (trace_id, intervened_step_id, intervened_chunk_hash)
      operator=paraphrase/...
      operator=distract/...
    mode=B/...
    mode=C/...
  calibration/
    synthetic_v1.parquet       # Phase 1 calibration set
    judge_labels_v1.parquet    # 100-pair human labels for GPT-4o judge calibration
```

Partition columns (`system_id`, `dataset_id`) are encoded in Hive-style directories so PyArrow `dataset.dataset(...)` can prune them automatically.

## Per-table schema sketches

(Field names match `src/traceaudit/trace/schema.py`. Types are PyArrow logical types.)

### `traces.parquet`
- `trace_id` `string` (primary, deterministic hash — see `open_questions.md` Q7)
- `schema_version` `string`
- `system_id`, `system_version`, `dataset_id`, `query_id` `string`
- `original_query` `string`
- `gold_answers` `list<string>`
- `seed` `int64`
- `config_hash` `string`
- `config_json` `string` (the full Hydra config as JSON; opaque to queries but available for replay)
- `final_answer` `string`
- `final_answer_tokens` `int32`
- `started_at`, `ended_at` `timestamp[us, tz=UTC]`
- `total_cost_usd` `float64`
- `inputs_hash`, `outputs_hash` `string`
- `git_commit_sha` `string`
- `package_versions_json` `string`

### `steps.parquet`
- `step_id` `string` (PK)
- `trace_id` `string` (FK)
- `parent_step_id` `string` (nullable, for trees)
- `step_index` `int32`
- `intent` `string` (enum-valued)
- `query` `string`
- `intermediate_answer` `string` (nullable)
- `decision_type` `string`
- `decision_rationale` `string` (nullable)
- `decision_next_query` `string` (nullable)
- `decision_confidence` `float64` (nullable)
- `decision_raw_signal_json` `string` (system-specific signals; opaque)
- `started_at`, `ended_at` `timestamp[us, tz=UTC]`

### `generation_events.parquet`
- `event_id` `string` (PK)
- `trace_id`, `step_id` `string` (FKs)
- `event_index_in_step` `int16`
- `model_id`, `model_provider`, `model_version` `string`
- `prompt_template_id` `string`
- `prompt_hash` `string` (SHA256 of `rendered_prompt`)
- `rendered_prompt` `string` (verbatim; required for replay)
- `temperature`, `top_p` `float64`
- `seed` `int64` (nullable)
- `max_tokens` `int32`
- `response_text` `string`
- `prompt_tokens`, `completion_tokens` `int32`
- `cached` `bool` (true if served from SQLite prompt cache)
- `finish_reason` `string`
- `logprobs_json` `string` (nullable; serialized if captured)
- `tool_calls_json` `string` (nullable; canonical-JSON of the tuple of `ToolCall` records issued in this generation, added with schema v0.1.1)
- `cost_usd` `float64`
- `latency_ms` `float64`
- `started_at` `timestamp[us, tz=UTC]`

### `retrieval_events.parquet`
- `event_id` `string` (PK)
- `trace_id`, `step_id` `string` (FKs)
- `retriever_id`, `retriever_version` `string`
- `retriever_config_json` `string`
- `query` `string`
- `rewritten_query` `string` (nullable)
- `top_k` `int32`
- `latency_ms` `float64`
- `started_at` `timestamp[us, tz=UTC]`

### `rerank_events.parquet`
- `event_id` `string` (PK)
- `trace_id`, `step_id` `string` (FKs)
- `reranker_id`, `reranker_version` `string`
- `latency_ms` `float64`

### `chunks.parquet` (global, dedup'd)
- `content_hash` `string` (PK, SHA256 of raw UTF-8 bytes)
- `text` `string`
- `source_id` `string` (document ID in source corpus)
- `source_offset_start`, `source_offset_end` `int64` (nullable)
- `metadata_json` `string`
- `token_count` `int32` (nullable; tokens in `text` per the generator's tokenizer at recording time; added with schema v0.1.1)
- `first_seen_at` `timestamp[us, tz=UTC]`

### `chunk_appearances.parquet` (the many-to-many)
- `trace_id`, `step_id`, `content_hash` `string`
- `role` `string` ∈ {`retrieved`, `reranked`, `in_context`}
- `position` `int32` (rank in the list for this role)
- `score` `float64` (nullable; retrieval or reranker score)
- One row per (trace, step, chunk, role). A chunk can have up to three rows per step (one per role).

## Why a single global `chunks.parquet`

Across HotpotQA-500, MuSiQue-500, 2WikiMultiHopQA-500, Bamboogle ~125, and any heterogeneous datasets, the same Wikipedia paragraph will be retrieved by many traces and many systems. A single deduplicated table:

- Cuts disk footprint substantially (rough back-of-envelope: a typical CRAG run retrieves ~10 chunks per query × 500 queries × ~4 systems = 20 000 retrieval slots that collapse to a few thousand unique chunks).
- Makes "find every trace where chunk X appeared in context" a single PyArrow filter, which is the core query for cross-trace intervention bookkeeping.
- Means the intervention operators (`remove`, `paraphrase`, `distract`) only need to be applied once per unique chunk per operator config, not per appearance — the SQLite prompt cache stores the operator's output keyed by `content_hash` and operator config.

## Writes and concurrency

- Single-writer per partition (one process appends to one `(system_id, dataset_id)` partition at a time). PyArrow append-mode parquet does not support concurrent writers safely.
- Recorder buffers a trace in memory, then writes all of its rows across the six tables atomically: write each table's new rows to a `*.parquet.partial` file in the partition, then rename. If any file write fails, delete the partials.
- For Phase 2/3 multi-system parallelism, partition isolation gives us system-level write concurrency without locking.

## Versioning

- `SCHEMA_VERSION` constant in `src/traceaudit/trace/schema.py`. Bumped on any breaking change.
- The replayer refuses traces with a different *major* schema version. Minor bumps (additive fields with defaults) are forward-compatible.
- A `migrations/` folder will hold one Python script per breaking version transition. Not needed in Phase 0.

## Open follow-ups for this proposal

1. Logprob capture: tokenizing-free closed APIs make this lossy. Recommendation: capture only when the provider returns them, store as JSON in `logprobs_json`, never depend on them for determinism.
2. Tool-use payloads (function-calling agents): treat tool calls as additional `generation_events` rows with `prompt_template_id = "tool_<name>"`. Confirm at first agent integration.
3. Embedding caches: vectors live outside this layout, in Qdrant. We store the embedding model identifier on `chunks` but not the vector itself.
