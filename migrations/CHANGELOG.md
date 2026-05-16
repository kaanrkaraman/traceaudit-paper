# Schema CHANGELOG

Logs every change to the trace schema in `src/traceaudit/trace/schema.py`. Bump the version
in the same commit as the change.

Semantic versioning:
- **Major** bump — breaking change that requires migrating existing on-disk
  traces. The replayer refuses traces with a different major version.
- **Minor** bump — additive, forward-compatible (new optional field with default).
- **Patch** bump — documentation- or validator-only change.

---

## 0.1.2 — 2026-05-14

Approved as D25 in the compact-handoff briefing. Honest framing: technically
a breaking change to the determinism witness — `outputs_hash` payload
structure changes shape, and any 0.1.1 trace with non-`None` `tool_calls`
would produce a different `outputs_hash` under 0.1.2 — but no real traces
exist yet (only the synthetic test fixture), so a patch bump is honest about
the schema state. Same reasoning as the 0.1.0 → 0.1.1 rename.

- **`compute_outputs_hash`** — signature changed from
  `(final_answer, generation_responses: list[str])` to
  `(final_answer, generation_events: Iterable[GenerationEvent])`. Each event
  now contributes `{"response_text": ev.response_text, "tool_calls": [{"name",
  "arguments_json", "call_id"}, ...]}`. `tool_calls=None` collapses to `[]`
  in the payload. `result_json` is intentionally excluded — it may not be
  known at record time when a tool call is in flight, and re-serialization
  variance would make the witness flap across replays. Outer wrapping is
  still `canonical_json({"final_answer": ..., "generation_outputs": [...]})`.
- **`SCHEMA_VERSION`** — `"0.1.1"` → `"0.1.2"`.
- **`Trace.schema_version`** Literal — `"0.1.1"` → `"0.1.2"`.
- **No field additions, renames, or removals. No new models. No validator changes.**

**Identity impact.** `outputs_hash` changes for every trace whose generation
events carry `tool_calls`. `trace_id`, `inputs_hash`, and `config_hash` are
unaffected — none depend on `outputs_hash` or on `tool_calls`. Pre-0.1.2
`outputs_hash` values cannot be compared directly against 0.1.2 and would
need recomputation against the new semantics; today this is moot because
the synthetic test fixture is the only trace in existence.

## 0.1.1 — 2026-05-14

Approved at the Phase 0 schema review. Bumped while no traces existed; the
rename below is technically a breaking change but is honestly recorded as a
patch bump because no on-disk artifacts are invalidated.

- **`RetrievalEvent`** — renamed `rewritten_query: str | None` to
  `pre_rewrite_query: str | None` so the name matches the semantics ("the
  original query, before any retriever-side rewriting"). The actually-sent
  query remains in `query`. New validator: if `pre_rewrite_query` is set, it
  must differ from `query` (set to `None` if no rewrite occurred). Resolves
  the reviewer-flagged naming inversion.
- **`GenerationEvent`** — added `tool_calls: tuple[ToolCall, ...] | None = None`
  for first-class structured tool / function calls. Introduces a new
  `ToolCall` model with fields `name`, `arguments_json` (canonical-JSON),
  `result_json` (optional, canonical-JSON when result known at capture time),
  and `call_id` (provider-issued when available, otherwise a deterministic
  hash via `compute_tool_call_id(name, arguments_json)`). Empty tuple is
  rejected — use `None` to denote "no tool calls."
- **`Chunk`** — added `token_count: int | None = None`. Tokenizer-specific;
  recorder populates against the generator model's tokenizer at recording
  time. Enables audit metrics such as "tokens of retrieved evidence the agent
  actually consumed."
- **`__all__`** — added `ToolCall` and `compute_tool_call_id`.

## 0.1.0 — 2026-05-14

Initial proposal. See `docs/storage_layout.md` for the storage layout the
schema serializes into. Superseded by 0.1.1 before any trace was captured.
