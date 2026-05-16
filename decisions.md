# Approved Methodological Decisions

Append-only record of methodological decisions the humans have **explicitly approved**, with reasoning. Items in `open_questions.md` move here only after sign-off.

Format:
- **D## — Title** (date approved, approver)
  - **Decision:** what was decided
  - **Reasoning:** why
  - **Resolves:** which `open_questions.md` entries this closes
  - **Revisit if:** trigger conditions for re-opening

---

## Project constants (fixed by the brief, not subject to re-decision)

- **D00 — Working title:** TraceAudit. Co-authors: Recep Kaan Karaman, Meftun Akarsu.
- **D01 — Target venues:** AAAI 2027 primary (early August 2026 deadline anticipated). ECIR 2027 fallback (October 2026 anticipated). NeurIPS / EMNLP workshop late August 2026 as hedge.
- **D02 — Hard cost ceiling:** Total commercial-LLM API spend < 2000 USD. GPT-4o calls capped at 500 across the entire project, reserved for the calibrated judge on headline results. GPT-4o-mini only when a closed model is strictly required. Replay-heavy work on Azure-hosted Llama 3.1 70B / 8B. Persistent SQLite prompt cache from day one. Halt and ask before pushing consumption above 1500 USD.
- **D03 — Three intervention modes:** A (fixed-policy ablation, regenerate only the final answer), B (counterfactual rollout, agent freely re-decides downstream), C (step-truncation). Mode A − Mode B gap is the headline quantity.
- **D04 — Three intervention operators:** remove, paraphrase (cross-family with generator), distract (cosine in [0.6, 0.85] vs BGE-large, factually wrong but retrievable).
- **D05 — URR definition:** Fraction of generation-useful chunks per trace, computed on Mode A. Reported in three flavors — F1-drop > 5pp, strict-EM flip, GPT-4o judge — with judge contingent on ≥ 85% accuracy on a 100-pair calibration set.
- **D06 — Validation gate (Phase 1 hard gate):** Synthetic calibration set ≥ 200 examples. URR must identify useful chunks with ≥ 90% agreement against ground truth, and negative-control paraphrases must give near-zero URR. If either fails, stop and rethink.
- **D07 — Audit priority order:** CRAG → IRCoT → Self-RAG → FLARE → (optional) A-RAG, MA-RAG. Reproduction parity within 2pp on at least one claimed dataset is a precondition for inclusion; failures are dropped and documented here.
- **D08 — Dataset priority order:** HotpotQA dev-500 (random_state 42), MuSiQue dev-500, 2WikiMultiHopQA dev-500, Bamboogle full (~125). Optional heterogeneous: FinanceBench or TAT-DQA at 300. Released query IDs accompany the paper. No NQ / TriviaQA / single-hop.
- **D09 — Tech stack:** Python 3.11; Hydra; Pydantic v2; Qdrant in Docker (self-hosted); BGE-large-en-v1.5 (sentence-transformers, local); BGE-reranker-v2-m3 (local); Azure OpenAI for GPT-4o-mini / GPT-4o; Azure-hosted Llama 3.1 70B / 8B; DiskCache + SQLite prompt cache; Parquet via PyArrow; W&B free academic; Docker compose; pre-commit (ruff + black); pytest with determinism tests.
- **D10 — Standing protocols:** No silent overwrites. Ask before destructive changes. Hard milestone stops per phase. Ask before every git commit. Never invent papers, benchmarks, or numbers — mark `UNVERIFIED` and add to `open_questions.md`.

## Phase 0 decisions

Approved 2026-05-14 at the schema review.

### Schema design (D11–D17)

- **D11 — Chunk identity:** `content_hash = sha256(text.encode("utf-8"))`, no normalization. A separate `normalized_hash` may be added later but never as a replacement.
  - **Reasoning:** "Exact bytes, exact identity. Normalization is irreversible and conflates distinct evidence."
  - **Resolves:** schema design point §2.
  - **Revisit if:** we find significant evidence drift from minor encoding differences (e.g. NFC vs NFKC Unicode forms) creating spurious distinct chunks at scale.

- **D12 — Tree-structured steps:** `Step.parent_step_id` is optional from day one; linear traces are a degenerate tree.
  - **Reasoning:** "Costs one nullable field now; saves a refactor when MA-RAG arrives in Phase 3."
  - **Resolves:** Q8.
  - **Revisit if:** MA-RAG is cut from Phase 3 scope and no other tree-shaped system arrives — at that point the field is dead weight but cheap to keep.

- **D13 — Multiple `GenerationEvent`s per `Step`:** an ordered tuple, with `event_index_in_step` contiguous from 0.
  - **Reasoning:** "Self-RAG (reflection then answer), CRAG (evaluator → optional rewriter → answer), FLARE (lookahead → optional answer) all need this."
  - **Resolves:** schema design point §4.
  - **Revisit if:** never — this is a fact of the audited systems' architectures.

- **D14 — `rendered_prompt` verbatim alongside `prompt_hash`:** the prompt cache is keyed on `prompt_hash`; the cache stores the response keyed by that hash.
  - **Reasoning:** "Required for deterministic replay and for the SQLite prompt cache to be the source of truth."
  - **Resolves:** schema design point §5.
  - **Revisit if:** disk footprint becomes a constraint (unlikely at our scale; few KB per call).

- **D15 — Deterministic `trace_id`:** `sha256(canonical_json({system_id, system_version, dataset_id, query_id, seed, config_hash}))`.
  - **Reasoning:** "Same inputs → same id, which lets the determinism tests assert byte-identical replay against the originally captured trace_id."
  - **Resolves:** Q7.
  - **Revisit if:** we ever need to record multiple traces with identical inputs (e.g. for sampling variance) — at that point we'd add a `run_index` to the hash inputs.

- **D16 — `Trace.outputs_hash` is the determinism witness:** `sha256(canonical_json({final_answer, ordered generation_responses}))`. Determinism tests assert it matches across replays.
  - **Reasoning:** "Phase 0 hypothesis H0.2 is that this matches across consecutive replays on ≥99% of HotpotQA-100 traces with a warm cache."
  - **Resolves:** schema design point §6.
  - **Revisit if:** any audited system has internal stochasticity that escapes seeding *and* legitimately should not be re-run — we'd document the carve-out per system.

- **D17 — All schema models are `frozen=True, extra="forbid"`:** traces are immutable; mutation requires `.model_copy(update=...)`; unknown fields raise on parse.
  - **Reasoning:** "Traces are immutable artifacts; any mutation must be an explicit copy. `extra='forbid'` catches typos and refuses to silently drop fields a future schema version added."
  - **Resolves:** schema design point §1.
  - **Revisit if:** never.

### Methodological decisions (D18–D21) — lifted from `open_questions.md`

- **D18 — URR hybrid (resolves Q1):** F1 drop > 5pp is the primary continuous signal. Twin strict-EM URR and judge-URR in the headline table. Judge-URR is contingent on the GPT-4o judge reaching ≥ 85% accuracy on the 100-pair manually labeled calibration set; if below 85%, drop judge-URR from the headline and report only F1-URR and EM-URR. Document the judge's calibration accuracy in the paper regardless.
  - **Reasoning (user):** "F1 drop > 5pp as the primary continuous signal, plus twin strict-EM URR and judge-URR reported in the headline table. Judge-URR is contingent on the GPT-4o judge reaching at least 85 percent accuracy."
  - **Resolves:** Q1.
  - **Revisit if:** the calibration set itself proves systematically biased (e.g. all single-hop) — we'd recalibrate on a multi-hop-skewed slice.

- **D19 — Six-table normalized Parquet storage (resolves Q6):** layout in `docs/storage_layout.md`. Chunks deduplicated globally by `content_hash`; many-to-many step↔chunk linkage in `chunk_appearances`.
  - **Reasoning (user):** "The six-table Parquet layout proposed in docs/storage_layout.md matches the normalized ChunkAppearance design in the schema and will be much faster to query than nested JSON."
  - **Resolves:** Q6.
  - **Revisit if:** paper supplementary requires a self-contained per-trace export — we'd add a nested export step rather than restructure the working store.

- **D20 — Project Day 0 is 2026-05-14 (resolves Q9):** all Day-0-anchored milestones (week 1 = May 14–21, etc.) use the 2026-05-14 anchor across all files. Harness "today" of 2026-05-15 is fine for runtime timestamps but does not shift milestones.
  - **Reasoning (user):** "Project Day 0 is 2026-05-14. Use this across all files including hypotheses.md, decisions.md timestamps, and any harness 'today' references."
  - **Resolves:** Q9.
  - **Revisit if:** never.

- **D21 — Hybrid determinism contract (resolves Q10):** For closed-model paths (Azure OpenAI gpt-4o-mini, gpt-4o) the SQLite prompt cache is the contract — a cache hit returns the stored response byte-for-byte; a cache miss on replay is treated as a replay failure to be logged and surfaced, **not silently re-queried**. For Phase 3 robustness experiments with open generators on vLLM, enable true byte-determinism (deterministic kernels, fixed seed, greedy decoding) and assert byte-identical outputs in the determinism tests.
  - **Reasoning (user):** "For closed-model paths, the SQLite prompt cache is the contract for replay determinism: a cache hit returns the stored response byte-for-byte; a cache miss on replay is treated as a replay failure to be logged and surfaced, not silently re-queried."
  - **Resolves:** Q10.
  - **Revisit if:** an Azure model deprecation forces us to re-query previously-cached prompts to refresh against a successor model — at that point we'd consciously invalidate and re-cache, not silently re-query mid-replay.

### Project-tooling decisions (D22–D23)

- **D22 — Dependency policy:** `pyproject.toml` uses `>=` ranges; `uv.lock` is committed from day one and is the reproducibility source of truth; `uv.lock` is never hand-edited.
  - **Reasoning (user):** "Lazy add via uv as consumers land … never pin minor versions there. Commit uv.lock from day one and never edit it by hand — the lockfile is the source of truth for reproducibility."
  - **How to apply:** Add new runtime deps with `uv add <pkg>`, dev deps with `uv add --dev <pkg>`. Let the resolver pick versions; do not set upper bounds in `pyproject.toml` unless a known incompatibility forces it. Reproduce another machine's environment with `uv sync`.
  - **Revisit if:** a transitive resolution conflict forces a temporary upper bound — in which case the bound is documented in `pyproject.toml` with a comment pointing to the conflict, and lifted once resolved.

- **D23 — Cache miss on replay surfaces as a custom exception:** Strict-replay cache misses raise `ReplayCacheMissError` (to be defined in `src/traceaudit/trace/exceptions.py` when the prompt-cache module lands in step 5d). The replayer does **not** catch this exception; a miss propagates as a hard replay failure with the missing `prompt_hash` and full request payload attached. No `None` sentinels, no `Optional[Response]` returns.
  - **Reasoning (user):** "Sentinels get swallowed by accident and replay determinism is too important to lose to a None-check that someone forgot."
  - **Companion to:** D21 (hybrid determinism contract). Together they say: cache hit → byte-exact response; cache miss in strict-replay mode → loud failure, no silent re-query.
  - **Where it lives:** `src/traceaudit/trace/exceptions.py` (will be created in step 5d). The decision is recorded now so `io.py` and downstream modules use a consistent error-handling pattern from the start.
  - **Revisit if:** never.

- **D24 (2026-05-14):** Project namespace is `traceaudit`. The top-level `trace` package name collides with the Python stdlib `trace` module under editable installs — `uv sync` adds `src/` to `sys.path` via a `.pth` file appended after stdlib, so `import trace` would resolve to stdlib, not ours. All project modules live under `traceaudit.*`. Rationale: avoid import-resolution ambiguity for any contributor or CI environment. Verified by inspecting `/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/trace.py`.
  - **Resolves:** the import-collision discovered between commits 2 and 5a.
  - **How to apply:** all imports use `traceaudit.*`; the `traceaudit` package root re-exports nothing (empty `__all__`) — consumers import from the leaf module they need.
  - **Revisit if:** never.

### Schema v0.1.2 (D25)

- **D25 — Tool calls in `outputs_hash`** (2026-05-14)
  - **Decision:** Tool calls are part of the deterministic output of a generation event. Replay equality requires `response_text` byte-equality **and** `tool_calls` structural equality on (`name`, `arguments_json` under canonical-JSON, `call_id`); `result_json` is compared only when both sides have it. `compute_outputs_hash` is extended to cover `tool_calls`. Schema bumped 0.1.1 → 0.1.2.
  - **Reasoning (user):** "ToolCall was captured structurally in v0.1.1 precisely to enable structural reasoning downstream; treating tool calls as opaque text in the determinism witness would defeat that choice and hide a real class of determinism failures inside response_text string comparison. Bump motivated now (before Phase 0 milestone) so the cache (5d) and replayer (5c) are written against the correct outputs_hash semantics from the start, not refactored after."
  - **Hash payload (precise):** per generation event in trace order, the payload contributes `{"response_text": ev.response_text, "tool_calls": [{"name": tc.name, "arguments_json": tc.arguments_json, "call_id": tc.call_id} for tc in (ev.tool_calls or [])]}`. `result_json` is intentionally excluded — it may not be known at record time when a tool call is in flight, and including it would make `outputs_hash` unstable across replays where the tool's return value re-serializes differently.
  - **Signature:** `compute_outputs_hash` accepts `GenerationEvent` objects directly; payload construction is internal to the function. Rationale: single source of truth for the hash payload shape — callers (recorder, fixture, future replayer assertions) cannot drift from each other on dict structure because they never construct the dicts.
  - **Identity impact:** `outputs_hash` changes for every trace whose generation events have `tool_calls` populated. `trace_id`, `inputs_hash`, `config_hash` are unaffected — none depend on `outputs_hash` or on `tool_calls`. Pre-0.1.2 `outputs_hash` values cannot be compared directly against 0.1.2; would need recomputation. Today this is moot because the synthetic fixture is the only trace in existence.
  - **Resolves:** schema follow-on from D14 / D16; no `open_questions.md` entry was open for it.
  - **Revisit if:** providers ship structured fields beyond `name` / `arguments_json` / `call_id` that the determinism witness should also pin (e.g. tool-call index disagreements between providers), or if symmetric `result_json` comparison becomes load-bearing for replay assertions.

## Deferred items (known limitations, schema v0.1.x)

- **DEFER-1 — `Chunk.metadata` stays an unschematized `dict[str, Any]` in schema v0.1.x.** Per-corpus expected keys (HotpotQA: `title`, `section`; FinanceBench / TAT-DQA: `is_table`, `table_id`, `row_idx`, `column_headers`; etc.) are documented in `docs/storage_layout.md` and validated at the dataset-loader boundary, not in the schema. Revisit in Phase 3 if the dict becomes a maintenance burden.

- **DEFER-2 — Timing fields are inconsistent across event types.** `Step` has `started_at` + `ended_at`; `GenerationEvent` / `RetrievalEvent` / `RerankEvent` have `started_at` + `latency_ms`. Defer normalization to a v0.2.0 schema bump after Phase 0 stabilizes. Not blocking; recorder can compute one form from the other if needed.
