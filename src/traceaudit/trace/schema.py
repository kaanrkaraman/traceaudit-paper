"""Trace schema for the TraceAudit project — *proposed, awaiting Phase 0 approval*.

This module is the single source of truth for the data model that every component
in the project — recorder, replayer, intervention engine, audit metrics, paper
figures — reads and writes. Until this module is approved, no other code is to
be written against it.

------------------------------------------------------------------------------
WHAT A TRACE IS
------------------------------------------------------------------------------

A `Trace` is the complete record of one agentic-RAG run on one query. It is the
unit of analysis for the entire paper.

Each `Trace` contains an ordered list of `Step`s. A `Step` is one round of the
agent's reasoning loop — sub-query formulation, retrieval, reranking, optional
intermediate answer, and a decision about what to do next. A `Step` may contain
several `GenerationEvent`s (e.g. Self-RAG generates a reflection token *and* an
answer in the same step), at most one `RetrievalEvent`, and at most one
`RerankEvent`.

`Step`s are nodes in a tree, not a list. `parent_step_id` is normally just the
previous step (a linear trace is a degenerate tree), but MA-RAG-style fan-out
or FLARE-style lookahead branches fit naturally. Linear-only systems leave
`parent_step_id` set to the previous step's `step_id`; the root step uses `None`.

`Chunk`s are referenced by a stable `content_hash` (SHA256 of raw UTF-8 bytes,
**no normalization** — see "Design choices" below). The same chunk text appearing
across two traces gets the same `content_hash`, which is the basis for both
deduplicated storage (`docs/storage_layout.md`) and reusable intervention
operator outputs.

------------------------------------------------------------------------------
DESIGN CHOICES (and why)
------------------------------------------------------------------------------

1. **Pydantic v2 with `frozen=True, extra="forbid"`.** Traces are immutable
   artifacts; any mutation must be an explicit `.model_copy(update=...)`.
   `extra="forbid"` catches typos in trace files and refuses to silently drop
   fields a future schema version added.

2. **Chunk identity is `sha256(text.encode("utf-8"))`, with no normalization.**
   Normalization (NFKC, lowercasing, whitespace collapsing) is irreversible and
   would conflate distinct evidence — e.g. "Mercury (planet)" and "Mercury
   (element)" can collide after aggressive normalization if their boilerplate
   matches. Exact bytes give exact identity. If we later need fuzzy matching
   we add a *separate* `normalized_hash` field; we never replace `content_hash`.

3. **Tree-structured steps from day one.** The cost is one nullable field. The
   refactor cost when MA-RAG arrives in Phase 3 (optional) would be far higher.
   Resolves `open_questions.md` Q8 pending approval.

4. **Multiple `GenerationEvent`s per `Step`.** Self-RAG emits a reflection
   token before answering; CRAG may run a retrieval evaluator + a query rewriter
   in the same step; FLARE generates a lookahead before deciding to retrieve.
   Each LLM call is a separate event with its own `rendered_prompt` and
   `prompt_hash` so the prompt cache can serve them independently.

5. **`rendered_prompt` stored verbatim.** Deterministic replay requires the
   exact prompt as sent to the model, post-templating. We accept the storage
   cost (a few KB per call) to guarantee replay. `prompt_hash` is the SHA256
   of the verbatim string; it is the cache key.

6. **Deterministic `trace_id`.** Computed from
   `(system_id, system_version, dataset_id, query_id, seed, config_hash)`.
   Same inputs → same ID. Two replays of the same configuration thus share
   a `trace_id`, which lets us spot accidental duplicates and de-dup safely.
   Resolves `open_questions.md` Q7 pending approval.

7. **Cost in USD on every `GenerationEvent`.** Aggregated up to
   `Trace.total_cost_usd`. Required for the 2000 USD project ceiling
   accounting in `costs.md` to be auditable from the trace store alone, with
   no separate ledger.

8. **`AgentDecision.raw_signal` is an opaque dict.** System-specific signals
   (CRAG retrieval-evaluator verdict, Self-RAG reflection token, FLARE
   lookahead confidence) live here. The typed `decision_type` enum captures
   the common abstraction; the dict captures the rest.

9. **Timestamps in UTC with microsecond precision.** Always tz-aware
   `datetime`; serialized as ISO 8601 with `Z` suffix.

10. **Schema version on the `Trace` root.** `SCHEMA_VERSION` constant below.
    Replayer refuses traces with a different major version. Minor bumps are
    additive and forward-compatible.

------------------------------------------------------------------------------
WHAT THIS MODULE DOES NOT INCLUDE
------------------------------------------------------------------------------

- I/O. Parquet write/read lives in `src/traceaudit/trace/io.py` (Phase 0, after approval).
- Recorder hooks. The recorder wraps an agent runtime; it lives in
  `src/traceaudit/trace/recorder.py`.
- Replayer logic. Determinism enforcement and cache wiring live in
  `src/traceaudit/trace/replayer.py`.
- Intervention operators. `remove`, `paraphrase`, `distract` live in
  `src/traceaudit/intervention/` from Phase 1 onwards.

Those modules will import from this one; this module imports only stdlib and
Pydantic v2.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

SCHEMA_VERSION: str = "0.1.1"
"""Semantic version of this trace schema.

- Major bump on any breaking change (field renamed, required field added,
  type narrowed). Replayer refuses mismatched major versions.
- Minor bump on additive, forward-compatible changes (optional field added).
- Patch bump on documentation or validator-only changes.

Bump the version *in the same commit* as the change, and add an entry to
`migrations/CHANGELOG.md`."""


# -----------------------------------------------------------------------------
# Type aliases for self-documentation. Pydantic treats these as `str`.
# -----------------------------------------------------------------------------

TraceId = Annotated[str, Field(min_length=1, max_length=128)]
"""Stable identifier for a trace. Deterministic hash; see `compute_trace_id`."""

StepId = Annotated[str, Field(min_length=1, max_length=128)]
"""Stable identifier for a step. Hash of `(trace_id, step_index)` by convention."""

EventId = Annotated[str, Field(min_length=1, max_length=128)]
"""Stable identifier for a generation / retrieval / rerank event."""

ContentHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""SHA256 hex digest of a chunk's raw UTF-8 bytes. Lowercase, exactly 64 chars."""

PromptHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""SHA256 hex digest of a verbatim rendered prompt. Cache key."""

ConfigHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""SHA256 hex digest of the canonical-JSON-serialized Hydra config."""


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class StepIntent(str, Enum):
    """The agent's intent for a given step.

    `INITIAL` is the first retrieval against the user's original query.
    `SUBQUERY` is a re-retrieval against a rewritten or decomposed query.
    `VERIFICATION` is a retrieval whose purpose is to confirm or refute a
    candidate answer (e.g. CRAG's confidence check, FLARE's lookahead).
    `ANSWER` is a step that produces no new retrieval, only a final answer.
    """

    INITIAL = "initial"
    SUBQUERY = "subquery"
    VERIFICATION = "verification"
    ANSWER = "answer"


class DecisionType(str, Enum):
    """The high-level action the agent took at the end of a step.

    System-specific nuances (Self-RAG reflection tokens, CRAG retrieval-evaluator
    verdicts, FLARE confidence thresholds) live in `AgentDecision.raw_signal`.
    """

    CONTINUE = "continue"  # take another step with a refined or new query
    STOP = "stop"  # accept current evidence, move to answer generation
    BRANCH = "branch"  # spawn child steps (MA-RAG fan-out, FLARE lookahead)
    REFINE_QUERY = "refine_query"  # CRAG-style query rewrite
    REJECT_RETRIEVAL = "reject_retrieval"  # CRAG "Incorrect" verdict triggering web search
    ANSWER = "answer"  # this step *is* the final answer generation


class ChunkAppearanceRole(str, Enum):
    """The role a chunk played at a step.

    A single chunk can appear in up to three roles in the same step: it was
    retrieved, then survived reranking, then ended up in the generator's
    context. The intervention engine cares which role(s) are touched.
    """

    RETRIEVED = "retrieved"
    RERANKED = "reranked"
    IN_CONTEXT = "in_context"


class ModelProvider(str, Enum):
    """LLM provider identifier. Extend as needed; replayer dispatches on this."""

    AZURE_OPENAI = "azure_openai"
    AZURE_MAAS = "azure_maas"  # Azure-hosted Llama 3.1 70B / 8B
    OPENAI = "openai"
    VLLM_LOCAL = "vllm_local"


# -----------------------------------------------------------------------------
# Base model with project-wide defaults
# -----------------------------------------------------------------------------


class _Frozen(BaseModel):
    """Project-wide Pydantic base: immutable, strict on unknown fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,  # never silently mutate user-facing strings
        validate_assignment=True,
        ser_json_timedelta="iso8601",
    )


# -----------------------------------------------------------------------------
# Leaf models
# -----------------------------------------------------------------------------


class Chunk(_Frozen):
    """A unit of retrieved evidence.

    Identity is `content_hash` (SHA256 of `text.encode("utf-8")`). The same
    chunk text retrieved across many traces, many systems, many steps shares
    one `content_hash` and therefore one row in the global `chunks.parquet`
    table (see `docs/storage_layout.md`).
    """

    content_hash: ContentHash = Field(
        description="SHA256 hex of raw UTF-8 bytes. Use `compute_content_hash(text)`.",
    )
    text: str = Field(
        description="Verbatim chunk text. Stored once globally; referenced by hash elsewhere.",
    )
    source_id: str = Field(
        description=(
            "Stable identifier of the source document in the underlying corpus "
            "(e.g. Wikipedia title for HotpotQA, table id for FinanceBench)."
        ),
    )
    source_offset_start: int | None = Field(
        default=None,
        ge=0,
        description="Character offset of the chunk's start in the source document, if known.",
    )
    source_offset_end: int | None = Field(
        default=None,
        ge=0,
        description="Character offset of the chunk's end in the source document, if known.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-corpus metadata (title, section header, table id, etc.). "
            "Opaque to the schema; queried only by dataset-aware code. "
            "Expected per-corpus keys are documented in `docs/storage_layout.md` "
            "and validated at the dataset-loader boundary, not here (see "
            "decisions.md DEFER-1)."
        ),
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Number of tokens in `text` when tokenized by the generator model's "
            "tokenizer. Tokenizer-specific — recorder populates this against the "
            "generator's tokenizer at recording time. `None` when the count is "
            "not available (e.g. the chunk was retrieved by an offline pipeline "
            "with no generator in scope). Enables audit metrics like 'tokens of "
            "retrieved evidence the agent actually consumed.'"
        ),
    )

    @field_validator("content_hash")
    @classmethod
    def _hash_is_lower_hex(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("content_hash must be lowercase hex")
        return v

    @model_validator(mode="after")
    def _hash_matches_text(self) -> "Chunk":
        expected = compute_content_hash(self.text)
        if expected != self.content_hash:
            raise ValueError(
                f"content_hash mismatch: stored {self.content_hash[:12]}…, "
                f"recomputed {expected[:12]}…"
            )
        if (self.source_offset_start is None) != (self.source_offset_end is None):
            raise ValueError("source offsets must both be set or both be None")
        if (
            self.source_offset_start is not None
            and self.source_offset_end is not None
            and self.source_offset_end < self.source_offset_start
        ):
            raise ValueError("source_offset_end must be ≥ source_offset_start")
        return self


class ToolCall(_Frozen):
    """One tool / function call issued by the model within a `GenerationEvent`.

    Several agentic systems we plan to audit emit semantically distinct payloads
    via tool calls — FLARE's lookahead probability can come back as a tool
    result, MA-RAG sub-agents are dispatched via structured invocations, and
    web-search agents call the search tool with a query string. Parsing those
    out of `response_text` as raw JSON is brittle; capturing them as first-class
    structured records preserves identity and makes replay / intervention
    straightforward.

    Tool round-trips (call → tool execution → next model turn) cross step
    boundaries; `tool_calls` only records what the model emitted in this single
    generation. The downstream effect (a subsequent retrieval, a child step,
    another generation) is recorded by whichever event captures it.
    """

    name: str = Field(
        description="Tool / function name as the model invoked it (e.g. 'web_search').",
    )
    arguments_json: str = Field(
        description=(
            "Canonical-JSON of the call's arguments. Always produce this with "
            "`canonical_json(...)` so key ordering is stable and `call_id` "
            "hashing stays deterministic across replays."
        ),
    )
    result_json: str | None = Field(
        default=None,
        description=(
            "Canonical-JSON of the tool result, when the result was known at "
            "recording time. `None` when the tool round-trip is captured "
            "elsewhere (e.g. the result of a `web_search` call surfaces as a "
            "subsequent `RetrievalEvent`)."
        ),
    )
    call_id: str = Field(
        description=(
            "Provider-issued call id (e.g. OpenAI `tool_call_id`) when "
            "available; otherwise a deterministic fallback computed by "
            "`compute_tool_call_id(name, arguments_json)` (SHA256 of canonical-"
            "JSON of `{name, arguments_json}`). Stable across replays."
        ),
    )


class GenerationEvent(_Frozen):
    """One call to an LLM.

    The combination of (`model_id`, `model_version`, `rendered_prompt`,
    `temperature`, `top_p`, `seed`, `max_tokens`) defines the request. The
    prompt cache is keyed on `prompt_hash` plus the deterministic-relevant
    parameters; cache hits set `cached=True`.
    """

    event_id: EventId
    event_index_in_step: int = Field(
        ge=0,
        description="0-based index among generation events within a step.",
    )

    model_id: str = Field(
        description=(
            "Provider's model identifier, version-pinned where the provider exposes one "
            "(e.g. 'gpt-4o-mini-2024-07-18', 'meta-llama-3.1-70b-instruct')."
        ),
    )
    model_provider: ModelProvider
    model_version: str | None = Field(
        default=None,
        description=(
            "Provider-reported runtime version when not encoded in model_id. "
            "Some Azure deployments expose a separate 'system_fingerprint'."
        ),
    )

    prompt_template_id: str = Field(
        description=(
            "Stable id of the prompt template that produced this prompt "
            "(e.g. 'crag.retrieval_evaluator.v1'). Lets us audit which prompt "
            "phrasings produce which behaviors."
        ),
    )
    prompt_hash: PromptHash = Field(
        description="SHA256 of rendered_prompt (UTF-8). Prompt cache key.",
    )
    rendered_prompt: str = Field(
        description=(
            "Verbatim prompt as sent to the model, post-templating. Required for "
            "deterministic replay. Includes any system message and tool definitions."
        ),
    )

    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    seed: int | None = Field(
        default=None,
        description="Provider-side seed if set. Closed APIs may ignore it; we record what we sent.",
    )
    max_tokens: int = Field(gt=0)

    response_text: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    finish_reason: str = Field(
        description="Provider-reported reason (stop, length, content_filter, tool_calls, …).",
    )
    logprobs_json: str | None = Field(
        default=None,
        description=(
            "Provider logprobs, serialized to JSON. Captured when available, never "
            "depended on for determinism. None when the provider does not expose them."
        ),
    )
    tool_calls: tuple[ToolCall, ...] | None = Field(
        default=None,
        description=(
            "Structured tool / function calls the model issued in this event. "
            "`None` for plain-completion events (the common case). Use `None` to "
            "denote 'no tool calls' — an empty tuple is rejected so the two "
            "encodings of 'nothing' don't drift apart."
        ),
    )

    cached: bool = Field(
        description="True if served from the SQLite prompt cache (no provider call made).",
    )
    cost_usd: float = Field(
        ge=0.0,
        description=(
            "Marginal cost of this call in USD, computed from token counts and the "
            "model's pricing in `costs.md`. Zero when `cached=True`."
        ),
    )
    latency_ms: float = Field(ge=0.0)
    started_at: datetime

    @field_validator("prompt_hash")
    @classmethod
    def _hash_is_lower_hex(cls, v: str) -> str:
        if v != v.lower():
            raise ValueError("prompt_hash must be lowercase hex")
        return v

    @model_validator(mode="after")
    def _prompt_hash_matches(self) -> "GenerationEvent":
        expected = compute_prompt_hash(self.rendered_prompt)
        if expected != self.prompt_hash:
            raise ValueError(
                f"prompt_hash mismatch: stored {self.prompt_hash[:12]}…, "
                f"recomputed {expected[:12]}…"
            )
        if self.cached and self.cost_usd != 0.0:
            raise ValueError("cached generations must have cost_usd == 0.0")
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.tool_calls is not None and len(self.tool_calls) == 0:
            raise ValueError(
                "tool_calls must be None or a non-empty tuple; an empty tuple "
                "is rejected — use None to denote 'no tool calls'"
            )
        return self


class ChunkAppearance(_Frozen):
    """One (step, chunk, role) entry. Many-to-many between steps and chunks."""

    content_hash: ContentHash
    role: ChunkAppearanceRole
    position: int = Field(
        ge=0,
        description="0-based rank within this role's list at this step.",
    )
    score: float | None = Field(
        default=None,
        description=(
            "Retrieval score (role=RETRIEVED) or reranker score (role=RERANKED). "
            "None for role=IN_CONTEXT (no native score)."
        ),
    )


class RetrievalEvent(_Frozen):
    """One call to a retriever."""

    event_id: EventId
    retriever_id: str = Field(
        description=(
            "Stable id of the retrieval system (e.g. "
            "'bge-large-en-v1.5+qdrant', 'web-search-tavily', 'bm25-pyserini')."
        ),
    )
    retriever_version: str | None = Field(default=None)
    retriever_config_json: str = Field(
        description=(
            "Canonical-JSON of the retriever's hyperparameters (top_k, filters, "
            "min_score, hybrid weights, etc.). Required for replay."
        ),
    )
    pre_rewrite_query: str | None = Field(
        default=None,
        description=(
            "The query as the agent originally formulated it, before any "
            "retriever-side rewriting. `None` when no rewrite occurred. The "
            "actually-sent query lives in `query` below."
        ),
    )
    query: str = Field(
        description=(
            "The query as actually sent to the retriever. If `pre_rewrite_query` "
            "is set, this is the post-rewrite form."
        ),
    )
    top_k: int = Field(gt=0)
    latency_ms: float = Field(ge=0.0)
    started_at: datetime

    @model_validator(mode="after")
    def _validate(self) -> "RetrievalEvent":
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if (
            self.pre_rewrite_query is not None
            and self.pre_rewrite_query == self.query
        ):
            raise ValueError(
                "pre_rewrite_query must differ from query — set it to None if no "
                "rewrite occurred"
            )
        return self


class RerankEvent(_Frozen):
    """One reranking pass over an input chunk list."""

    event_id: EventId
    reranker_id: str = Field(
        description="Stable id (e.g. 'bge-reranker-v2-m3', 'cohere-rerank-3').",
    )
    reranker_version: str | None = Field(default=None)
    latency_ms: float = Field(ge=0.0)
    started_at: datetime

    @model_validator(mode="after")
    def _tz_aware(self) -> "RerankEvent":
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        return self


class AgentDecision(_Frozen):
    """What the agent decided to do at the end of a step.

    `decision_type` is the typed abstraction shared across all audited systems.
    `raw_signal` is the system-specific verbatim signal (CRAG verdict, Self-RAG
    reflection token, FLARE confidence). The intervention engine routes on
    `decision_type`; the paper's qualitative analysis reads `raw_signal`.
    """

    decision_type: DecisionType
    rationale: str | None = Field(
        default=None,
        description="Free-text rationale if the agent emits one (some do, some don't).",
    )
    next_query: str | None = Field(
        default=None,
        description="If decision_type is CONTINUE/REFINE_QUERY, the next query.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent-reported confidence (e.g. FLARE lookahead probability).",
    )
    raw_signal: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "System-specific opaque signal. Examples: "
            "{'crag_verdict': 'Ambiguous'}, "
            "{'self_rag_reflect': 'ISREL=Relevant ISSUP=Fully'}, "
            "{'flare_lookahead_prob': 0.42}."
        ),
    )


class Step(_Frozen):
    """One step in the agentic trace — a node in the trace tree."""

    step_id: StepId
    parent_step_id: StepId | None = Field(
        default=None,
        description="None only for the root step.",
    )
    step_index: int = Field(
        ge=0,
        description="Depth-first index of this step within the trace.",
    )

    intent: StepIntent
    query: str = Field(
        description="The query under which this step operates (after any rewrite).",
    )

    generation_events: tuple[GenerationEvent, ...] = Field(
        default_factory=tuple,
        description=(
            "Ordered LLM calls within this step. Self-RAG: reflection then answer. "
            "CRAG: retrieval evaluator, optional query rewriter, optional final answer. "
            "Empty tuple is legal for pure-retrieval steps that defer generation."
        ),
    )
    retrieval_event: RetrievalEvent | None = Field(
        default=None,
        description="At most one retrieval per step. None for pure-generation steps.",
    )
    rerank_event: RerankEvent | None = Field(
        default=None,
        description="At most one rerank per step. None if reranking did not run.",
    )

    chunk_appearances: tuple[ChunkAppearance, ...] = Field(
        default_factory=tuple,
        description=(
            "All (chunk, role) entries for this step. Replaces the three parallel "
            "lists (retrieved / reranked / in_context) with one normalized form. "
            "A chunk that was retrieved, kept by rerank, and used in context has "
            "three entries here."
        ),
    )

    intermediate_answer: str | None = Field(
        default=None,
        description="Candidate answer emitted at this step, if any.",
    )

    decision: AgentDecision

    started_at: datetime
    ended_at: datetime

    @model_validator(mode="after")
    def _structural(self) -> "Step":
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("step timestamps must be timezone-aware")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be ≥ started_at")
        # generation event indices must be 0..N-1 in order
        for i, ev in enumerate(self.generation_events):
            if ev.event_index_in_step != i:
                raise ValueError(
                    f"generation_events[{i}].event_index_in_step "
                    f"= {ev.event_index_in_step}, expected {i}"
                )
        # chunk appearance positions must be unique within (role)
        seen: dict[ChunkAppearanceRole, set[int]] = {}
        for app in self.chunk_appearances:
            bucket = seen.setdefault(app.role, set())
            if app.position in bucket:
                raise ValueError(
                    f"duplicate position {app.position} for role {app.role.value}"
                )
            bucket.add(app.position)
        return self


# -----------------------------------------------------------------------------
# Root model
# -----------------------------------------------------------------------------


class Trace(_Frozen):
    """The complete record of one agentic-RAG run on one query.

    Identity is `trace_id`, which is a deterministic hash of the inputs that
    define the run (`compute_trace_id`). Replaying the same inputs reproduces
    the same `trace_id` — a property the determinism tests rely on.
    """

    trace_id: TraceId
    schema_version: Literal["0.1.1"] = Field(
        default=SCHEMA_VERSION,
        description="Bump in lockstep with the module-level SCHEMA_VERSION constant.",
    )

    # Provenance
    system_id: str = Field(description="e.g. 'crag', 'ircot', 'self-rag', 'flare'.")
    system_version: str = Field(
        description="Git commit SHA of the audited system implementation, or release tag.",
    )
    dataset_id: str = Field(description="e.g. 'hotpotqa-dev', 'musique-dev'.")
    query_id: str = Field(description="Stable id of the query within the dataset.")

    # Inputs
    original_query: str = Field(description="The user's verbatim query.")
    gold_answers: tuple[str, ...] = Field(
        description=(
            "Reference answers used for evaluation only. Never visible to the agent. "
            "Tuple, not list, to enforce immutability."
        ),
    )

    # Environment
    seed: int = Field(description="Master RNG seed for this run.")
    config_hash: ConfigHash = Field(
        description="SHA256 of canonical-JSON-serialized Hydra config.",
    )
    config_json: str = Field(
        description="Full canonical-JSON of the Hydra config. Opaque; included for replay.",
    )
    git_commit_sha: str = Field(
        description="Git commit SHA of this repository at the time of capture.",
    )
    package_versions_json: str = Field(
        description=(
            "Canonical-JSON of `{pkg_name: version}` for the packages whose behavior "
            "could affect the trace (openai, anthropic, sentence-transformers, "
            "qdrant-client, transformers, torch, …)."
        ),
    )

    # Steps
    steps: tuple[Step, ...] = Field(description="Steps in depth-first order.")

    # Final output
    final_answer: str
    final_answer_tokens: int = Field(ge=0)

    # Timing
    started_at: datetime
    ended_at: datetime
    total_cost_usd: float = Field(ge=0.0)

    # Determinism witnesses
    inputs_hash: str = Field(
        description=(
            "SHA256 of canonical-JSON of the run's inputs "
            "(system_id, system_version, dataset_id, query_id, seed, config_hash, "
            "original_query). Two runs with the same inputs_hash should produce "
            "the same outputs_hash on a warm cache."
        ),
    )
    outputs_hash: str = Field(
        description=(
            "SHA256 of canonical-JSON of (final_answer, [ev.response_text for ev in "
            "all generation events in step order]). Determinism tests assert this "
            "is byte-stable across replays."
        ),
    )

    @model_validator(mode="after")
    def _structural(self) -> "Trace":
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("trace timestamps must be timezone-aware")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be ≥ started_at")

        if not self.steps:
            raise ValueError("trace must contain at least one step")

        # step_index contiguous, root has parent=None
        step_ids = {s.step_id for s in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("duplicate step_id within trace")
        for i, step in enumerate(self.steps):
            if step.step_index != i:
                raise ValueError(f"steps[{i}].step_index = {step.step_index}, expected {i}")
            if step.parent_step_id is not None and step.parent_step_id not in step_ids:
                raise ValueError(
                    f"step {step.step_id} references unknown parent {step.parent_step_id}"
                )

        if self.steps[0].parent_step_id is not None:
            raise ValueError("root step (steps[0]) must have parent_step_id=None")

        # total_cost_usd matches sum of event costs (allow tiny FP slack)
        summed = sum(
            ev.cost_usd for step in self.steps for ev in step.generation_events
        )
        if abs(summed - self.total_cost_usd) > 1e-6:
            raise ValueError(
                f"total_cost_usd {self.total_cost_usd} ≠ sum of event costs {summed}"
            )

        return self


# -----------------------------------------------------------------------------
# Hash helpers — single source of truth for all hashing in the project.
# -----------------------------------------------------------------------------


def compute_content_hash(text: str) -> str:
    """SHA256 of `text.encode('utf-8')`, lowercase hex. No normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_prompt_hash(rendered_prompt: str) -> str:
    """SHA256 of the verbatim rendered prompt. The prompt cache's key."""
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys, no whitespace — stable across Python versions.

    Used for hashing configs and output bundles. Never use `json.dumps(obj)` for
    hash inputs anywhere in the project; always go through this function.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(config: dict[str, Any]) -> str:
    """SHA256 of the canonical-JSON of a Hydra config dict."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def compute_trace_id(
    *,
    system_id: str,
    system_version: str,
    dataset_id: str,
    query_id: str,
    seed: int,
    config_hash: str,
) -> str:
    """Deterministic trace_id from inputs. Same inputs → same id."""
    payload = canonical_json(
        {
            "system_id": system_id,
            "system_version": system_version,
            "dataset_id": dataset_id,
            "query_id": query_id,
            "seed": seed,
            "config_hash": config_hash,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_inputs_hash(
    *,
    system_id: str,
    system_version: str,
    dataset_id: str,
    query_id: str,
    seed: int,
    config_hash: str,
    original_query: str,
) -> str:
    """Hash of the full inputs bundle. Witnesses determinism alongside `outputs_hash`."""
    payload = canonical_json(
        {
            "system_id": system_id,
            "system_version": system_version,
            "dataset_id": dataset_id,
            "query_id": query_id,
            "seed": seed,
            "config_hash": config_hash,
            "original_query": original_query,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_outputs_hash(*, final_answer: str, generation_responses: list[str]) -> str:
    """Hash of (final_answer, ordered generation responses). The determinism witness."""
    payload = canonical_json(
        {"final_answer": final_answer, "generation_responses": generation_responses}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_tool_call_id(*, name: str, arguments_json: str) -> str:
    """Deterministic fallback `call_id` for providers that don't return one.

    Same `(name, arguments_json)` → same id, so two identical tool calls share
    an id even across replays. Providers that return their own ids (e.g. OpenAI
    `tool_call_id`) should pass those through unmodified rather than recomputing.
    """
    payload = canonical_json({"name": name, "arguments_json": arguments_json})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now() -> datetime:
    """`datetime.now(tz=UTC)` — single helper so timestamps are always tz-aware."""
    return datetime.now(tz=timezone.utc)


__all__ = [
    "SCHEMA_VERSION",
    "TraceId",
    "StepId",
    "EventId",
    "ContentHash",
    "PromptHash",
    "ConfigHash",
    "StepIntent",
    "DecisionType",
    "ChunkAppearanceRole",
    "ModelProvider",
    "Chunk",
    "ToolCall",
    "GenerationEvent",
    "ChunkAppearance",
    "RetrievalEvent",
    "RerankEvent",
    "AgentDecision",
    "Step",
    "Trace",
    "compute_content_hash",
    "compute_prompt_hash",
    "compute_config_hash",
    "compute_trace_id",
    "compute_inputs_hash",
    "compute_outputs_hash",
    "compute_tool_call_id",
    "canonical_json",
    "utc_now",
]
