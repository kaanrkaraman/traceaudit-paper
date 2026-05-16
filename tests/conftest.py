"""Pytest fixtures for trace I/O tests.

The central fixture is `synthetic_trace_and_chunks`, which builds a small but
realistic two-step trace with retrieval, multiple chunk roles, a tool call,
and exact cost-sum reconciliation. The same builder backs the second-trace
fixture used by the atomic-rename test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from traceaudit.trace.schema import (
    AgentDecision,
    Chunk,
    ChunkAppearance,
    ChunkAppearanceRole,
    DecisionType,
    GenerationEvent,
    ModelProvider,
    RetrievalEvent,
    Step,
    StepIntent,
    ToolCall,
    Trace,
    canonical_json,
    compute_config_hash,
    compute_content_hash,
    compute_inputs_hash,
    compute_outputs_hash,
    compute_prompt_hash,
    compute_tool_call_id,
    compute_trace_id,
)

_T0 = datetime(2026, 5, 14, 0, 0, 0, tzinfo=timezone.utc)


def _chunk(text: str, source_id: str) -> Chunk:
    return Chunk(
        content_hash=compute_content_hash(text),
        text=text,
        source_id=source_id,
        token_count=len(text.split()),
    )


def _gen(
    *,
    step_id: str,
    idx: int,
    prompt: str,
    response: str,
    cost: float,
    started_at: datetime,
    tool_calls: tuple[ToolCall, ...] | None = None,
) -> GenerationEvent:
    return GenerationEvent(
        event_id=f"{step_id}-gen-{idx}",
        event_index_in_step=idx,
        model_id="gpt-4o-mini-2024-07-18",
        model_provider=ModelProvider.AZURE_OPENAI,
        model_version="2024-07-18",
        prompt_template_id="test.v1",
        prompt_hash=compute_prompt_hash(prompt),
        rendered_prompt=prompt,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_tokens=256,
        response_text=response,
        prompt_tokens=max(1, len(prompt) // 4),
        completion_tokens=max(1, len(response) // 4),
        cached=False,
        finish_reason="stop",
        tool_calls=tool_calls,
        cost_usd=cost,
        latency_ms=120.0,
        started_at=started_at,
    )


def build_synthetic_trace(
    *,
    system_id: str = "test-system",
    dataset_id: str = "test-dataset",
    query_id: str = "q-001",
    original_query: str = "What is Mercury?",
) -> tuple[Trace, dict[str, Chunk]]:
    """Build a deterministic two-step trace plus the chunks it references."""
    config = {"model": "gpt-4o-mini", "top_k": 5, "seed": 42}
    config_hash = compute_config_hash(config)
    config_json = canonical_json(config)

    c1 = _chunk(
        "Mercury is the smallest planet in the Solar System.", "wiki:Mercury_(planet)"
    )
    c2 = _chunk("Mercury orbits closest to the Sun.", "wiki:Mercury_(planet)")
    c3 = _chunk("Mercury's day is longer than its year.", "wiki:Mercury_(planet)")
    chunks_map: dict[str, Chunk] = {c.content_hash: c for c in (c1, c2, c3)}

    # Step 1 — initial retrieval, a tool call from the model, then a stop decision.
    args_json = canonical_json({"query": original_query, "k": 5})
    tool_call = ToolCall(
        name="web_search",
        arguments_json=args_json,
        result_json=None,
        call_id=compute_tool_call_id(name="web_search", arguments_json=args_json),
    )
    gen1 = _gen(
        step_id="step-1",
        idx=0,
        prompt=f"Question: {original_query}\nUse the web_search tool if needed.",
        response="<calling web_search>",
        cost=0.0012,
        started_at=_T0,
        tool_calls=(tool_call,),
    )
    retrieval = RetrievalEvent(
        event_id="step-1-ret",
        retriever_id="bge-large-en-v1.5+qdrant",
        retriever_version="1.5",
        retriever_config_json=canonical_json({"top_k": 5, "min_score": 0.5}),
        pre_rewrite_query=None,
        query=original_query,
        top_k=5,
        latency_ms=42.0,
        started_at=_T0 + timedelta(milliseconds=200),
    )
    apps_step1 = (
        ChunkAppearance(
            content_hash=c1.content_hash,
            role=ChunkAppearanceRole.RETRIEVED,
            position=0,
            score=0.93,
        ),
        ChunkAppearance(
            content_hash=c2.content_hash,
            role=ChunkAppearanceRole.RETRIEVED,
            position=1,
            score=0.88,
        ),
        ChunkAppearance(
            content_hash=c3.content_hash,
            role=ChunkAppearanceRole.RETRIEVED,
            position=2,
            score=0.71,
        ),
        ChunkAppearance(
            content_hash=c1.content_hash,
            role=ChunkAppearanceRole.IN_CONTEXT,
            position=0,
            score=None,
        ),
        ChunkAppearance(
            content_hash=c2.content_hash,
            role=ChunkAppearanceRole.IN_CONTEXT,
            position=1,
            score=None,
        ),
    )
    step1 = Step(
        step_id="step-1",
        parent_step_id=None,
        step_index=0,
        intent=StepIntent.INITIAL,
        query=original_query,
        generation_events=(gen1,),
        retrieval_event=retrieval,
        rerank_event=None,
        chunk_appearances=apps_step1,
        intermediate_answer=None,
        decision=AgentDecision(
            decision_type=DecisionType.CONTINUE,
            rationale="initial retrieval ok; refine for orbit details",
            next_query="Mercury orbital period",
            confidence=0.7,
            raw_signal={"crag_verdict": "Ambiguous"},
        ),
        started_at=_T0,
        ended_at=_T0 + timedelta(seconds=1),
    )

    # Step 2 — pure answer generation, no retrieval, no tool calls.
    final_response = "Mercury is the smallest planet, closest to the Sun."
    gen2 = _gen(
        step_id="step-2",
        idx=0,
        prompt=(
            f"Given:\n- {c1.text}\n- {c2.text}\n"
            f"Question: {original_query}\nAnswer concisely."
        ),
        response=final_response,
        cost=0.0008,
        started_at=_T0 + timedelta(seconds=2),
    )
    step2 = Step(
        step_id="step-2",
        parent_step_id="step-1",
        step_index=1,
        intent=StepIntent.ANSWER,
        query=original_query,
        generation_events=(gen2,),
        retrieval_event=None,
        rerank_event=None,
        chunk_appearances=(),
        intermediate_answer=None,
        decision=AgentDecision(
            decision_type=DecisionType.ANSWER,
            rationale="evidence sufficient",
            confidence=0.9,
            raw_signal={},
        ),
        started_at=_T0 + timedelta(seconds=2),
        ended_at=_T0 + timedelta(seconds=3),
    )

    total_cost = gen1.cost_usd + gen2.cost_usd

    trace_id = compute_trace_id(
        system_id=system_id,
        system_version="test-0.1",
        dataset_id=dataset_id,
        query_id=query_id,
        seed=42,
        config_hash=config_hash,
    )
    inputs_hash = compute_inputs_hash(
        system_id=system_id,
        system_version="test-0.1",
        dataset_id=dataset_id,
        query_id=query_id,
        seed=42,
        config_hash=config_hash,
        original_query=original_query,
    )
    outputs_hash = compute_outputs_hash(
        final_answer=final_response,
        generation_responses=[gen1.response_text, gen2.response_text],
    )

    trace = Trace(
        trace_id=trace_id,
        system_id=system_id,
        system_version="test-0.1",
        dataset_id=dataset_id,
        query_id=query_id,
        original_query=original_query,
        gold_answers=("Mercury",),
        seed=42,
        config_hash=config_hash,
        config_json=config_json,
        git_commit_sha="0" * 40,
        package_versions_json=canonical_json(
            {"pydantic": "2.13.4", "pyarrow": "24.0.0", "python": "3.11.9"}
        ),
        steps=(step1, step2),
        final_answer=final_response,
        final_answer_tokens=len(final_response.split()),
        started_at=_T0,
        ended_at=_T0 + timedelta(seconds=3),
        total_cost_usd=total_cost,
        inputs_hash=inputs_hash,
        outputs_hash=outputs_hash,
    )
    return trace, chunks_map


@pytest.fixture
def synthetic_trace_and_chunks() -> tuple[Trace, dict[str, Chunk]]:
    return build_synthetic_trace()


@pytest.fixture
def second_synthetic_trace_and_chunks() -> tuple[Trace, dict[str, Chunk]]:
    return build_synthetic_trace(query_id="q-002", original_query="What is Venus?")
