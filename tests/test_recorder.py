"""Recorder behaviour and invariants.

The strongest single test is `test_recorder_reconstructs_the_synthetic_trace`:
it uses the same `synthetic_trace_and_chunks` fixture that backs the I/O round-
trip, takes its `Trace` apart, feeds it back through the `Recorder` API, and
asserts the resulting trace is byte-identical at every identity-bearing hash
(`trace_id`, `inputs_hash`, `outputs_hash`, `total_cost_usd`, every chunk's
`content_hash`, every generation event's `prompt_hash`). If the recorder's
hash computation, cost summation, or chunk handling drifts, this test fails.

The remaining tests pin the failure modes the recorder is responsible for
catching at the recorder boundary, before the schema validators do:

- unregistered chunk references in a step's `chunk_appearances`,
- chunk-hash collision (same `content_hash`, different `Chunk` content),
- non-contiguous `step_index`,
- orphan `parent_step_id`,
- finalize with zero steps,
- naive (tz-unaware) `started_at` / `ended_at`,
- end-to-end round-trip through `TraceStore`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from traceaudit.trace.io import TraceStore
from traceaudit.trace.recorder import Recorder
from traceaudit.trace.schema import (
    AgentDecision,
    Chunk,
    ChunkAppearance,
    ChunkAppearanceRole,
    DecisionType,
    GenerationEvent,
    ModelProvider,
    Step,
    StepIntent,
    Trace,
    compute_content_hash,
    compute_prompt_hash,
)


def _recorder_for(trace: Trace, *, clock=None) -> Recorder:
    """Construct a Recorder pre-loaded with the same metadata as `trace`."""
    kwargs: dict = dict(
        system_id=trace.system_id,
        system_version=trace.system_version,
        dataset_id=trace.dataset_id,
        query_id=trace.query_id,
        original_query=trace.original_query,
        gold_answers=trace.gold_answers,
        seed=trace.seed,
        config=json.loads(trace.config_json),
        git_commit_sha=trace.git_commit_sha,
        package_versions=json.loads(trace.package_versions_json),
        started_at=trace.started_at,
    )
    if clock is not None:
        kwargs["clock"] = clock
    return Recorder(**kwargs)


def test_recorder_reconstructs_the_synthetic_trace(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    """Headline test: Recorder API reproduces a ground-truth Trace byte-identically."""
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)

    rec.register_chunks(direct_chunks.values())
    for step in direct_trace.steps:
        rec.append_step(step)
    rebuilt, rebuilt_chunks = rec.finalize(
        final_answer=direct_trace.final_answer,
        final_answer_tokens=direct_trace.final_answer_tokens,
        ended_at=direct_trace.ended_at,
    )

    # Every identity-bearing field is byte-identical.
    assert rebuilt.trace_id == direct_trace.trace_id
    assert rebuilt.inputs_hash == direct_trace.inputs_hash
    assert rebuilt.outputs_hash == direct_trace.outputs_hash
    assert rebuilt.config_hash == direct_trace.config_hash
    assert rebuilt.config_json == direct_trace.config_json
    assert rebuilt.package_versions_json == direct_trace.package_versions_json
    assert rebuilt.total_cost_usd == pytest.approx(
        direct_trace.total_cost_usd, abs=1e-12
    )
    assert rebuilt.started_at == direct_trace.started_at
    assert rebuilt.ended_at == direct_trace.ended_at
    assert rebuilt.steps == direct_trace.steps
    assert rebuilt.gold_answers == direct_trace.gold_answers
    assert rebuilt.final_answer == direct_trace.final_answer
    assert rebuilt.schema_version == direct_trace.schema_version

    # Chunks dict is trimmed exactly to the referenced set (same set the direct
    # fixture exposes).
    assert set(rebuilt_chunks.keys()) == set(direct_chunks.keys())
    for h, c in direct_chunks.items():
        assert rebuilt_chunks[h] == c


def test_recorder_output_round_trips_through_trace_store(
    tmp_path: Path,
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    """End-to-end: Recorder -> TraceStore.write -> TraceStore.read preserves identity."""
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    rec.register_chunks(direct_chunks.values())
    for step in direct_trace.steps:
        rec.append_step(step)
    trace, chunks = rec.finalize(
        final_answer=direct_trace.final_answer,
        final_answer_tokens=direct_trace.final_answer_tokens,
        ended_at=direct_trace.ended_at,
    )

    store = TraceStore(tmp_path)
    store.write_trace(trace, chunks)
    read_back = store.read_trace(
        trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
    )
    assert read_back.trace_id == trace.trace_id
    assert read_back.inputs_hash == trace.inputs_hash
    assert read_back.outputs_hash == trace.outputs_hash


def test_recorder_rejects_step_referencing_unregistered_chunk(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, _ = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    # Skip register_chunk entirely; the first step has chunk_appearances.
    first_step = direct_trace.steps[0]
    assert first_step.chunk_appearances, "fixture must exercise chunk appearances"
    with pytest.raises(ValueError, match="not registered"):
        rec.append_step(first_step)


def test_recorder_rejects_chunk_hash_collision(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    """Re-registering the same content_hash with a different Chunk raises."""
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    original = next(iter(direct_chunks.values()))
    rec.register_chunk(original)
    # Same content_hash, different source_id — surfaces as a collision.
    impostor = Chunk(
        content_hash=original.content_hash,
        text=original.text,
        source_id=original.source_id + "-impostor",
        token_count=original.token_count,
    )
    with pytest.raises(ValueError, match="hash collision"):
        rec.register_chunk(impostor)


def test_recorder_register_chunk_is_idempotent_on_identical_re_registration(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    chunk = next(iter(direct_chunks.values()))
    rec.register_chunk(chunk)
    rec.register_chunk(chunk)  # no-op, must not raise
    assert rec.chunks[chunk.content_hash] == chunk


def test_recorder_rejects_non_contiguous_step_index(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    rec.register_chunks(direct_chunks.values())
    # The fixture's step 2 has step_index=1; appending it first should fail.
    second_step = direct_trace.steps[1]
    assert second_step.step_index == 1
    with pytest.raises(ValueError, match="step_index"):
        rec.append_step(second_step)


def test_recorder_rejects_orphan_parent_step_id(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    rec.register_chunks(direct_chunks.values())
    # Append a step_index=0 step whose parent_step_id is non-None: invalid root.
    root = direct_trace.steps[0]
    bad_root = root.model_copy(update={"parent_step_id": "ghost-step"})
    with pytest.raises(ValueError, match="root step"):
        rec.append_step(bad_root)


def test_recorder_rejects_non_root_with_unknown_parent(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    rec.register_chunks(direct_chunks.values())
    rec.append_step(direct_trace.steps[0])
    second = direct_trace.steps[1]
    bad_second = second.model_copy(update={"parent_step_id": "ghost-step"})
    with pytest.raises(ValueError, match="does not match"):
        rec.append_step(bad_second)


def test_recorder_finalize_requires_at_least_one_step(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, _ = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    with pytest.raises(ValueError, match="no appended steps"):
        rec.finalize(
            final_answer="anything",
            final_answer_tokens=1,
            ended_at=direct_trace.ended_at,
        )


def test_recorder_rejects_tz_naive_timestamps(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, _ = synthetic_trace_and_chunks
    with pytest.raises(ValueError, match="timezone-aware"):
        Recorder(
            system_id=direct_trace.system_id,
            system_version=direct_trace.system_version,
            dataset_id=direct_trace.dataset_id,
            query_id=direct_trace.query_id,
            original_query=direct_trace.original_query,
            gold_answers=direct_trace.gold_answers,
            seed=direct_trace.seed,
            config=json.loads(direct_trace.config_json),
            git_commit_sha=direct_trace.git_commit_sha,
            package_versions=json.loads(direct_trace.package_versions_json),
            started_at=datetime(2026, 5, 14),  # naive
        )


def test_recorder_rejects_ended_at_before_started_at(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    direct_trace, direct_chunks = synthetic_trace_and_chunks
    rec = _recorder_for(direct_trace)
    rec.register_chunks(direct_chunks.values())
    for step in direct_trace.steps:
        rec.append_step(step)
    earlier = direct_trace.started_at - timedelta(seconds=1)
    with pytest.raises(ValueError, match="precedes started_at"):
        rec.finalize(
            final_answer=direct_trace.final_answer,
            final_answer_tokens=direct_trace.final_answer_tokens,
            ended_at=earlier,
        )


def test_recorder_trace_id_is_deterministic_across_constructions(
    synthetic_trace_and_chunks: tuple[Trace, dict[str, Chunk]],
) -> None:
    """Two recorders built with the same inputs produce the same `trace_id`."""
    direct_trace, _ = synthetic_trace_and_chunks
    rec_a = _recorder_for(direct_trace)
    rec_b = _recorder_for(direct_trace)
    assert rec_a.trace_id == rec_b.trace_id
    assert rec_a.inputs_hash == rec_b.inputs_hash
    assert rec_a.config_hash == rec_b.config_hash


def test_recorder_uses_clock_when_started_at_not_provided() -> None:
    """When `started_at` is omitted, the injected `clock` callable is consulted."""
    pinned = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    rec = Recorder(
        system_id="x",
        system_version="0",
        dataset_id="x",
        query_id="q",
        original_query="?",
        gold_answers=("a",),
        seed=0,
        config={},
        git_commit_sha="0" * 40,
        package_versions={},
        clock=lambda: pinned,
    )
    assert rec.started_at == pinned


def test_recorder_finalize_uses_clock_when_ended_at_not_provided() -> None:
    """`ended_at` defaults to `clock()` when omitted on finalize."""
    pinned_start = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)
    pinned_end = datetime(2026, 5, 14, 12, 0, 5, tzinfo=timezone.utc)
    clock_calls = iter([pinned_start, pinned_end])
    rec = Recorder(
        system_id="x",
        system_version="0",
        dataset_id="ds",
        query_id="q",
        original_query="?",
        gold_answers=("a",),
        seed=0,
        config={},
        git_commit_sha="0" * 40,
        package_versions={},
        clock=lambda: next(clock_calls),
    )
    text = "evidence"
    c = Chunk(content_hash=compute_content_hash(text), text=text, source_id="src:1")
    rec.register_chunk(c)
    prompt = "What?"
    ev = GenerationEvent(
        event_id="e0",
        event_index_in_step=0,
        model_id="m",
        model_provider=ModelProvider.AZURE_OPENAI,
        prompt_template_id="t.v1",
        prompt_hash=compute_prompt_hash(prompt),
        rendered_prompt=prompt,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        response_text="ok",
        prompt_tokens=1,
        completion_tokens=1,
        cached=False,
        finish_reason="stop",
        cost_usd=0.0,
        latency_ms=1.0,
        started_at=pinned_start,
    )
    step = Step(
        step_id="s0",
        parent_step_id=None,
        step_index=0,
        intent=StepIntent.ANSWER,
        query="?",
        generation_events=(ev,),
        chunk_appearances=(
            ChunkAppearance(
                content_hash=c.content_hash,
                role=ChunkAppearanceRole.IN_CONTEXT,
                position=0,
            ),
        ),
        decision=AgentDecision(decision_type=DecisionType.ANSWER),
        started_at=pinned_start,
        ended_at=pinned_end,
    )
    rec.append_step(step)
    trace, _ = rec.finalize(final_answer="ok", final_answer_tokens=1)
    assert trace.started_at == pinned_start
    assert trace.ended_at == pinned_end
