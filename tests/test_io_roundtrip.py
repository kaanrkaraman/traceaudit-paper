"""Six-table write/read round-trip.

Asserts that a synthetic Trace written through `TraceStore.write_trace` and
read back through `TraceStore.read_trace` is byte-identical at every level
the schema considers identity-bearing: `outputs_hash`, `inputs_hash`, every
chunk's `content_hash`, every generation event's `prompt_hash`, and every
tool call's `call_id`.
"""

from __future__ import annotations

from pathlib import Path

from traceaudit.trace.io import TraceStore


def test_roundtrip_preserves_all_hashes(
    tmp_path: Path, synthetic_trace_and_chunks
) -> None:
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)

    store.write_trace(trace, chunks)
    read_back = store.read_trace(
        trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
    )

    # Trace-level identity.
    assert read_back.trace_id == trace.trace_id
    assert read_back.inputs_hash == trace.inputs_hash
    assert read_back.outputs_hash == trace.outputs_hash
    assert read_back.schema_version == trace.schema_version

    # Step count and structure preserved.
    assert len(read_back.steps) == len(trace.steps)
    for orig_step, new_step in zip(trace.steps, read_back.steps):
        assert new_step.step_id == orig_step.step_id
        assert new_step.parent_step_id == orig_step.parent_step_id
        assert new_step.step_index == orig_step.step_index
        assert new_step.intent == orig_step.intent
        assert new_step.decision.decision_type == orig_step.decision.decision_type
        assert new_step.decision.raw_signal == orig_step.decision.raw_signal
        assert len(new_step.generation_events) == len(orig_step.generation_events)
        assert len(new_step.chunk_appearances) == len(orig_step.chunk_appearances)

    # Every generation event's prompt_hash roundtrips, and tool_calls survive.
    orig_prompt_hashes = [
        ev.prompt_hash for s in trace.steps for ev in s.generation_events
    ]
    new_prompt_hashes = [
        ev.prompt_hash for s in read_back.steps for ev in s.generation_events
    ]
    assert new_prompt_hashes == orig_prompt_hashes

    orig_tool_call_ids = [
        tc.call_id
        for s in trace.steps
        for ev in s.generation_events
        if ev.tool_calls is not None
        for tc in ev.tool_calls
    ]
    new_tool_call_ids = [
        tc.call_id
        for s in read_back.steps
        for ev in s.generation_events
        if ev.tool_calls is not None
        for tc in ev.tool_calls
    ]
    assert new_tool_call_ids == orig_tool_call_ids
    assert len(new_tool_call_ids) > 0, "fixture must exercise at least one tool call"

    # Every chunk's content_hash roundtrips through the global chunks table.
    referenced = {app.content_hash for s in trace.steps for app in s.chunk_appearances}
    read_chunks = store.read_chunks(referenced)
    assert set(read_chunks.keys()) == referenced
    for h, original in chunks.items():
        if h in referenced:
            ch = read_chunks[h]
            assert ch.content_hash == h
            assert ch.text == original.text
            assert ch.source_id == original.source_id
            assert ch.token_count == original.token_count


def test_roundtrip_rejects_chunks_dict_with_missing_or_extra_hashes(
    tmp_path: Path, synthetic_trace_and_chunks
) -> None:
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)

    # Missing
    incomplete = dict(list(chunks.items())[:-1])
    try:
        store.write_trace(trace, incomplete)
    except ValueError as e:
        assert "missing" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing chunks")

    # Extra
    from traceaudit.trace.schema import Chunk, compute_content_hash

    bogus_text = "not referenced by any step"
    extras = {
        compute_content_hash(bogus_text): Chunk(
            content_hash=compute_content_hash(bogus_text),
            text=bogus_text,
            source_id="wiki:Bogus",
        )
    }
    too_many = chunks | extras
    try:
        store.write_trace(trace, too_many)
    except ValueError as e:
        assert "unreferenced" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for extra chunks")
