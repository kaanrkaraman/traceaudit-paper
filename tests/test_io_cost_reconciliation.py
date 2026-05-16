"""Cost-sum reconciliation invariant fires on read after a tampered write.

The `Trace` model validator asserts `total_cost_usd == sum(ev.cost_usd for ev
in all_generation_events)`. Round-tripping a valid trace must pass that
validator; tampering with the on-disk `generation_events.parquet` to inflate
one event's `cost_usd` must cause `read_trace` to raise
`pydantic.ValidationError` with a message that mentions `total_cost_usd`.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from traceaudit.trace.io import (
    GENERATION_EVENTS_SCHEMA,
    TraceStore,
)


def test_valid_trace_passes_cost_reconciliation_round_trip(
    tmp_path: Path, synthetic_trace_and_chunks
) -> None:
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)

    store.write_trace(trace, chunks)
    read_back = store.read_trace(
        trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
    )

    summed = sum(ev.cost_usd for s in read_back.steps for ev in s.generation_events)
    assert abs(summed - read_back.total_cost_usd) < 1e-9


def test_tampered_cost_usd_is_caught_by_validator_on_read(
    tmp_path: Path, synthetic_trace_and_chunks
) -> None:
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)
    store.write_trace(trace, chunks)

    # Mutate the first generation event's cost_usd in-place on disk.
    partition = store._partition_dir(trace.system_id, trace.dataset_id)
    gen_file = partition / "generation_events.parquet"
    rows = pq.read_table(gen_file).to_pylist()
    assert rows, "fixture must produce at least one generation event"
    rows[0]["cost_usd"] = rows[0]["cost_usd"] + 999.99
    tampered = pa.Table.from_pylist(rows, schema=GENERATION_EVENTS_SCHEMA)
    pq.write_table(tampered, gen_file)

    with pytest.raises(ValidationError) as exc_info:
        store.read_trace(
            trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
        )
    assert "total_cost_usd" in str(exc_info.value)
