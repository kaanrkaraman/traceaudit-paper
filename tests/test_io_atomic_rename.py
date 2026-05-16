"""Atomic-rename-per-partition guarantee.

Simulates a crash by monkey-patching `os.replace` (the rename primitive
`io._append_and_rename` uses) to raise mid-write. The contract under test:

1. A first-write failure leaves the partition with no readable `traces.parquet`,
   so `read_trace` raises a clean `FileNotFoundError` — never a partial read.
2. A second-write failure preserves the previously-committed trace exactly;
   the partial may exist on disk but is not visible to readers, and the
   never-committed trace is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import traceaudit.trace.io as io_module
from traceaudit.trace.io import TraceStore


def test_first_write_failure_is_invisible_to_readers(
    tmp_path: Path, synthetic_trace_and_chunks, monkeypatch
) -> None:
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)

    def boom(_src, _dst):
        raise OSError("simulated mid-rename crash")

    monkeypatch.setattr(io_module.os, "replace", boom)

    with pytest.raises(OSError, match="simulated"):
        store.write_trace(trace, chunks)

    # No partition committed -> clean FileNotFoundError on read.
    with pytest.raises(FileNotFoundError):
        store.read_trace(
            trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
        )


def test_second_write_failure_preserves_committed_first_trace(
    tmp_path: Path,
    synthetic_trace_and_chunks,
    second_synthetic_trace_and_chunks,
    monkeypatch,
) -> None:
    trace1, chunks1 = synthetic_trace_and_chunks
    trace2, chunks2 = second_synthetic_trace_and_chunks
    store = TraceStore(tmp_path)

    # First trace commits successfully.
    store.write_trace(trace1, chunks1)

    # On the next write, fail the very first rename so trace2 never lands.
    real_replace = io_module.os.replace

    def fail_on_next_rename(src, dst):
        raise OSError("simulated mid-rename crash on second write")

    monkeypatch.setattr(io_module.os, "replace", fail_on_next_rename)
    with pytest.raises(OSError, match="simulated"):
        store.write_trace(trace2, chunks2)
    monkeypatch.setattr(io_module.os, "replace", real_replace)

    # Trace 1 still readable and byte-identical.
    read_back = store.read_trace(
        trace1.trace_id, system_id=trace1.system_id, dataset_id=trace1.dataset_id
    )
    assert read_back.trace_id == trace1.trace_id
    assert read_back.outputs_hash == trace1.outputs_hash
    assert read_back.inputs_hash == trace1.inputs_hash

    # Trace 2 is absent — readers see a clean KeyError, not partial data.
    with pytest.raises(KeyError):
        store.read_trace(
            trace2.trace_id, system_id=trace2.system_id, dataset_id=trace2.dataset_id
        )

    # Sanity: a `.partial` file may exist from the aborted rename, but the
    # store's reader never globs for it — only `<file>.parquet` is opened.
    # We assert directory state directly: every committed file ends in
    # `.parquet` (and belongs to trace1, same partition as trace2), and any
    # `.partial` orphans from the aborted second write are present but
    # ignored by the reader.
    partition = store._partition_dir(trace2.system_id, trace2.dataset_id)
    visible_parquets = list(partition.glob("*.parquet"))
    assert visible_parquets, "trace1 should have left at least traces.parquet"
    partials = list(partition.glob("*.partial"))
    assert partials, (
        "expected at least one orphan .partial from the aborted second write; "
        "if none is present the test is no longer exercising the coexistence case"
    )
    for f in partials:
        assert f.name.endswith(".parquet.partial"), (
            f"unexpected orphan file name {f.name!r}; only `<table>.parquet.partial` "
            "is produced by _append_and_rename"
        )


def test_stale_partial_on_cold_store_is_invisible(tmp_path: Path) -> None:
    """A `.partial` orphaned by a crashed first-write must not be opened.

    The reader resolves `<file>.parquet` by exact name (never globs). A
    sibling `<file>.parquet.partial` containing garbage bytes is left on
    disk, and `read_trace` must raise `FileNotFoundError` from the missing
    committed `traces.parquet`, never attempt to parse the partial.
    """
    store = TraceStore(tmp_path)
    system_id, dataset_id = "test-system", "test-dataset"
    partition = store._partition_dir(system_id, dataset_id)
    partition.mkdir(parents=True, exist_ok=True)

    stale = partition / "traces.parquet.partial"
    stale.write_bytes(b"this is not a valid parquet file")

    with pytest.raises(FileNotFoundError):
        store.read_trace(
            "any-trace-id", system_id=system_id, dataset_id=dataset_id
        )

    # Reader did not delete or rename the orphan; cleanup is out of scope.
    assert stale.exists()


def test_committed_parquet_with_stale_partial_returns_committed_state(
    tmp_path: Path, synthetic_trace_and_chunks
) -> None:
    """Coexisting `<file>.parquet` and `<file>.parquet.partial`: reader wins committed.

    Simulates the steady state after a crashed second-write: every table in
    the partition has both the committed `.parquet` and an orphan `.partial`
    next to it. The roundtrip must succeed against the committed files; the
    partials are inert.
    """
    trace, chunks = synthetic_trace_and_chunks
    store = TraceStore(tmp_path)
    store.write_trace(trace, chunks)

    partition = store._partition_dir(trace.system_id, trace.dataset_id)
    table_files = [
        "traces.parquet",
        "steps.parquet",
        "generation_events.parquet",
        "retrieval_events.parquet",
        "chunk_appearances.parquet",
    ]
    for name in table_files:
        committed = partition / name
        assert committed.exists(), f"setup precondition failed: {committed} missing"
        stale = partition / (name + ".partial")
        stale.write_bytes(b"orphan bytes from a previous failed write")

    read_back = store.read_trace(
        trace.trace_id, system_id=trace.system_id, dataset_id=trace.dataset_id
    )
    assert read_back.trace_id == trace.trace_id
    assert read_back.outputs_hash == trace.outputs_hash
    assert read_back.inputs_hash == trace.inputs_hash
    assert len(read_back.steps) == len(trace.steps)

    # All orphan .partial files still on disk and untouched by the reader.
    for name in table_files:
        assert (partition / (name + ".partial")).exists()
