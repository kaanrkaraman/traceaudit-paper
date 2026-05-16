"""Recorder — captures an agent execution into a `Trace`.

The recorder is the bridge between an audited agent's native control flow and
the immutable `Trace` artifact that downstream code (the replayer, intervention
engine, audit metrics) operates on. It is *deliberately thin*: it does not try
to abstract over the schema, because every system adapter (CRAG, IRCoT,
Self-RAG, FLARE) will know the schema anyway. The recorder centralizes only
what the schema cannot enforce locally:

- top-level trace metadata that must stay consistent across all steps
  (system_id, dataset_id, seed, config, git commit, package versions);
- a chunk registry deduplicated by `content_hash`, with collision detection;
- step-append invariants (contiguous `step_index`, valid `parent_step_id`,
  every referenced chunk previously registered);
- derived fields computed at `finalize()`: `total_cost_usd` summed from
  generation events, `outputs_hash` from final answer + ordered responses,
  `trace_id` and `inputs_hash` derived once at construction.

The caller drives the loop:

    rec = Recorder(system_id="crag", system_version=git_sha, dataset_id="...",
                   query_id="q-001", original_query="...", gold_answers=("..."),
                   seed=42, config={...}, git_commit_sha=our_sha,
                   package_versions={...})

    for chunk in retrieved_chunks:
        rec.register_chunk(chunk)
    rec.append_step(step1)
    # ... more steps ...
    trace, chunks = rec.finalize(final_answer="...", final_answer_tokens=12)

The returned `(trace, chunks)` pair is directly compatible with
`TraceStore.write_trace(trace, chunks)`.

Clock injection
---------------

`started_at` is captured at construction (either from `started_at=` kwarg or
from `clock()`); `ended_at` similarly at `finalize`. Tests pin both to a known
timestamp via the `started_at=` / `ended_at=` kwargs; production callers let
the default `utc_now` clock fire.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Any

from traceaudit.trace.schema import (
    Chunk,
    Step,
    Trace,
    canonical_json,
    compute_config_hash,
    compute_inputs_hash,
    compute_outputs_hash,
    compute_trace_id,
    utc_now,
)


class Recorder:
    """Accumulate steps and chunks for one agent run; finalize into a `Trace`."""

    def __init__(
        self,
        *,
        system_id: str,
        system_version: str,
        dataset_id: str,
        query_id: str,
        original_query: str,
        gold_answers: Iterable[str],
        seed: int,
        config: Mapping[str, Any],
        git_commit_sha: str,
        package_versions: Mapping[str, str],
        started_at: datetime | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.system_id = system_id
        self.system_version = system_version
        self.dataset_id = dataset_id
        self.query_id = query_id
        self.original_query = original_query
        self.gold_answers: tuple[str, ...] = tuple(gold_answers)
        self.seed = seed
        self.git_commit_sha = git_commit_sha
        self._clock = clock

        # Frozen-once derived metadata.
        self.config_json = canonical_json(dict(config))
        self.config_hash = compute_config_hash(dict(config))
        self.package_versions_json = canonical_json(dict(package_versions))
        self.trace_id = compute_trace_id(
            system_id=system_id,
            system_version=system_version,
            dataset_id=dataset_id,
            query_id=query_id,
            seed=seed,
            config_hash=self.config_hash,
        )
        self.inputs_hash = compute_inputs_hash(
            system_id=system_id,
            system_version=system_version,
            dataset_id=dataset_id,
            query_id=query_id,
            seed=seed,
            config_hash=self.config_hash,
            original_query=original_query,
        )

        self.started_at = self._require_tz_aware(
            started_at if started_at is not None else clock(),
            "started_at",
        )

        self._steps: list[Step] = []
        self._chunks: dict[str, Chunk] = {}

    # ---- Public API ----------------------------------------------------------

    def register_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the registry. Idempotent on identical re-registration.

        If a chunk with the same `content_hash` is already registered, this is
        a no-op when the two `Chunk` objects compare equal, and a `ValueError`
        otherwise (hash collision — two different texts producing the same
        hash is cryptographically improbable, but two different `Chunk`
        instances with the same hash but different metadata is a likely real
        bug we want to surface loudly).
        """
        existing = self._chunks.get(chunk.content_hash)
        if existing is None:
            self._chunks[chunk.content_hash] = chunk
            return
        if existing != chunk:
            raise ValueError(
                f"chunk hash collision on {chunk.content_hash[:12]}…: a different "
                f"Chunk instance is already registered for this content_hash; "
                f"this is either a real SHA256 collision (vanishingly unlikely) "
                f"or two Chunk objects disagreeing on source_id / offsets / "
                f"metadata / token_count for identical text — investigate before "
                f"re-running"
            )
        # Idempotent: same chunk, no-op.

    def register_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Bulk-register; identical to a loop over `register_chunk`."""
        for c in chunks:
            self.register_chunk(c)

    def append_step(self, step: Step) -> None:
        """Append a fully-built `Step`. Validates contiguity and references.

        Invariants checked here (in addition to whatever the `Step` model's
        own validators enforce):

        - `step.step_index` equals the current step count (contiguous append).
        - `step.parent_step_id` is `None` only for the root step (step_index=0)
          and otherwise references a previously-appended step.
        - Every `content_hash` named in `step.chunk_appearances` was
          registered via `register_chunk` before this call.
        """
        expected_index = len(self._steps)
        if step.step_index != expected_index:
            raise ValueError(
                f"step.step_index = {step.step_index} but recorder is at "
                f"position {expected_index}; steps must be appended in order"
            )
        if expected_index == 0:
            if step.parent_step_id is not None:
                raise ValueError(
                    f"root step (step_index=0) must have parent_step_id=None; "
                    f"got {step.parent_step_id!r}"
                )
        else:
            if step.parent_step_id is None:
                raise ValueError(
                    f"non-root step (step_index={step.step_index}) must have "
                    f"a parent_step_id; got None"
                )
            known_ids = {s.step_id for s in self._steps}
            if step.parent_step_id not in known_ids:
                raise ValueError(
                    f"step.parent_step_id={step.parent_step_id!r} does not match "
                    f"any previously-appended step; known: {sorted(known_ids)}"
                )
        for app in step.chunk_appearances:
            if app.content_hash not in self._chunks:
                raise ValueError(
                    f"step references chunk {app.content_hash[:12]}… that was not "
                    f"registered with the recorder; call register_chunk(...) before "
                    f"append_step(...)"
                )
        self._steps.append(step)

    @property
    def steps(self) -> tuple[Step, ...]:
        """Snapshot of currently-appended steps."""
        return tuple(self._steps)

    @property
    def chunks(self) -> Mapping[str, Chunk]:
        """Live view of the chunk registry. Read-only by convention."""
        return self._chunks

    def finalize(
        self,
        *,
        final_answer: str,
        final_answer_tokens: int,
        ended_at: datetime | None = None,
    ) -> tuple[Trace, dict[str, Chunk]]:
        """Materialize the captured run into an immutable `Trace`.

        Computes:
        - `total_cost_usd` = sum of `ev.cost_usd` for every generation event.
        - `outputs_hash` from `final_answer` and the ordered list of every
          generation event's `response_text` (in step order, then event order
          within step).

        Returns the `Trace` and a chunks dict trimmed to exactly the
        `content_hash` values referenced by the steps — directly compatible
        with `TraceStore.write_trace(trace, chunks)`.

        Schema validators fire during `Trace` construction; any invariant
        violation raises `pydantic.ValidationError` from this method.
        """
        if not self._steps:
            raise ValueError(
                "cannot finalize a Recorder with no appended steps; call "
                "append_step at least once first"
            )

        ended = self._require_tz_aware(
            ended_at if ended_at is not None else self._clock(),
            "ended_at",
        )
        if ended < self.started_at:
            raise ValueError(
                f"ended_at ({ended.isoformat()}) precedes started_at "
                f"({self.started_at.isoformat()})"
            )

        total_cost = sum(
            ev.cost_usd for step in self._steps for ev in step.generation_events
        )
        generation_responses = [
            ev.response_text for step in self._steps for ev in step.generation_events
        ]
        outputs_hash = compute_outputs_hash(
            final_answer=final_answer,
            generation_responses=generation_responses,
        )

        referenced_hashes = {
            app.content_hash for step in self._steps for app in step.chunk_appearances
        }
        chunks_out = {h: self._chunks[h] for h in referenced_hashes}

        trace = Trace(
            trace_id=self.trace_id,
            system_id=self.system_id,
            system_version=self.system_version,
            dataset_id=self.dataset_id,
            query_id=self.query_id,
            original_query=self.original_query,
            gold_answers=self.gold_answers,
            seed=self.seed,
            config_hash=self.config_hash,
            config_json=self.config_json,
            git_commit_sha=self.git_commit_sha,
            package_versions_json=self.package_versions_json,
            steps=tuple(self._steps),
            final_answer=final_answer,
            final_answer_tokens=final_answer_tokens,
            started_at=self.started_at,
            ended_at=ended,
            total_cost_usd=total_cost,
            inputs_hash=self.inputs_hash,
            outputs_hash=outputs_hash,
        )
        return trace, chunks_out

    # ---- Internals -----------------------------------------------------------

    @staticmethod
    def _require_tz_aware(ts: datetime, name: str) -> datetime:
        if ts.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware (UTC recommended)")
        return ts


__all__ = ["Recorder"]
