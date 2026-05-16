"""Trace-replay error types.

Per **D23** in `decisions.md`: cache misses in strict-replay mode surface as
`ReplayCacheMissError` — a custom exception, never a sentinel. The replayer
must not catch this in its main path; a miss propagates as a hard replay
failure with the missing `prompt_hash` and the full request payload attached
so the caller can decide whether to re-record or abort.

This module lives next to `io.py` so the error type is available before the
prompt-cache implementation lands (step 5d). Downstream consumers (replayer,
cache wrapper) import from here.
"""

from __future__ import annotations

from typing import Any


class ReplayCacheMissError(LookupError):
    """Raised when the replayer's strict-mode prompt cache misses a hash.

    The miss is treated as a replay failure rather than silently re-querying
    the provider, because the byte-determinism contract for closed models
    (D21) is broken once we re-query. Re-recording — not re-querying —
    is the only way to recover deterministically.

    `LookupError` is the parent because a cache miss is, semantically, a
    failed lookup; users catching `LookupError` in adjacent code will also
    catch this naturally.

    Attributes:
        prompt_hash: SHA256 hex of the rendered prompt that missed the cache.
        request_payload: The full request dict (model_id, rendered_prompt,
            temperature, top_p, seed, max_tokens, …) — captured so the caller
            can decide whether to re-record or abort the replay.
    """

    def __init__(self, *, prompt_hash: str, request_payload: dict[str, Any]) -> None:
        self.prompt_hash = prompt_hash
        self.request_payload = request_payload
        super().__init__(
            f"cache miss on replay: prompt_hash={prompt_hash[:12]}…; "
            "re-record the trace or disable strict-replay mode to proceed"
        )


__all__ = ["ReplayCacheMissError"]
