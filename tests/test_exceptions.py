"""`ReplayCacheMissError` shape — defined per D23 ahead of the replayer.

Verifies that the exception is a proper subclass, carries `prompt_hash` and
`request_payload` attributes, and renders a useful message. The replayer
(step 5c) and prompt cache (step 5d) both import this class, so the
contract is locked here.
"""

from __future__ import annotations

import pytest

from traceaudit.trace.exceptions import ReplayCacheMissError


def test_replay_cache_miss_error_is_a_lookup_error() -> None:
    e = ReplayCacheMissError(prompt_hash="a" * 64, request_payload={"k": "v"})
    assert isinstance(e, LookupError)
    assert isinstance(e, Exception)


def test_replay_cache_miss_error_carries_prompt_hash_and_payload() -> None:
    payload = {
        "model_id": "gpt-4o-mini-2024-07-18",
        "rendered_prompt": "What is Mercury?",
        "temperature": 0.0,
    }
    e = ReplayCacheMissError(prompt_hash="b" * 64, request_payload=payload)
    assert e.prompt_hash == "b" * 64
    assert e.request_payload == payload
    # Message includes the prompt-hash prefix for grep-ability without leaking the full hash.
    assert "b" * 12 in str(e)
    assert "re-record" in str(e) or "rerecord" in str(e)


def test_replay_cache_miss_error_requires_keyword_args() -> None:
    with pytest.raises(TypeError):
        ReplayCacheMissError("a" * 64, {"k": "v"})  # type: ignore[misc]
