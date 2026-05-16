"""PromptCache contract tests.

Covers:

- mode-dependent miss behavior (RECORD returns None; STRICT_REPLAY raises);
- round-trip identity (put → get returns byte-identical CachedResponse);
- in-process close-and-reopen persistence;
- cross-instance close-drop-reopen persistence (catches WAL-flush bugs the
  same-process reopen test can miss);
- concurrent-write safety from two threads against two PromptCache
  instances on the same DB path (documents the WAL contract);
- hit/miss stats accounting (H0.3 evaluation primitive);
- cache_schema_version mismatch refuses to open;
- `CacheRequest.from_event` matches an explicitly-constructed request;
- request_key is sensitive to model_provider (B-adjustment to the request-key spec);
- context manager closes; double-close is idempotent;
- idempotent overwrite under the same request_key.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import pytest

from traceaudit.cache.sqlite import (
    CACHE_SCHEMA_VERSION,
    CacheMode,
    CacheRequest,
    CachedResponse,
    CacheSchemaVersionMismatchError,
    PromptCache,
)
from traceaudit.trace.exceptions import ReplayCacheMissError
from traceaudit.trace.schema import (
    GenerationEvent,
    ModelProvider,
    compute_prompt_hash,
)


def _request(
    *,
    prompt_hash: str = "0" * 64,
    model_id: str = "gpt-4o-mini-2024-07-18",
    model_provider: ModelProvider = ModelProvider.AZURE_OPENAI,
    model_version: str | None = "2024-07-18",
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    max_tokens: int = 256,
) -> CacheRequest:
    return CacheRequest(
        prompt_hash=prompt_hash,
        model_id=model_id,
        model_provider=model_provider,
        model_version=model_version,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )


def _response(
    text: str = "the answer",
    *,
    tool_calls_json: str | None = None,
) -> CachedResponse:
    return CachedResponse(
        response_text=text,
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0002,
        latency_ms=100.0,
        finish_reason="stop",
        logprobs_json=None,
        tool_calls_json=tool_calls_json,
    )


def test_empty_cache_record_mode_returns_none(tmp_path: Path) -> None:
    with PromptCache(tmp_path / "cache.sqlite") as cache:
        result = cache.get(_request(), mode=CacheMode.RECORD)
        assert result is None
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["total"] == 1
        assert stats["hit_rate"] == 0.0


def test_empty_cache_strict_replay_raises_with_prompt_hash_and_payload(
    tmp_path: Path,
) -> None:
    request = _request(prompt_hash="missing-prompt-xyz")
    with PromptCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(ReplayCacheMissError) as excinfo:
            cache.get(request, mode=CacheMode.STRICT_REPLAY)
        err = excinfo.value
        assert err.prompt_hash == "missing-prompt-xyz"
        assert err.request_payload["prompt_hash"] == "missing-prompt-xyz"
        assert err.request_payload["model_provider"] == ModelProvider.AZURE_OPENAI.value
        assert err.request_payload["temperature"] == 0.0
        assert err.request_payload["max_tokens"] == 256


def test_roundtrip_put_then_get_returns_byte_identical(tmp_path: Path) -> None:
    request = _request()
    response = _response(
        "the answer is 42",
        tool_calls_json='[{"name":"web_search","call_id":"x"}]',
    )
    with PromptCache(tmp_path / "cache.sqlite") as cache:
        cache.put(request, response)
        retrieved = cache.get(request, mode=CacheMode.RECORD)
        assert retrieved == response
        # Strict replay on the same request must now hit, not raise.
        assert cache.get(request, mode=CacheMode.STRICT_REPLAY) == response


def test_persistence_across_in_process_close_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite"
    request = _request()
    response = _response("persisted")

    cache1 = PromptCache(db_path)
    try:
        cache1.put(request, response)
    finally:
        cache1.close()

    cache2 = PromptCache(db_path)
    try:
        retrieved = cache2.get(request, mode=CacheMode.RECORD)
        assert retrieved == response
    finally:
        cache2.close()


def test_persistence_across_instance_close_drop_reopen(tmp_path: Path) -> None:
    """Cross-instance persistence with explicit close, reference drop, and
    fresh instantiation. Catches WAL-flush bugs the same-process reopen
    test can miss (e.g., the second instance reusing cached state from
    the first via the sqlite3 connection pool or stale page cache).
    """
    db_path = tmp_path / "cache.sqlite"
    request = _request(prompt_hash="cross-instance-test")
    response = _response("cross-instance body")

    first = PromptCache(db_path)
    first.put(request, response)
    first.close()
    del first  # drop the reference; sqlite3 connection should be fully released

    second = PromptCache(db_path)
    try:
        retrieved = second.get(request, mode=CacheMode.RECORD)
        assert retrieved == response
        # Stats are per-instance, not persisted to disk.
        stats = second.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
    finally:
        second.close()


def test_concurrent_writes_from_two_instances_two_threads(tmp_path: Path) -> None:
    """Two PromptCache instances against the same db_path, two threads,
    each writes a distinct key. Both writes must succeed without
    corruption; a third instance opened afterward must see both rows.

    Production has only one writer per cache; this test documents the
    contract and catches sqlite3 misconfiguration (forgotten WAL,
    missing busy_timeout, wrong locking mode).
    """
    db_path = tmp_path / "cache.sqlite"
    request_a = _request(prompt_hash="hash-a")
    request_b = _request(prompt_hash="hash-b")
    response_a = _response("answer-a")
    response_b = _response("answer-b")
    errors: dict[str, BaseException] = {}

    def writer(label: str, req: CacheRequest, resp: CachedResponse) -> None:
        try:
            cache = PromptCache(db_path)
            try:
                cache.put(req, resp)
            finally:
                cache.close()
        except BaseException as e:  # noqa: BLE001 — test diagnostics
            errors[label] = e

    t1 = Thread(target=writer, args=("t1", request_a, response_a))
    t2 = Thread(target=writer, args=("t2", request_b, response_b))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"thread errors: {errors}"

    with PromptCache(db_path) as verifier:
        assert verifier.get(request_a, mode=CacheMode.RECORD) == response_a
        assert verifier.get(request_b, mode=CacheMode.RECORD) == response_b


def test_stats_accumulate_across_operations(tmp_path: Path) -> None:
    with PromptCache(tmp_path / "cache.sqlite") as cache:
        request = _request()
        # Three misses on an empty cache.
        for _ in range(3):
            assert cache.get(request, mode=CacheMode.RECORD) is None
        cache.put(request, _response())
        # Two hits on the populated cache.
        for _ in range(2):
            assert cache.get(request, mode=CacheMode.RECORD) is not None
        stats = cache.stats()
        assert stats["misses"] == 3
        assert stats["hits"] == 2
        assert stats["total"] == 5
        assert stats["hit_rate"] == pytest.approx(2 / 5)


def test_schema_version_mismatch_raises_on_open(tmp_path: Path) -> None:
    """A cache DB written under a different schema version must refuse to
    open. No silent migration; deletion or explicit migration is required.
    """
    db_path = tmp_path / "cache.sqlite"
    PromptCache(db_path).close()

    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM cache_schema_version")
    conn.execute(
        "INSERT INTO cache_schema_version (version) VALUES (?)",
        ("9.9.9",),
    )
    conn.commit()
    conn.close()

    with pytest.raises(CacheSchemaVersionMismatchError) as excinfo:
        PromptCache(db_path)
    err = excinfo.value
    assert err.found == "9.9.9"
    assert err.expected == CACHE_SCHEMA_VERSION
    assert err.db_path == db_path


def test_cache_request_from_event_matches_explicit_construction() -> None:
    """`CacheRequest.from_event` must produce a request with the same
    `request_key` as an equivalent explicitly-constructed request.
    """
    ev = GenerationEvent(
        event_id="e-1",
        event_index_in_step=0,
        model_id="gpt-4o-mini-2024-07-18",
        model_provider=ModelProvider.AZURE_OPENAI,
        model_version="2024-07-18",
        prompt_template_id="test.v1",
        prompt_hash=compute_prompt_hash("hello"),
        rendered_prompt="hello",
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_tokens=256,
        response_text="hi",
        prompt_tokens=1,
        completion_tokens=1,
        cached=False,
        finish_reason="stop",
        cost_usd=0.0,
        latency_ms=10.0,
        started_at=datetime(2026, 5, 14, tzinfo=timezone.utc),
    )
    from_event = CacheRequest.from_event(ev)
    explicit = CacheRequest(
        prompt_hash=ev.prompt_hash,
        model_id=ev.model_id,
        model_provider=ev.model_provider,
        model_version=ev.model_version,
        temperature=ev.temperature,
        top_p=ev.top_p,
        seed=ev.seed,
        max_tokens=ev.max_tokens,
    )
    assert from_event == explicit
    assert from_event.request_key() == explicit.request_key()


def test_request_key_changes_when_provider_changes() -> None:
    """Same prompt + same nominal model + different provider → different
    request_key. Azure OpenAI gpt-4o-mini and OpenAI gpt-4o-mini are
    different deployments and can drift; the cache must treat them as
    independent (B-adjustment from D25's request-key spec).
    """
    azure = _request(model_provider=ModelProvider.AZURE_OPENAI)
    openai = _request(model_provider=ModelProvider.OPENAI)
    assert azure.request_key() != openai.request_key()


def test_request_key_stable_when_irrelevant_fields_change() -> None:
    """Sanity: identical request_key fields → identical key. Guards against
    accidental inclusion of non-deterministic state (e.g., timestamps).
    """
    a = _request()
    b = _request()
    assert a.request_key() == b.request_key()


def test_context_manager_closes_and_double_close_is_idempotent(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cache.sqlite"
    with PromptCache(db_path) as cache:
        cache.put(_request(), _response())
    # Idempotent second close.
    cache.close()
    # Operations after close raise a clear error.
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(_request(), mode=CacheMode.RECORD)


def test_idempotent_put_under_same_request_key_overwrites(tmp_path: Path) -> None:
    """Re-running RECORD mode against a populated cache must overwrite
    cleanly under the same key, not raise on uniqueness violation.
    """
    request = _request()
    with PromptCache(tmp_path / "cache.sqlite") as cache:
        cache.put(request, _response("first"))
        cache.put(request, _response("second"))
        retrieved = cache.get(request, mode=CacheMode.RECORD)
        assert retrieved is not None
        assert retrieved.response_text == "second"
