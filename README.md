# TraceAudit

Counterfactual auditing of agentic RAG. Methodological decisions live in `decisions.md`; items awaiting input in `open_questions.md`; pre-registered hypotheses (immutable once recorded) in `hypotheses.md`; running cost ledger in `costs.md`. Schema changelog: `migrations/CHANGELOG.md`. Trace storage layout: `docs/storage_layout.md`.

## Bootstrap

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already present. From the repository root, `uv sync` materializes the locked Python 3.11 virtual environment in `.venv/` from `uv.lock`. Either `source .venv/bin/activate` to enter the venv or prefix commands with `uv run`. Verify the install with `uv run pytest`.
