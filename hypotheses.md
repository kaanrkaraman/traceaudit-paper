# Pre-registered Hypotheses

Hypotheses are recorded **before** running each phase, dated, and **immutable** thereafter. If a hypothesis turns out wrong, do not edit it — append a post-mortem entry referencing it. The point is to keep ourselves honest about prior expectations.

---

## Phase 0 (2026-05-14 → 2026-05-28) — Trace schema, recorder, replayer, CRAG reproduction

**Locked 2026-05-14** at the schema review. H0.1–H0.3 are now immutable; if any turns out wrong, append a post-mortem entry referencing the relevant H, do not edit the hypothesis itself.

### H0.1 — CRAG on HotpotQA-100 reproduces published numbers within 2pp
- **Outcome variable:** Exact-match (EM) and F1 on HotpotQA dev-100 sub-sample (random_state 42).
- **Direction:** Both within ±2pp of the CRAG paper's reported HotpotQA numbers (which we will pull from the published table at run time and pin into the experiment log — `UNVERIFIED` until pulled).
- **Why:** The brief requires reproduction parity within 2pp as a precondition for audit. If we miss, the paper or our setup has a hidden divergence.
- **Pre-mortem:** If we miss, the most likely causes are (a) different OpenAI model snapshot than the paper used, (b) sub-sample composition variance, (c) prompt-template drift between the paper and the reference implementation.

### H0.2 — Trace replay is byte-identical given seeded inputs and a warm prompt cache
- **Outcome variable:** SHA256 of the serialized `Trace.outputs_hash` field across two consecutive replays of the same captured trace, with a warm SQLite cache and `temperature=0, seed=fixed`.
- **Direction:** Equal across replays for ≥ 99% of HotpotQA-100 traces. The ≤ 1% slack covers any agentic system whose internal randomness escapes our seeding (we'll investigate any failure but tolerate a small tail before Phase 1).
- **Why:** Determinism underpins counterfactual intervention — Mode A's "everything else held fixed" is meaningless without it.
- **Pre-mortem:** Likely failure modes: closed-API non-determinism on cache misses, un-seeded tie-breaking in reranker, dictionary-iteration order in some agent prompt builders.

### H0.3 — Prompt cache hit rate exceeds 95% on the second-pass replay
- **Outcome variable:** SQLite cache hit ratio on the second consecutive replay of the same trace set.
- **Direction:** ≥ 95% hit rate (the ≤ 5% miss budget covers any prompts whose rendering depends on wall-clock or other un-pinned context).
- **Why:** Cache effectiveness directly drives whether we stay under the 2000 USD ceiling.

---

## Phase 1 (2026-05-28 → 2026-06-18) — Interventions, calibration, first CRAG audit

*To be recorded after Phase 0 milestone passes and before any Phase 1 experiment is launched.*

## Phase 2 (2026-06-18 → 2026-07-23) — Multi-system audit

*To be recorded after Phase 1 milestone.*

## Phase 3 (2026-07-23 → 2026-08-13) — Robustness sweeps and optional systems

*To be recorded after Phase 2 milestone.*

## Phase 4 (2026-08-13 onward) — Paper, figures, supplementary, revisions

*To be recorded at phase entry.*
