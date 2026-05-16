# Open Questions

Append-only log of decisions needing human input. Each entry: date opened, context, alternatives considered, current recommendation, status.

When a question is resolved, **do not delete it** — change `Status:` to `Resolved (YYYY-MM-DD)` and link to the corresponding entry in `decisions.md`.

---

## Q1 — Answer-change threshold for URR
- **Opened:** 2026-05-14
- **Phase:** 0 (must resolve before Phase 1 calibration)
- **Context:** URR is defined as the fraction of generation-useful chunks in a trace. We need a precise definition of "generation-useful" — the trigger that flips a chunk from useless to useful when its absence is compared against its presence on the final answer.
- **Alternatives considered:**
  1. F1 drop > 5pp (between answer-with-chunk and answer-without-chunk, against gold answers). Continuous, well-understood, but tied to lexical overlap and may miss paraphrastic equivalence.
  2. Strict exact-match flip (EM before → not-EM after, or vice versa). Discrete and crisp but very coarse — many real shifts will be invisible.
  3. GPT-4o binary semantic-equivalence judge. Captures paraphrase but expensive and unverified — must be calibrated against human labels.
  4. Hybrid: report strict-EM URR and judge-URR as twin headline numbers, F1-drop as a continuous companion.
- **Recommendation:** Hybrid (option 4). F1 drop > 5pp as the primary continuous signal for sensitivity sweeps; strict-EM URR and judge-URR as twin headline numbers in the main table. Judge-URR contingent on calibration ≥ 85% accuracy on 100 manually labeled pairs (per brief).
- **Status:** Resolved 2026-05-14 — see `decisions.md` D18.

## Q2 — Paraphrase-model identity (avoiding generator confound)
- **Opened:** 2026-05-14
- **Phase:** 0 → 1
- **Context:** Mode A/B intervention operator "paraphrase" rewrites a chunk while preserving semantics. If the paraphraser and the audited generator share a model family, paraphrastic style alone may correlate with generator behavior — a confound.
- **Alternatives considered:**
  1. Always paraphrase with Llama (different family from GPT-based generators).
  2. Always paraphrase with GPT-4o-mini and accept the confound.
  3. Cross-family rule: paraphrase model family must differ from generator family; document the pairing per experiment.
- **Recommendation:** Option 3 (cross-family rule). For audits of GPT-based agents, paraphrase with Llama 3.1 70B; for the GPT-4o-mini → Llama 3.1 70B generator-robustness swap in Phase 3, paraphrase with GPT-4o-mini. Pairing recorded in the trace config.
- **Status:** Open

## Q3 — Distractor similarity band
- **Opened:** 2026-05-14
- **Phase:** 0 → 1
- **Context:** The "distract" operator replaces a chunk with a plausibly retrievable but factually wrong chunk. Too similar and we don't know if we're testing semantics or surface form; too dissimilar and the retriever wouldn't have surfaced it.
- **Alternatives considered:**
  1. Cosine similarity in [0.6, 0.85] against BGE-large-en-v1.5 embeddings (brief's recommendation).
  2. Wider band [0.5, 0.9] for more candidate diversity.
  3. Adaptive band tied to the original chunk's similarity to its top-retrieved neighbors (whatever the retriever returned for that step, the distractor should fall in the same shell).
- **Recommendation:** Option 1 as primary, with option 3 as a robustness check on a sub-sample. Reject any generated distractor outside the band and re-sample (cap 5 retries before flagging the chunk as un-distractable and skipping).
- **Status:** Open

## Q4 — Mode B rollout count and aggregation
- **Opened:** 2026-05-14
- **Phase:** 1 → 2
- **Context:** Mode B (counterfactual rollout) lets the agent freely re-decide downstream. The agent is stochastic (temperature > 0 in most baselines), so a single rollout is noisy.
- **Alternatives considered:**
  1. 1 rollout, temperature 0. Cheap but loses the trajectory-sensitivity signal we care about.
  2. 3 rollouts, report median and IQR (brief's recommendation).
  3. 5 rollouts, report mean ± bootstrap CI. More statistically clean but ~67% more expensive.
- **Recommendation:** Option 2 (3 rollouts, median + IQR). Promote to 5 only for the headline 2×2 system×dataset cells in Phase 2 if budget permits — decide at the Phase 1 milestone.
- **Status:** Open

## Q5 — Mode B early-stop handling
- **Opened:** 2026-05-14
- **Phase:** 1 → 2
- **Context:** In Mode B the agent may stop earlier (or never reach the original step count) after the intervention. Is an early stop a valid trajectory or a failure mode?
- **Alternatives considered:**
  1. Force agent to take ≥ original number of steps. Violates "free re-decision" premise.
  2. Allow early stop, treat the truncated trajectory as the answer, report it normally.
  3. Allow early stop but tag it as a *trajectory-change failure mode* in a separate column of the results table (brief's recommendation).
- **Recommendation:** Option 3. Report early-stop rate alongside answer-flip rate. An agent that stops early after losing a chunk is exhibiting evidence sensitivity, which is itself informative.
- **Status:** Open

## Q6 — Storage layout for traces
- **Opened:** 2026-05-14
- **Phase:** 0
- **Context:** Parquet via PyArrow is fixed by the brief. The choice is between one nested Parquet file per trace vs normalized tables joined on stable IDs. See `docs/storage_layout.md` for full proposal.
- **Alternatives considered:**
  1. One file per trace, nested (List/Struct columns). Self-contained, easy to ship, but heavy duplication of chunk text across traces and awkward set-level statistics.
  2. Normalized tables: `traces`, `steps`, `generation_events`, `retrieval_events`, `chunks` (deduped), `chunk_appearances`. Joins are cheap with PyArrow datasets; intervention sweeps benefit from chunk deduplication.
  3. Hybrid: normalized for shared corpora, single nested file per trace as a portable export format.
- **Recommendation:** Option 2 (normalized) as the working store. Option 3's nested export deferred until paper-supplementary stage.
- **Status:** Resolved 2026-05-14 — see `decisions.md` D19.

## Q7 — Trace ID generation
- **Opened:** 2026-05-14
- **Phase:** 0
- **Context:** Stable identifiers matter for replay and joins.
- **Alternatives considered:**
  1. UUIDv4. Unique but non-deterministic; two replays of the same inputs get different IDs.
  2. Deterministic hash of (system_id, system_version, dataset_id, query_id, seed, config_hash). Same inputs → same ID, which lets us spot accidental duplicates and re-runs.
- **Recommendation:** Option 2 (deterministic). UUIDv4 only as fallback if the deterministic hash inputs are incomplete.
- **Status:** Open

## Q8 — Tree-structured trace support
- **Opened:** 2026-05-14
- **Phase:** 0
- **Context:** Most current target systems (CRAG, IRCoT, Self-RAG, FLARE) produce a linear sequence of steps. MA-RAG (optional Phase 3) fans out across sub-agents. Should the schema natively support trees via `parent_step_id`, or stay linear and refactor later?
- **Alternatives considered:**
  1. Linear only, refactor when MA-RAG arrives.
  2. Tree-capable from day one via optional `parent_step_id`. Linear traces are a degenerate tree (each step's parent is the previous one).
- **Recommendation:** Option 2. Cost is one nullable field; the refactor cost in Phase 3 is much higher.
- **Status:** Open

## Q9 — Project start date discrepancy
- **Opened:** 2026-05-14
- **Phase:** 0
- **Context:** Your message header says "Today is May 14 2026"; the harness context says today is 2026-05-15. Phase 0 spans "weeks 1 to 2 May 14 to May 28" so the intended Day 0 is 2026-05-14.
- **Recommendation:** Treat 2026-05-14 as Project Day 0 across all tracking files; ignore the one-day harness drift unless it matters for a deadline calculation.
- **Status:** Resolved 2026-05-14 — see `decisions.md` D20.

## Q10 — Determinism guarantees and the cache
- **Opened:** 2026-05-14
- **Phase:** 0
- **Context:** "Determinism tests in CI" + "re-run a trace and assert byte-identical outputs given seeded inputs." Closed-API LLMs (GPT-4o, GPT-4o-mini) are not bit-deterministic even at temperature 0 / fixed seed — they drift across server-side updates. The SQLite prompt cache hides this for any prompt we have seen before, but a fresh prompt's response is non-reproducible at the byte level.
- **Alternatives considered:**
  1. Determinism only via the cache: replay = "every prompt is a cache hit." If a prompt is missing, the determinism test fails loudly. This is honest but means we cannot determinism-test on a cold cache.
  2. Local-only determinism tests using a small open model (e.g., a frozen-weights Llama 3.1 8B in vLLM with fixed seed). Real bit-determinism, smaller scope.
  3. Hybrid: cache-based determinism for closed-model replay paths; vLLM-based determinism for the swap-generator robustness path.
- **Recommendation:** Option 3 (hybrid). Cache is the contract for closed models; vLLM for the open path.
- **Status:** Resolved 2026-05-14 — see `decisions.md` D21.

## Q11 — Adjacent CIKM '26 triple-robustness paper (anonymous submission)
- **Opened:** 2026-05-14
- **Phase:** spans 1, 2, 4 (cross-phase tracking; sub-parts close independently at their respective phases)
- **Context:** A friend shared an anonymous CIKM 2026 submission titled "Universal Pathologies, Conditional Consequences: A Triple-Robustness Analysis of RAG for Multi-Hop Traceability." Methodologically orthogonal to TraceAudit — they vary embedder × corpus × judge while holding the architecture matrix fixed; we hold the architecture fixed and intervene on individual chunks. Different headline metric (ALCE citation P/R/F1 + multi-judge κ vs URR), different object of study (their five pipelines vs published agentic systems CRAG/IRCoT/Self-RAG/FLARE). Adjacent enough to cite, not adjacent enough to redirect.

### Q11.a — Judge fragility risk to URR-judge headline
Their C3 finding (GPT-5.4 same-judge self-κ = 0.137 across embedder swap on identical items, 41% verdict flips) is a direct threat to our GPT-4o judge contract (D02, D05). Mitigation to evaluate in Phase 2: pair GPT-4o with a Llama-3.1-70B-as-judge on the URR headline subset and report inter-judge agreement alongside the headline number. If agreement is poor, drop judge-URR from the headline and rely on F1-URR and EM-URR only. Calibration set work in Phase 1 should include a paired-judge sanity check before locking the judge contract.

### Q11.b — MuSiQue subsample alignment
We use `random_state=42` over MuSiQue dev (D08). They use a 200-query 2/3/4-hop stratified subset over Wikipedia paragraph chains with only `REFERENCES` edges. Once their paper hits arXiv (currently anonymous CIKM 2026 submission), revisit whether to: (a) keep our disjoint slice, (b) align to their slice for direct comparability on the 1-hop / 2-hop / 3+-hop strata they define, or (c) report both. No action today; decision deferred until their preprint is citable.

### Q11.c — Framing paragraph for related work
Once their paper is publicly citable, the related-work section needs an explicit positioning paragraph distinguishing axis-variation (their lever) from chunk-intervention (our lever). Their C2a (over-citation universal) and C2b (faithfulness consequence corpus-conditional) findings are consistent with — and provide empirical motivation for — our causal question of whether over-cited evidence is actually necessary. Save the proposed wording for Phase 4 draft.

- **Status:** Open (cross-phase tracking entry; sub-parts close independently at their respective phases)
