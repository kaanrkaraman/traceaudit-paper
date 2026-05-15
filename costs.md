# Cost Tracking

Hard ceiling: **2000 USD total** across all commercial LLM APIs. Pause-and-ask threshold: **1500 USD**. GPT-4o call ceiling: **500 calls total**, reserved for the calibrated judge on headline results.

Update after every experiment. Pull token counts from the SQLite prompt cache (`SELECT model, SUM(prompt_tokens), SUM(completion_tokens) FROM cache_entries GROUP BY model;`) plus any provider invoice reconciliation.

## Running totals

| Date | Phase | Experiment | Model | Prompt tok | Completion tok | Cost (USD) | Running total | GPT-4o calls used |
|------|-------|------------|-------|-----------:|---------------:|-----------:|--------------:|------------------:|
| 2026-05-14 | 0 | (project start) | — | 0 | 0 | 0.00 | 0.00 | 0 |

## Provider pricing reference

- GPT-4o (Azure OpenAI): `UNVERIFIED — confirm from Azure portal at project start`
- GPT-4o-mini (Azure OpenAI): `UNVERIFIED — confirm from Azure portal at project start`
- Llama 3.1 70B (Azure-hosted MaaS): `UNVERIFIED — confirm flat compute rate at project start`
- Llama 3.1 8B (Azure-hosted MaaS): `UNVERIFIED — confirm flat compute rate at project start`

These need to be filled in before the first paid experiment. Added to `open_questions.md`? — no, they're operational and resolve on first Azure portal visit, not methodological. Track here.

## Burn-rate guardrails

- After each experiment, compute remaining budget = `2000 − running_total`.
- If a planned experiment's expected cost > `0.25 × remaining_budget`, stop and ask before launching.
- If `running_total ≥ 1500` for any reason, stop and ask before any further paid call.
- GPT-4o call counter must be < 500 at all times. If a planned batch would push it above, stop and ask.

## Cache effectiveness

Track each session:
- Cache hit rate (hits / total requests)
- Tokens served from cache
- Estimated cost avoided (USD)

Target: ≥ 80% hit rate on replay-heavy Mode A sweeps after the first run completes.
