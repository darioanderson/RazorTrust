# RazorTrust Synthetic Dataset v3.1 — Longitudinal Research Data

This research generator upgrades the prior scenario-row mechanics dataset into an episode-based longitudinal simulator.

Key properties:

- 14 existing behavior families remain unchanged at the label level.
- Merchant baselines are **estimated from simulated historical transactions**, not assigned directly.
- Baseline interval is `[t-31d, t-1d)` by default; the current decision window is `[t-24h, t)`.
- Merchant heterogeneity, cohort differences, weekday seasonality, growth, measurement noise, partial attacks and benign confounders are simulated.
- Legitimate and risky behavior ranges deliberately overlap so a single feature cannot define the answer.
- Ring families share device/customer identifiers across groups for later graph research.
- The locked 13-feature contract is used unchanged.
- Results are synthetic research only and cannot authorize production financial action.

Recommended staged scale:

1. Smoke: 8 merchants/family = 112 hold windows.
2. Research: 80 merchants/family = 1,120 hold windows.
3. Large: 200 merchants/family = 2,800 hold windows per seed.
4. Multi-seed benchmark: seeds 20260903, 20260917, 20261001; 8,400 hold windows total at 200/family.

The generator streams merchant episodes internally and persists only derived hold-window features plus dataset summaries. This keeps disk use manageable while still forcing the feature engine to derive baselines from raw simulated history.
