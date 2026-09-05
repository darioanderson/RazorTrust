# ML real-data retraining audit — 2026-09-05

This audit distinguishes an executable model from a model that is eligible for production settlement
decisions. A failed safety gate was not removed or renamed as a pass.

## Verified real data available in this repository

The usable public dataset is OpenML dataset 1597 / ULB-Worldline credit-card fraud: 284,807
transactions, 492 fraud rows, 30 numeric inputs, and a two-day time span. Its local source SHA-256 is
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

This dataset supports transaction-level XGBoost, Isolation Forest, reconstruction autoencoder,
calibration, and drift research. It does not contain merchant identity, device or geography identity,
refund/chargeback lifecycle, settlement holds, evidence outcomes, or entity edges. It therefore cannot
truthfully retrain the serving 13-feature settlement model, graph models, sequence models, or the
three-action conformal policy.

`var/real-history-input.csv` contains only a header and is not a labelled training dataset.
`var/ieee-cis-public-case-2987003.csv` contains one labelled row and is a single UI case, not a valid
training or evaluation sample.

## Blocked-family disposition

| Model family | Existing block | Real ULB compatible | Action in this audit |
|---|---|---:|---|
| XGBoost settlement v3B/v3.2 | Safe RELEASE recall/separability gate | No | Retain block; add strict `--dataset-dir` training path for a future real settlement export |
| IF2 / AE-OOD / R4D3 novelty | Synthetic-only settlement evidence | Partly | Retrain transaction-level XGBoost, IF, and two autoencoders on ULB; keep settlement promotion blocked |
| Unknown-family / family generalization | Synthetic family labels | No | Retain block; ULB has no attack-family or merchant grouping contract |
| GraphSAGE / FDH graph | Graph-statistics ablation failed | No | Retain block; ULB has no entity graph |
| LSTM / temporal sequence | Temporal ablation and seed sensitivity | No | Retain block; two-day row time is not merchant multi-window history |
| Settlement conformal layer | Baseline undercoverage | No | Retain block; binary transaction labels cannot validate the three-action settlement policy |
| Public XGBoost / IF / autoencoder | Research-only | Yes | Fresh chronological real-data retrain and untouched test |

## Fresh ULB result

Run directory: `artifacts/research/public-fraud-retrain-real-ulb-final-20260905`.

- Split: 170,884 chronological train / 56,961 validation / 56,962 untouched test.
- Supervised XGBoost untouched-test PR-AUC: `0.79483579`.
- Isolation Forest test PR-AUC / precision / recall: `0.06186954 / 0.07920792 / 0.21333333`.
- Deep autoencoder test PR-AUC / precision / recall: `0.20015361 / 0.21621622 / 0.53333333`.
- Novelty architecture and override selection used validation only. Supervised validation cost was
  `4347.47`; IF-only, IF+shallow-AE, and IF+deep-AE costs were `8317.91`, `8474.53`, and `8781.96`.
- Final novelty decision: `NO_NOVELTY_OVERRIDE_SELECTED_ON_VALIDATION`.
- The shallow autoencoder converged in 122/200 iterations. The deep autoencoder reached the
  200-iteration limit and is explicitly recorded as non-converged; it is retained for research but not
  selected.
- Production eligibility: false. Automatic RELEASE: false.

The trained XGBoost, StandardScaler, Isolation Forest, and both MLP reconstruction autoencoders are
retained in the run directory. Their SHA-256 values are embedded in `public-fraud-report.json`.

## Real settlement retraining path

`scripts/train_model.py --dataset-dir <directory>` now consumes only a `PUBLIC_REAL` or
`PARTNER_REAL` export satisfying the strict manifest and point-in-time contracts. It validates exact
file hashes plus the aggregate content hash, row counts, unique merchants, and entity references. Raw
records remain in place and are not copied into the model artifact directory.

No conforming real settlement export is present in this repository, so no settlement candidate was
fabricated from ULB or from the one-row UI files. That is the remaining data dependency.
