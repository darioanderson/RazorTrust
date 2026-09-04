# Public and partner data validation

RazorTrust now separates two questions that must not be conflated:

1. The ULB/Kaggle benchmark evaluates transaction-level fraud detection on 284,807 anonymized card
   transactions, including 492 frauds. It is real but has no merchant identity, device, geography,
   refund, chargeback, or settlement-hold lifecycle. It therefore tests external fraud/anomaly mechanics
   and business-cost calculations, not the 13-feature merchant settlement model.
2. The settlement model accepts a hash-verified `PUBLIC_REAL` or `PARTNER_REAL` dataset through
   `--dataset-dir`. Its manifest must state the license/agreement and permitted purpose, and its JSONL
   files must satisfy the same point-in-time transaction, merchant-baseline, and adjudicated-hold
   contracts used by the feature builder.

Synthetic settlement training is no longer implicit. It requires the explicit
`--allow-synthetic` flag and remains labelled `SYNTHETIC_MECHANICS_ONLY` in every output report.

## ULB benchmark

Download the original public dataset from
<https://www.kaggle.com/mlg-ulb/creditcardfraud/home>, retain its license metadata, and run:

```powershell
.venv\Scripts\python scripts\evaluate_public_fraud.py creditcardfraud.zip `
  --output artifacts/public-fraud --estimators 300
```

The benchmark uses chronological 60/20/20 partitions. It fits preprocessing only on training rows,
chooses supervised and novelty thresholds on validation rows, and opens the sealed test once. It
reports PR-AUC, precision, recall, fraud amount not stopped, and estimated legitimate settlement-delay
cost. The delay-cost rate is a declared assumption until an operator supplies observed outcome costs.

Two reconstruction architectures are compared against Isolation Forest at the same 99.5th-percentile
legitimate validation threshold. An autoencoder passes only if both its sealed-test precision and recall
meet or exceed Isolation Forest. A passing detector may contribute only an escalation override; it can
never create a release. The report includes PR curves and, for a selected model, a t-SNE plot of its
bottleneck. If no architecture passes, no latent plot is presented as a promoted result.

## Real settlement dataset contract

`--dataset-dir` requires:

- `dataset_manifest.json`
- `merchants.jsonl`
- `transactions.jsonl`
- `holds.jsonl`

The manifest uses `RealSettlementDatasetManifest` and records `data_origin`, agreement/license,
permitted purpose, row counts, and SHA-256 for every input file. Holds must carry adjudicated outcomes;
future outcomes and identities remain excluded from the model matrix. A partner export should be
de-identified before entering this directory and must follow the organization's retention and data
localization policy.

No bundled dataset is redistributed in the source ZIP.
