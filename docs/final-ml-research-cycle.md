# Final ML research cycle

The single authorized post-v3B research cycle ended with
`FINAL_ML_STOP_NO_SAFE_RELEASE_CANDIDATE`.

Before any new model fitting, the signed v3B artifact, dataset hash, and locked partition manifest
were verified. Re-scoring reproduced the prior final development result exactly: 3.0% false
RELEASE and 10.0% true-RELEASE recall.

The final cycle compared 12 bounded GPU candidates: three point-in-time feature sets across four
predeclared XGBoost configurations. Selection used expanding-window temporal out-of-fold predictions
from development only. The primary objective combined partial ROC-AUC at 5% FPR, recall under the
same constraint, and average precision; global AUROC was reported but not optimized alone.

The selected two-stage candidate reached 5.0% false RELEASE and 53.57% recall in temporal OOF
development. On the untouched final development POLICY gate, it reached 7.33% false RELEASE and
43.33% recall. Because 7.33% exceeds the unchanged 5% ceiling, the gate failed. The threshold was
not changed, the candidate was not promoted, and sealed, stress, future-time, unseen-merchant, and
unknown-family sets were not opened.

No Dataset v3.2 corpus exists in the supplied repository. To preserve reproducibility, the cycle
used the unchanged, hash-verified synthetic v3.1 longitudinal corpus signed into v3B and documented
the discrepancy. No replacement data was generated, and no production data was used.

Evidence is under `artifacts/research/final-cycle-v3c-20260903/`; the mandatory pre-training
reproduction is under `artifacts/research/v3b-reproduction-before-final-cycle-20260903/`.
