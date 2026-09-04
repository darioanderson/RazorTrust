# Competitive ML Validation Hotfix 2B1

Packaging-corrected release of Hotfix 2B.

Benchmark behavior is unchanged from Hotfix 2B:
- bounded OpenML retry/backoff,
- OpenML cache cleanup only between outer retry attempts,
- explicit local `--csv` fallback,
- source transport provenance in the report.

Packaging hardening:
- excludes `__pycache__`, `.pytest_cache`, `*.pyc`, and other generated Python cache artifacts,
- installer refuses a payload containing generated cache artifacts,
- payload hashes are verified before any project file is modified.

Safety invariants remain unchanged:
- champion remains `xgb-if-settlement@2`,
- HUMAN_ONLY remains enabled,
- automatic RELEASE remains disabled,
- no SEALED TEST or stress data is accessed,
- no model promotion or money movement is enabled,
- benchmark methodology and TRAIN / CALIBRATION / POLICY / TEST partitions are unchanged from Hotfix 2A.
