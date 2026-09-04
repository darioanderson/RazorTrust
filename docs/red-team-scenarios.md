# Fail-closed red-team scenarios

The invariant is simple: a fault may pause or escalate a settlement, but it must never create an
automatic release. Each scenario below has an automated assertion or an operational verification step.

| Scenario | Injection | Required result | Evidence |
|---|---|---|---|
| Non-finite feature | NaN or positive/negative infinity | Schema rejection before scoring | `tests/test_red_team.py` |
| Invalid ratio | Ratio below zero or above one | Schema rejection before scoring | `tests/test_red_team.py` |
| Feature construction fault | Arithmetic/value/runtime exception | `ESCALATE`, `MODEL_ERROR`, release probability zero | `tests/test_workflow.py` |
| Model unavailable | Missing or corrupt signed release | Readiness fails; evaluation uses fail-closed adapter | API/model tests |
| Model artifact tampering | Modify a signed release file | Hash/signature verification failure before deserialization | `tests/test_modeling.py` |
| OPA unavailable | Refuse connection or return invalid result | `ESCALATE`, `POLICY_ERROR`; critical alert | workflow/API tests |
| Authorization OPA unavailable | Refuse authorization request | HTTP 503, never implicit authorization | API tests |
| Database unavailable before commit | Disconnect during decision transaction | No decision/audit/outbox partial commit; request fails | SQL transaction test/fault-injection drill |
| Audit mutation/deletion | Change a canonical event or `prev_hash` | Chain verification and readiness fail | workflow/audit tests |
| OPA bundle corruption | Alter a signed bundle file | OPA retains current bundle and reports activation failure | signed-bundle release drill |
| Shadow unsafe release | Candidate releases where incumbent did not | Promotion gate fails | `tests/test_shadow_alerts.py` |

Production drills must capture the trace ID, alert ID, policy/model versions, audit head before and after
the injection, database transaction state, and named incident owner. OPA bundle verification should be
tested against the exact production configuration because `opa eval` and `opa test` are policy tests,
not runtime signature-verification substitutes.
