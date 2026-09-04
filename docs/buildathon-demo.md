# RazorTrust five-minute buildathon demo

## 0:00–0:35 — Open with the control principle

“RazorTrust is not an autonomous payment decision-maker. AI recommends, a named human authorizes, and the system records the proof chain.”

Show the title and architecture slides. State that RELEASE, EVIDENCE_NEEDED, and ESCALATE are recommendations, not fund-movement instructions.

## 0:35–1:15 — Establish research credibility

Show the v3A audit and v3B result slides. The final GPU experiment achieved 3.0% false RELEASE but only 10.0% true RELEASE recall against the locked minimum of 20.0%. Therefore the experiment failed, sealed/stress data stayed untouched, and the stopping decision is `STOP_ML`.

## 1:15–2:35 — Walk through an evidence case

1. Start the local app with `uvicorn razortrust.api:app --reload` in local policy mode.
2. Create a settlement hold.
3. Run the demo evaluation.
4. Point out probabilities, expected cost, decision reason, main contributing signals, payment identity, device/geo evidence, transaction timeline, recommended evidence, and hash-linked audit entries.

## 2:35–3:35 — Demonstrate evidence and human authorization

1. Submit the one permitted evidence package.
2. Resolve the evidence round.
3. Choose the applicable human action: approve RELEASE, request evidence, escalate, or override AI.
4. Enter a mandatory reason and rationale, plus agent/session and transaction attribution.
5. Sign and record the action. Show that the case changes state only after this step.

## 3:35–4:15 — Export the investigation dossier

Open the one-click Evidence PDF. Show the decision summary, uncertainty, signals, document hash, model/feature/policy/cost versions, actor attribution, audit timeline, and explicit human-approval warning.

## 4:15–4:45 — Prove fail-closed behavior

Run `python scripts/run_failure_demo.py`. The simulated model outage must produce `ESCALATE`, move the case to `HUMAN_REVIEW`, and preserve a valid audit chain. A failed artifact signature also fails readiness rather than enabling RELEASE.

## 4:45–5:00 — Close

“The model hit its stopping condition. The product did not stop: RazorTrust makes uncertainty inspectable, decisions attributable, and every operational action human-authorized.”
