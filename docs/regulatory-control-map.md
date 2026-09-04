# Regulatory control map for review

This is an engineering traceability document, not legal advice or a certification. Legal/compliance
must validate the applicable RBI, PSS Act, KYC/AML/CFT, PCI DSS, privacy, retention, and contractual
requirements for the deploying entity.

RBI's payment-aggregator material covers authorization, merchant onboarding, escrow and settlement,
fraud/risk management, dispute handling, information-system audit, forensic readiness, regulatory
access, cryptography, and data sovereignty. Primary references:

- RBI discussion paper and technology/security controls:
  <https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?ID=943&UrlPage=>
- RBI escrow-account notification:
  <https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=11996>
- RBI cross-jurisdiction payment-aggregator comparison:
  <https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?ID=1214>

| Review question | RazorTrust evidence | Remaining owner |
|---|---|---|
| Why was this settlement held? | Decision ID, plain-language reasons, top attributed features, policy reason, model/policy/cost/schema versions | Compliance approves templates |
| What must the merchant provide? | Versioned evidence checklist, exact deadline, single-round state transition | Operations defines accepted document policy |
| Who made the final decision? | Authenticated analyst ID, timestamp, outcome, reason code, rationale, policy reference | IAM and reviewer training |
| Can a record be changed silently? | RFC 8785 event, per-case sequence, previous hash, record hash, transactional outbox | External immutable checkpoint/store |
| Can a model or policy silently change? | Ed25519 model release verification; signed OPA bundle build/config; version fields in each decision | KMS/HSM/Vault custody and CI signing identity |
| Does a system fault release funds? | Explicit fail-closed adapters and red-team matrix | Production fault-injection drills |
| Are settlements timely? | Trigger, decision, evidence deadline, analyst-decision timestamps | Legal maps product SLAs to current RBI directions |
| Are transaction data localized and minimized? | Hash-only evidence references, de-identified model contract, and no raw transactions in audit or telemetry contracts | Data inventory, localization and retention controls |

Regulator-facing exports should include the case ID, trace ID, timestamps, amount/currency from the
settlement system of record, final action, structured and plain-language reasons, evidence references,
analyst identity/outcome, policy and model versions, cost-matrix version, signature/key IDs, and the
audit-chain hashes. The application does not invent missing amount, escrow, or legal-basis fields; those
must come from the payment platform's authoritative ledger.
