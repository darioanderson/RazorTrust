package razortrust.holds

import rego.v1

health := true
policy_version := "hold-policy@1"

default decision := {
	"allowed_decision": "ESCALATE",
	"guardrail_triggered": true,
	"reasons": ["DEFAULT_ESCALATE"],
	"policy_version": "hold-policy@1",
}

decision := escalation("SYSTEM_ERROR") if input.system_error

decision := escalation("NOVELTY_OVERRIDE") if input.novelty_override

decision := escalation("INVALID_EVIDENCE_ROUND") if input.evidence_round > 1

decision := {
	"allowed_decision": "RELEASE",
	"guardrail_triggered": false,
	"reasons": ["RELEASE_GUARDRAILS_PASSED"],
	"policy_version": policy_version,
} if {
	not input.system_error
	not input.novelty_override
	input.evidence_round <= 1
	input.candidate_decision == "RELEASE"
	input.evidence_round == 0
	input.probabilities.release >= input.thresholds.release
}

decision := {
	"allowed_decision": "RELEASE",
	"guardrail_triggered": false,
	"reasons": ["SUPPORTED_EVIDENCE_RELEASE"],
	"policy_version": policy_version,
} if {
	not input.system_error
	not input.novelty_override
	input.evidence_round == 1
	input.candidate_decision == "RELEASE"
	input.evidence_assessment.verdict == "SUPPORTED"
	input.probabilities.escalate < input.evidence_release_risk_cap
}

decision := {
	"allowed_decision": "EVIDENCE_NEEDED",
	"guardrail_triggered": false,
	"reasons": ["ONE_EVIDENCE_ROUND_AVAILABLE"],
	"policy_version": policy_version,
} if {
	not input.system_error
	not input.novelty_override
	input.evidence_round == 0
	input.candidate_decision == "EVIDENCE_NEEDED"
}

decision := {
	"allowed_decision": "ESCALATE",
	"guardrail_triggered": false,
	"reasons": ["CANDIDATE_ESCALATE"],
	"policy_version": policy_version,
} if {
	not input.system_error
	not input.novelty_override
	input.evidence_round <= 1
	input.candidate_decision == "ESCALATE"
}

decision := escalation("EVIDENCE_ROUND_EXHAUSTED") if {
	not input.system_error
	not input.novelty_override
	input.evidence_round == 1
	input.candidate_decision == "EVIDENCE_NEEDED"
}

escalation(reason) := {
	"allowed_decision": "ESCALATE",
	"guardrail_triggered": true,
	"reasons": [reason],
	"policy_version": policy_version,
}
