package razortrust.holds_test

import data.razortrust.holds
import rego.v1

base_input := {
	"candidate_decision": "RELEASE",
	"probabilities": {"release": 0.94, "evidence_needed": 0.04, "escalate": 0.02},
	"thresholds": {"release": 0.80, "escalate": 0.55},
	"novelty_override": false,
	"evidence_round": 0,
	"system_error": false,
}

test_safe_release if {
	result := holds.decision with input as base_input
	result.allowed_decision == "RELEASE"
}

test_system_error_escalates if {
	result := holds.decision with input as object.union(base_input, {"system_error": true})
	result.allowed_decision == "ESCALATE"
	result.guardrail_triggered
}

test_novelty_escalates if {
	result := holds.decision with input as object.union(base_input, {"novelty_override": true})
	result.allowed_decision == "ESCALATE"
}

test_second_evidence_round_is_not_available if {
	evidence_input := object.union(base_input, {
		"candidate_decision": "EVIDENCE_NEEDED",
		"evidence_round": 1,
	})
	result := holds.decision with input as evidence_input
	result.allowed_decision == "ESCALATE"
}
