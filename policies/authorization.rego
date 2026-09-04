package razortrust.authz

import rego.v1

policy_version := "api-authz@1"

roles_by_action := {
	"CREATE_HOLD": {"MERCHANT", "RISK_SERVICE", "ADMIN"},
	"EVALUATE_HOLD": {"RISK_ANALYST", "RISK_SERVICE", "ADMIN"},
	"VIEW_HOLD": {"MERCHANT", "RISK_ANALYST", "RISK_SERVICE", "ADMIN"},
	"LIST_HOLDS": {"MERCHANT", "RISK_ANALYST", "RISK_SERVICE", "ADMIN"},
	"SUBMIT_EVIDENCE": {"MERCHANT", "EVIDENCE_SERVICE", "ADMIN"},
	"VIEW_AUDIT": {"RISK_ANALYST", "ADMIN"},
	"VIEW_OPERATOR_DASHBOARD": {"RISK_ANALYST", "ADMIN"},
	"CHECK_POLICY": {"RISK_SERVICE", "ADMIN"},
	"RECORD_ANALYST_OUTCOME": {"RISK_ANALYST", "ADMIN"},
	"VIEW_INTEGRATION": {"RISK_ANALYST", "RISK_SERVICE", "ADMIN"},
	"MANAGE_INTEGRATION": {"RISK_SERVICE", "ADMIN"},
	"SUBMIT_TELEMETRY": {"MERCHANT", "RISK_SERVICE", "ADMIN"},
	"VIEW_FEATURE_CONTRACT": {"RISK_ANALYST", "RISK_SERVICE", "ADMIN"},
	"CREATE_CHECKOUT": {"MERCHANT", "RISK_SERVICE", "ADMIN"},
	"VERIFY_CHECKOUT": {"MERCHANT", "RISK_SERVICE", "ADMIN"},
}

role_allowed if input.principal.role in roles_by_action[input.action]

resource_allowed if input.principal.role != "MERCHANT"

resource_allowed if {
	input.principal.role == "MERCHANT"
	input.principal.merchant_id != ""
	input.principal.merchant_id == input.merchant_id
}

decision := {
	"allowed": true,
	"reasons": ["AUTHORIZED"],
	"policy_version": policy_version,
} if {
	role_allowed
	resource_allowed
}

default decision := {
	"allowed": false,
	"reasons": ["ACCESS_DENIED"],
	"policy_version": "api-authz@1",
}
