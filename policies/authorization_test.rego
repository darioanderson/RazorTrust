package razortrust.authz_test

import data.razortrust.authz
import rego.v1

merchant := {
	"principal": {"id": "merchant-a", "role": "MERCHANT", "merchant_id": "merchant-a"},
	"action": "VIEW_HOLD",
	"resource_type": "HOLD",
	"resource_id": "hold-1",
	"merchant_id": "merchant-a",
}

test_merchant_can_view_own_hold if {
	result := authz.decision with input as merchant
	result.allowed
}

test_merchant_cannot_view_another_merchants_hold if {
	request := object.union(merchant, {"merchant_id": "merchant-b"})
	result := authz.decision with input as request
	not result.allowed
}

test_merchant_cannot_view_audit if {
	request := object.union(merchant, {"action": "VIEW_AUDIT"})
	result := authz.decision with input as request
	not result.allowed
}

test_analyst_can_view_audit if {
	request := object.union(merchant, {
		"principal": {"id": "analyst-1", "role": "RISK_ANALYST", "merchant_id": ""},
		"action": "VIEW_AUDIT",
	})
	result := authz.decision with input as request
	result.allowed
}

test_unknown_action_denied if {
	request := object.union(merchant, {"action": "DELETE_EVERYTHING"})
	result := authz.decision with input as request
	not result.allowed
}
