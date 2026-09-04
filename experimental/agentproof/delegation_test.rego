package razortrust.delegation_test

import data.razortrust.delegation
import rego.v1

valid := {
	"signature_verified": true,
	"mandate": {
		"agent_id": "agent-1",
		"merchant_id": "merchant-1",
		"action": "SETTLE",
		"currency": "INR",
		"maximum_amount": 100000,
		"valid_from_ns": 10,
		"valid_until_ns": 20,
	},
	"context": {
		"agent_id": "agent-1",
		"merchant_id": "merchant-1",
		"action": "SETTLE",
		"currency": "INR",
		"amount": 50000,
		"occurred_at_ns": 15,
	},
}

test_valid_signed_mandate_permits if {
	delegation.permit with input as valid
}

test_amount_over_limit_denies if {
	not delegation.permit with input as object.union(valid, {
		"context": object.union(valid.context, {"amount": 100001}),
	})
}

test_bad_signature_denies if {
	not delegation.permit with input as object.union(valid, {"signature_verified": false})
}
