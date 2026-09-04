package razortrust.delegation

import rego.v1

default permit := false

permit if {
	input.signature_verified
	input.context.agent_id == input.mandate.agent_id
	input.context.merchant_id == input.mandate.merchant_id
	input.context.action == input.mandate.action
	input.context.currency == input.mandate.currency
	input.context.amount <= input.mandate.maximum_amount
	input.context.occurred_at_ns >= input.mandate.valid_from_ns
	input.context.occurred_at_ns <= input.mandate.valid_until_ns
}

decision := {
	"permitted": permit,
	"policy_version": "delegation-policy@1",
	"reasons": reasons,
}

reasons := ["SIGNED_MANDATE_PERMITS"] if permit

reasons := ["DELEGATION_DENIED"] if not permit
