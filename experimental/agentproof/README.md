# Archived AgentProof direction

The first prototype explored signed human-to-agent payment delegation. That is a different product from the locked Settlement Hold Risk Verifier, so it is intentionally excluded from the application package and architecture.

The useful idea retained from that experiment is Ed25519 signing. In the current system it signs model, policy, evaluation, and audit-checkpoint manifests—not payment mandates.
