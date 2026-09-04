# Cross-merchant privacy prototype

`DPFederatedAggregator` is the minimum auditable cross-merchant aggregation layer:

- every merchant update is Ed25519-signed and tied to one round;
- duplicate merchants, mixed rounds, dimensions, invalid signatures, and fewer than the configured
  minimum clients are rejected;
- each update is L2-clipped and each client sample weight is capped;
- the weighted average receives Gaussian noise derived from the configured epsilon, delta, clipping
  norm, and maximum bounded client weight;
- the returned object contains only the noisy vector, privacy parameters, and content commitments—not
  merchant IDs or individual deltas;
- production noise uses the operating system's secure random source. A deterministic seed exists only
  for tests and is labelled in the output.

This is a DP aggregation and authenticity prototype, not a claim of deployed secure multiparty
computation. A single-process coordinator necessarily sees submitted updates in memory. Production
deployment must add mutually authenticated transport and a reviewed secure-aggregation protocol or
trusted execution boundary, enforce a cumulative privacy accountant across rounds, rotate merchant
keys, and obtain privacy/legal approval. Raw transaction rows are never part of the update contract.
