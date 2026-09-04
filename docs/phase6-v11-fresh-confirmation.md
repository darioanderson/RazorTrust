# Phase 6 V1.1 — Fresh Conformal Confirmation

Status: research-only.

V1.1 does not rewrite the Phase 6 V1 report or its failed pre-registered gate.
The V1 decision remains frozen evidence.

## Why V1.1 exists

Phase 6 V1 used a two-sided acceptance rule that required the nominal 90%
coverage target to lie inside the empirical Wilson 95% interval. This
incorrectly penalized conservative overcoverage.

Standard conformal classification targets marginal coverage at least equal to
the requested confidence level. V1.1 therefore uses fresh data and a corrected
undercoverage-oriented gate.

## Fresh data

The frozen default seeds are:

- 20260923: development
- 20260924: strict split-conformal conformalization
- 20260925: independent research test

The canonical SEALED TEST and stress set remain untouched.

## Frozen V1.1 gate

V1.1 passes only if all conditions hold:

1. empirical marginal coverage is at least 0.90;
2. a one-sided exact binomial test with alternative `coverage < 0.90` does not
   find statistically significant undercoverage at alpha 0.05;
3. ambiguous prediction sets never route to RELEASE;
4. empty prediction sets never route to RELEASE;
5. full prediction sets never route to RELEASE;
6. SEALED TEST remains unused;
7. no production promotion occurs.

The binomial test is an additional validation diagnostic. It does not replace
the finite-sample conformal guarantee.

## Controlled shift

The same canonical shift probes are retained at +0.50 and +1.00 reference
standard deviations. No formal conformal coverage guarantee is claimed after
detected distribution shift.

## Safety

The champion remains `xgb-if-settlement@2`.
The active enforcement runtime remains `human-only@1`.
Automatic RELEASE, automatic promotion, and automatic retraining remain
disabled.
