from __future__ import annotations

from collections import defaultdict
from uuid import UUID

import pandas as pd

from ..domain import HoldCase, HoldEvaluationInput, TransactionEvent
from ..features import FEATURE_COLUMNS, build_point_in_time_features
from ..synthetic import SyntheticHold, SyntheticMerchant

METADATA_COLUMNS = (
    "merchant_id",
    "cohort",
    "scenario_family",
    "true_risk_state",
    "evidence_resolvable",
    "attack_family",
    "operational_target",
)
FORBIDDEN_MODEL_COLUMNS = {
    "merchant_id",
    "transaction_id",
    "customer_id",
    "ring_id",
    "scenario_family",
    "attack_family",
    "true_risk_state",
    "operational_target",
}


def build_training_frame(
    merchants: list[SyntheticMerchant],
    transactions: list[TransactionEvent],
    holds: list[SyntheticHold],
) -> pd.DataFrame:
    """Build one point-in-time model row per synthetic hold."""
    merchant_by_id = {merchant.merchant_id: merchant for merchant in merchants}
    transactions_by_merchant: dict[str, list[TransactionEvent]] = defaultdict(list)
    for transaction in transactions:
        transactions_by_merchant[transaction.merchant_id].append(transaction)

    rows: list[dict[str, object]] = []
    for synthetic_hold in holds:
        create = synthetic_hold.hold
        merchant = merchant_by_id[create.merchant_id]
        hold = HoldCase(
            hold_id=UUID(str(create.request_id)),
            **create.model_dump(),
        )
        features = build_point_in_time_features(
            hold,
            HoldEvaluationInput(
                baseline=merchant.baseline,
                transactions=transactions_by_merchant[merchant.merchant_id],
                cohort=merchant.cohort,
            ),
        )
        rows.append(
            {
                **features.model_dump(),
                "merchant_id": merchant.merchant_id,
                "cohort": merchant.cohort,
                "scenario_family": synthetic_hold.scenario_family,
                "true_risk_state": synthetic_hold.true_risk_state,
                "evidence_resolvable": synthetic_hold.evidence_resolvable,
                "attack_family": synthetic_hold.attack_family,
                "operational_target": synthetic_hold.operational_target,
            }
        )

    frame = pd.DataFrame(rows, columns=(*FEATURE_COLUMNS, *METADATA_COLUMNS))
    if tuple(frame.columns[: len(FEATURE_COLUMNS)]) != FEATURE_COLUMNS:
        raise RuntimeError("training frame does not start with the locked feature contract")
    return frame


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the locked model inputs, in contract order."""
    matrix = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    leaked = set(matrix.columns) & FORBIDDEN_MODEL_COLUMNS
    if leaked:
        raise RuntimeError(f"forbidden model columns detected: {sorted(leaked)}")
    return matrix
