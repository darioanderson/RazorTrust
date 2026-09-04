from __future__ import annotations

from datetime import datetime, timedelta

from .domain import (
    EvidenceRequirement,
    EvidenceType,
    FeatureContribution,
    HoldDecision,
    MerchantGuidance,
)

_FEATURE_MESSAGES = {
    "volume_delta_z": "Your transaction volume is unusually high compared with your baseline.",
    "gmv_delta_z": "Your gross transaction value increased sharply compared with your baseline.",
    "ticket_size_delta_z": "Your average transaction amount is unusually different from normal.",
    "new_device_ratio": "An unusually large share of transactions came from new devices.",
    "new_geo_ratio": "An unusually large share of customers came from new locations.",
    "refund_rate_delta_z": "Your refund rate is unusually high compared with your baseline.",
    "chargeback_rate_delta_z": (
        "Your chargeback rate is unusually high compared with your baseline."
    ),
    "failed_auth_ratio": (
        "The recent window contains an unusually high share of failed authorizations."
    ),
    "volume_trend_slope": (
        "Transaction volume accelerated unusually quickly during the review window."
    ),
    "interarrival_time_cv": (
        "Transaction timing is unusually concentrated compared with normal activity."
    ),
    "device_entropy": "Recent transactions show an unusual device-usage pattern.",
    "geo_entropy": "Recent transactions show an unusual geographic pattern.",
    "amount_distribution_kl": (
        "The mix of transaction amounts differs materially from your baseline."
    ),
    "isolation_forest_percentile": (
        "Recent activity is unlike the legitimate patterns seen during training."
    ),
}

_FEATURE_EVIDENCE = {
    "volume_delta_z": EvidenceType.CAMPAIGN,
    "gmv_delta_z": EvidenceType.CAMPAIGN,
    "ticket_size_delta_z": EvidenceType.INVENTORY,
    "new_device_ratio": EvidenceType.AUTH_INCIDENT,
    "new_geo_ratio": EvidenceType.GEO_EXPANSION,
    "refund_rate_delta_z": EvidenceType.FULFILLMENT,
    "chargeback_rate_delta_z": EvidenceType.FULFILLMENT,
    "failed_auth_ratio": EvidenceType.AUTH_INCIDENT,
    "volume_trend_slope": EvidenceType.CAMPAIGN,
    "interarrival_time_cv": EvidenceType.AUTH_INCIDENT,
    "device_entropy": EvidenceType.AUTH_INCIDENT,
    "geo_entropy": EvidenceType.GEO_EXPANSION,
    "amount_distribution_kl": EvidenceType.INVENTORY,
    "isolation_forest_percentile": EvidenceType.OTHER,
}

_REQUIREMENTS = {
    EvidenceType.CAMPAIGN: EvidenceRequirement(
        evidence_type=EvidenceType.CAMPAIGN,
        title="Campaign or sales-event proof",
        description="Upload the campaign brief, dated promotion schedule, and matching invoices.",
    ),
    EvidenceType.FULFILLMENT: EvidenceRequirement(
        evidence_type=EvidenceType.FULFILLMENT,
        title="Fulfilment and refund records",
        description=(
            "Upload dispatch proof, refund logs, and the relevant customer-resolution records."
        ),
    ),
    EvidenceType.GEO_EXPANSION: EvidenceRequirement(
        evidence_type=EvidenceType.GEO_EXPANSION,
        title="Geographic expansion proof",
        description=(
            "Upload launch documentation and invoices or orders for the new service region."
        ),
    ),
    EvidenceType.AUTH_INCIDENT: EvidenceRequirement(
        evidence_type=EvidenceType.AUTH_INCIDENT,
        title="Authentication or device incident report",
        description=(
            "Upload the incident timeline, affected device ranges, and remediation evidence."
        ),
    ),
    EvidenceType.INVENTORY: EvidenceRequirement(
        evidence_type=EvidenceType.INVENTORY,
        title="Invoice and inventory proof",
        description="Upload supplier invoices, inventory records, and matching customer orders.",
    ),
    EvidenceType.OTHER: EvidenceRequirement(
        evidence_type=EvidenceType.OTHER,
        title="Transaction-context evidence",
        description="Upload dated records that explain the unusual transaction pattern.",
    ),
}


def build_merchant_guidance(
    decision: HoldDecision,
    top_features: list[FeatureContribution],
    *,
    created_at: datetime,
    deadline_hours: int = 48,
) -> MerchantGuidance | None:
    if decision != HoldDecision.EVIDENCE_NEEDED:
        return None
    selected_features = [item.feature for item in top_features[:3]]
    reasons = [
        _FEATURE_MESSAGES.get(feature, f"The {feature} signal needs verification.")
        for feature in selected_features
    ]
    if not reasons:
        reasons = ["Recent transaction activity differs materially from your established baseline."]
    evidence_types = []
    for feature in selected_features:
        evidence_type = _FEATURE_EVIDENCE.get(feature, EvidenceType.OTHER)
        if evidence_type not in evidence_types:
            evidence_types.append(evidence_type)
    if not evidence_types:
        evidence_types = [EvidenceType.OTHER]
    submit_by = created_at + timedelta(hours=deadline_hours)
    return MerchantGuidance(
        summary="Your settlement is temporarily held while we verify unusual recent activity.",
        reasons=reasons,
        required_evidence=[_REQUIREMENTS[evidence_type] for evidence_type in evidence_types],
        submit_by=submit_by,
        next_step=(
            f"Submit one complete evidence package by {submit_by.isoformat()}; "
            "otherwise the case moves to human review."
        ),
    )
