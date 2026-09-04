from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from razortrust.domain import FeatureVector


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_model_inputs_are_rejected_before_scoring(invalid: float) -> None:
    payload = {
        "volume_delta_z": invalid,
        "gmv_delta_z": 0.0,
        "ticket_size_delta_z": 0.0,
        "new_device_ratio": 0.0,
        "new_geo_ratio": 0.0,
        "refund_rate_delta_z": 0.0,
        "chargeback_rate_delta_z": 0.0,
        "failed_auth_ratio": 0.0,
        "volume_trend_slope": 0.0,
        "interarrival_time_cv": 0.0,
        "device_entropy": 0.0,
        "geo_entropy": 0.0,
        "amount_distribution_kl": 0.0,
    }
    with pytest.raises(ValidationError, match="finite number"):
        FeatureVector.model_validate(payload)


def test_out_of_range_ratios_are_rejected_before_scoring() -> None:
    with pytest.raises(ValidationError):
        FeatureVector(
            volume_delta_z=0,
            gmv_delta_z=0,
            ticket_size_delta_z=0,
            new_device_ratio=1.1,
            new_geo_ratio=0,
            refund_rate_delta_z=0,
            chargeback_rate_delta_z=0,
            failed_auth_ratio=0,
            volume_trend_slope=0,
            interarrival_time_cv=0,
            device_entropy=0,
            geo_entropy=0,
            amount_distribution_kl=0,
        )
