from __future__ import annotations

import pytest

from experimental.federated.federated import (
    DPFederatedAggregator,
    DPFederatedConfig,
    FederatedUpdate,
    SignedFederatedUpdate,
    sign_federated_update,
)
from razortrust.security import generate_release_keypair


def _signed_updates(count: int = 3):
    private_key, public_key = generate_release_keypair()
    updates = [
        sign_federated_update(
            FederatedUpdate(
                merchant_id=f"merchant-{index}",
                round_id="round-1",
                sample_count=100 + index,
                delta=[100.0, float(index)],
            ),
            key_id="merchant-key",
            private_key_b64=private_key,
        )
        for index in range(count)
    ]
    return updates, public_key


def test_dp_federated_aggregation_verifies_clips_and_releases_only_aggregate() -> None:
    updates, public_key = _signed_updates()
    aggregator = DPFederatedAggregator(
        DPFederatedConfig(epsilon=2.0, delta=1e-5, clip_norm=1.0, min_clients=3),
        {"merchant-key": public_key},
    )

    result = aggregator.aggregate(updates, test_seed=42)

    assert result.client_count == 3
    assert result.dimension == 2
    assert result.gaussian_sigma > 0
    assert len(result.update_commitments) == 3
    assert result.deterministic_test_noise is True
    assert "merchant" not in result.model_dump_json()


def test_dp_federated_aggregation_rejects_tampering_and_too_few_clients() -> None:
    updates, public_key = _signed_updates()
    aggregator = DPFederatedAggregator(
        DPFederatedConfig(epsilon=2.0, delta=1e-5, clip_norm=1.0, min_clients=3),
        {"merchant-key": public_key},
    )
    tampered = updates[0].model_copy(
        update={"update": updates[0].update.model_copy(update={"delta": [0.0, 0.0]})}
    )

    with pytest.raises(ValueError, match="signature"):
        aggregator.aggregate([tampered, *updates[1:]], test_seed=42)
    with pytest.raises(ValueError, match="minimum client"):
        aggregator.aggregate(updates[:2], test_seed=42)


def test_signed_update_requires_registered_key() -> None:
    updates, _ = _signed_updates()
    untrusted = SignedFederatedUpdate.model_validate(updates[0].model_dump())
    aggregator = DPFederatedAggregator(
        DPFederatedConfig(epsilon=2.0, delta=1e-5, clip_norm=1.0, min_clients=3),
        {},
    )
    with pytest.raises(ValueError, match="signature"):
        aggregator.aggregate([untrusted, *updates[1:]], test_seed=42)
