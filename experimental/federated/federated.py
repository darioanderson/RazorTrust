from __future__ import annotations

import hashlib
import math
import secrets
from collections.abc import Mapping

import numpy as np
from pydantic import Field, model_validator

from razortrust.audit import canonical_json
from razortrust.domain import StrictModel
from razortrust.security import sign_manifest, verify_manifest


class FederatedUpdate(StrictModel):
    merchant_id: str = Field(min_length=1, max_length=128)
    round_id: str = Field(min_length=1, max_length=128)
    sample_count: int = Field(ge=1)
    delta: list[float] = Field(min_length=1, max_length=100_000)

    @model_validator(mode="after")
    def require_finite_delta(self) -> FederatedUpdate:
        if not np.isfinite(np.asarray(self.delta, dtype=float)).all():
            raise ValueError("federated update values must be finite")
        return self

    def signing_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class SignedFederatedUpdate(StrictModel):
    update: FederatedUpdate
    key_id: str = Field(min_length=1, max_length=128)
    signature_b64: str = Field(min_length=1)


class DPFederatedConfig(StrictModel):
    epsilon: float = Field(gt=0, le=20)
    delta: float = Field(gt=0, lt=1)
    clip_norm: float = Field(gt=0)
    min_clients: int = Field(default=5, ge=3)
    max_samples_per_client: int = Field(default=10_000, ge=1)


class DPAggregationResult(StrictModel):
    schema_version: str = "1.0"
    round_id: str
    client_count: int
    dimension: int
    epsilon: float
    delta: float
    clip_norm: float
    gaussian_sigma: float
    noisy_average: list[float]
    update_commitments: list[str]
    deterministic_test_noise: bool
    aggregator_version: str = "bounded-fedavg-gaussian@1"


def sign_federated_update(
    update: FederatedUpdate, *, key_id: str, private_key_b64: str
) -> SignedFederatedUpdate:
    return SignedFederatedUpdate(
        update=update,
        key_id=key_id,
        signature_b64=sign_manifest(update.signing_payload(), private_key_b64),
    )


class DPFederatedAggregator:
    """Verify, clip, aggregate, and noise cross-merchant model updates.

    The API accepts signed updates and releases only the differentially private aggregate. Transport
    encryption and a production SMPC protocol remain deployment responsibilities; this class does
    not claim that a single-process test harness hides updates from its own process memory.
    """

    def __init__(self, config: DPFederatedConfig, public_keys: Mapping[str, str]) -> None:
        self.config = config
        self.public_keys = dict(public_keys)

    def aggregate(
        self, updates: list[SignedFederatedUpdate], *, test_seed: int | None = None
    ) -> DPAggregationResult:
        if len(updates) < self.config.min_clients:
            raise ValueError("federated round does not meet the minimum client threshold")
        merchant_ids = [item.update.merchant_id for item in updates]
        if len(set(merchant_ids)) != len(merchant_ids):
            raise ValueError("a merchant may submit only one update per round")
        round_ids = {item.update.round_id for item in updates}
        if len(round_ids) != 1:
            raise ValueError("all federated updates must use the same round_id")
        dimensions = {len(item.update.delta) for item in updates}
        if len(dimensions) != 1:
            raise ValueError("federated update dimensions do not match")
        for signed in updates:
            public_key = self.public_keys.get(signed.key_id)
            if public_key is None or not verify_manifest(
                signed.update.signing_payload(), signed.signature_b64, public_key
            ):
                raise ValueError("federated update signature verification failed")

        clipped = np.vstack([self._clip(item.update.delta) for item in updates])
        counts = np.asarray(
            [min(item.update.sample_count, self.config.max_samples_per_client) for item in updates],
            dtype=float,
        )
        weights = counts / counts.sum()
        average = np.average(clipped, axis=0, weights=weights)
        sensitivity = 2 * self.config.clip_norm * float(weights.max())
        sigma = (
            sensitivity * math.sqrt(2 * math.log(1.25 / self.config.delta)) / self.config.epsilon
        )
        noise = self._gaussian_noise(len(average), sigma, test_seed)
        commitments = sorted(
            hashlib.sha256(canonical_json(item.update.signing_payload())).hexdigest()
            for item in updates
        )
        return DPAggregationResult(
            round_id=next(iter(round_ids)),
            client_count=len(updates),
            dimension=len(average),
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            clip_norm=self.config.clip_norm,
            gaussian_sigma=sigma,
            noisy_average=(average + noise).tolist(),
            update_commitments=commitments,
            deterministic_test_noise=test_seed is not None,
        )

    def _clip(self, delta: list[float]) -> np.ndarray:
        values = np.asarray(delta, dtype=float)
        norm = float(np.linalg.norm(values))
        if norm > self.config.clip_norm:
            values = values * (self.config.clip_norm / norm)
        return values

    @staticmethod
    def _gaussian_noise(dimension: int, sigma: float, test_seed: int | None) -> np.ndarray:
        if test_seed is not None:
            return np.random.default_rng(test_seed).normal(0.0, sigma, size=dimension)
        secure_random = secrets.SystemRandom()
        return np.asarray([secure_random.gauss(0.0, sigma) for _ in range(dimension)])
