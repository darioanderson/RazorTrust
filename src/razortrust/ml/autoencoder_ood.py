from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupShuffleSplit
from torch import nn


@dataclass(frozen=True)
class AutoencoderArchitecture:
    name: str
    hidden_widths: tuple[int, ...]
    latent_dim: int = 4


COMPACT_ARCHITECTURE = AutoencoderArchitecture("compact-13-8-4-8-13", (8,), 4)
WIDE_ARCHITECTURE = AutoencoderArchitecture("wide-13-16-8-4-8-16-13", (16, 8), 4)
DEFAULT_ARCHITECTURES = (COMPACT_ARCHITECTURE, WIDE_ARCHITECTURE)


class _DenoisingAutoencoder(nn.Module):
    def __init__(self, width: int, architecture: AutoencoderArchitecture, dropout: float) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = width
        for hidden in architecture.hidden_widths:
            encoder_layers.extend([nn.Linear(previous, hidden), nn.GELU()])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            previous = hidden
        encoder_layers.append(nn.Linear(previous, architecture.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = architecture.latent_dim
        for hidden in reversed(architecture.hidden_widths):
            decoder_layers.extend([nn.Linear(previous, hidden), nn.GELU()])
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            previous = hidden
        decoder_layers.append(nn.Linear(previous, width))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(values)
        return self.decoder(latent), latent


@dataclass(frozen=True)
class AutoencoderOODScores:
    reconstruction_error: np.ndarray
    latent_mahalanobis_sq: np.ndarray
    latent_vectors: np.ndarray


class AutoencoderOODScorer:
    """Normal-only denoising AE plus latent-space OOD distance.

    This is a research novelty component, not a fraud classifier. It is fitted only
    on legitimate training merchants. Thresholds must be calibrated on a disjoint
    legitimate calibration partition by the caller.
    """

    version = "autoencoder-ood@2-research"

    def __init__(
        self,
        *,
        architecture: AutoencoderArchitecture = COMPACT_ARCHITECTURE,
        seed: int = 42,
        epochs: int = 120,
        patience: int = 15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        input_noise_std: float = 0.03,
        dropout: float = 0.05,
    ) -> None:
        self.architecture = architecture
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_noise_std = input_noise_std
        self.dropout = dropout
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.model: _DenoisingAutoencoder | None = None
        self.latent_covariance: LedoitWolf | None = None
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None

    def fit(
        self, legitimate_features: pd.DataFrame, merchant_groups: pd.Series
    ) -> AutoencoderOODScorer:
        if len(legitimate_features) < 30:
            raise ValueError("autoencoder OOD requires at least 30 legitimate training rows")
        groups = merchant_groups.astype(str).reset_index(drop=True)
        features = legitimate_features.reset_index(drop=True)
        if len(groups) != len(features):
            raise ValueError("merchant_groups length does not match legitimate_features")
        if groups.nunique() < 5:
            raise ValueError("autoencoder OOD requires at least five legitimate merchant groups")
        self.columns = list(features.columns)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=self.seed)
        fit_idx, validation_idx = next(splitter.split(features, groups=groups))
        fit_frame = features.iloc[fit_idx].reset_index(drop=True)
        validation_frame = features.iloc[validation_idx].reset_index(drop=True)

        self._seed_everything(self.seed)
        raw_fit = fit_frame.to_numpy(dtype=np.float32)
        self.mean = raw_fit.mean(axis=0)
        self.scale = raw_fit.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0

        train_tensor = self._standardized_tensor(fit_frame)
        validation_tensor = self._standardized_tensor(validation_frame)
        self.model = _DenoisingAutoencoder(train_tensor.shape[1], self.architecture, self.dropout)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.SmoothL1Loss()
        generator = torch.Generator().manual_seed(self.seed + 17)
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        stale = 0
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            noise = torch.randn(train_tensor.shape, generator=generator) * self.input_noise_std
            reconstruction, _ = self.model(train_tensor + noise)
            train_loss = loss_fn(reconstruction, train_tensor)
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                validation_reconstruction, _ = self.model(validation_tensor)
                validation_loss = float(loss_fn(validation_reconstruction, validation_tensor))
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone() for key, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is None:
            raise RuntimeError("autoencoder training failed to produce a valid checkpoint")
        self.model.load_state_dict(best_state)
        self.best_epoch = best_epoch
        self.best_validation_loss = best_loss

        # Fit the latent reference distribution on the legitimate training partition.
        # Shrinkage keeps the 4-D covariance well-conditioned for finite samples.
        latent_fit = self._latent_vectors(features)
        self.latent_covariance = LedoitWolf(store_precision=True).fit(latent_fit)
        return self

    def score_details(self, features: pd.DataFrame) -> AutoencoderOODScores:
        self._check_ready(features)
        tensor = self._standardized_tensor(features.reset_index(drop=True))
        assert self.model is not None
        assert self.latent_covariance is not None
        self.model.eval()
        with torch.no_grad():
            reconstruction, latent = self.model(tensor)
            reconstruction_error = torch.mean((reconstruction - tensor) ** 2, dim=1).numpy()
            latent_values = latent.numpy()
        latent_distance = self.latent_covariance.mahalanobis(latent_values)
        return AutoencoderOODScores(
            reconstruction_error=np.asarray(reconstruction_error, dtype=float),
            latent_mahalanobis_sq=np.asarray(latent_distance, dtype=float),
            latent_vectors=np.asarray(latent_values, dtype=float),
        )

    def _latent_vectors(self, features: pd.DataFrame) -> np.ndarray:
        self._check_network_ready(features)
        tensor = self._standardized_tensor(features.reset_index(drop=True))
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(tensor).numpy()

    def _standardized_tensor(self, features: pd.DataFrame) -> torch.Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("autoencoder standardizer is not fitted")
        raw = features.to_numpy(dtype=np.float32)
        return torch.from_numpy((raw - self.mean) / self.scale)

    def _check_network_ready(self, features: pd.DataFrame) -> None:
        if self.model is None or self.mean is None or self.scale is None:
            raise RuntimeError("autoencoder OOD scorer is not fitted")
        if list(features.columns) != self.columns:
            raise ValueError("autoencoder OOD feature columns do not match fitted schema")

    def _check_ready(self, features: pd.DataFrame) -> None:
        self._check_network_ready(features)
        if self.latent_covariance is None:
            raise RuntimeError("autoencoder latent covariance is not fitted")

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)


@dataclass(frozen=True)
class UpperTailCalibration:
    target_false_alarm_rate: float
    threshold: float
    calibration_size: int


def calibrate_upper_tail(
    legitimate_scores: np.ndarray, *, target_false_alarm_rate: float
) -> UpperTailCalibration:
    values = np.asarray(legitimate_scores, dtype=float)
    if values.ndim != 1 or len(values) < 20:
        raise ValueError("upper-tail calibration requires at least 20 legitimate scores")
    if not np.isfinite(values).all():
        raise ValueError("upper-tail calibration scores must be finite")
    if not 0 < target_false_alarm_rate < 0.5:
        raise ValueError("target_false_alarm_rate must be in (0, 0.5)")
    ordered = np.sort(values)
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - target_false_alarm_rate)))
    rank = min(max(rank, 1), len(ordered))
    return UpperTailCalibration(
        target_false_alarm_rate=float(target_false_alarm_rate),
        threshold=float(ordered[rank - 1]),
        calibration_size=len(ordered),
    )
