from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn


class _Autoencoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 16),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Linear(4, 8),
            nn.SiLU(),
            nn.Linear(8, 16),
            nn.SiLU(),
            nn.Linear(16, width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class AutoencoderNoveltyScorer:
    """Small reconstruction model; its output is uncertainty, never a guilt label."""

    version = "autoencoder-novelty@1"

    def __init__(
        self,
        *,
        seed: int = 42,
        epochs: int = 200,
        patience: int = 20,
        input_noise_std: float = 0.03,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.patience = patience
        self.input_noise_std = input_noise_std
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.reference_errors: np.ndarray | None = None
        self.model: _Autoencoder | None = None

    def fit(self, legitimate_features: pd.DataFrame) -> AutoencoderNoveltyScorer:
        groups = pd.Series([f"row-{index}" for index in range(len(legitimate_features))])
        return self.fit_group_cross_fitted(legitimate_features, groups)

    def fit_group_cross_fitted(
        self,
        legitimate_features: pd.DataFrame,
        merchant_groups: pd.Series,
        *,
        n_splits: int = 5,
    ) -> AutoencoderNoveltyScorer:
        if len(legitimate_features) < 20:
            raise ValueError("autoencoder requires at least 20 legitimate training rows")
        if merchant_groups.nunique() < n_splits:
            raise ValueError("autoencoder cross-fitting requires enough merchant groups")
        self.columns = list(legitimate_features.columns)
        oof_errors = np.empty(len(legitimate_features), dtype=float)
        assigned = np.zeros(len(legitimate_features), dtype=bool)
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold, (fit_index, reference_index) in enumerate(
            splitter.split(legitimate_features, groups=merchant_groups), start=1
        ):
            temporary = AutoencoderNoveltyScorer(
                seed=self.seed + fold,
                epochs=self.epochs,
                patience=self.patience,
                input_noise_std=self.input_noise_std,
            )
            temporary.columns = self.columns
            temporary._fit_network(legitimate_features.iloc[fit_index])
            oof_errors[reference_index] = temporary._reconstruction_errors(
                legitimate_features.iloc[reference_index]
            )
            assigned[reference_index] = True
        if not assigned.all():
            raise RuntimeError("autoencoder cross-fitting did not score every reference row")
        self._fit_network(legitimate_features)
        self.reference_errors = np.sort(oof_errors)
        return self

    def _fit_network(self, legitimate_features: pd.DataFrame) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        raw = legitimate_features.to_numpy(dtype=np.float32)
        self.mean = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        tensor = torch.from_numpy((raw - self.mean) / self.scale)
        self.model = _Autoencoder(tensor.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = nn.SmoothL1Loss()
        self.model.train()
        best_loss = float("inf")
        stale_epochs = 0
        for _ in range(self.epochs):
            optimizer.zero_grad()
            noisy = tensor + torch.randn_like(tensor) * self.input_noise_std
            reconstruction = self.model(noisy)
            loss = loss_fn(reconstruction, tensor)
            loss.backward()
            optimizer.step()
            current_loss = float(loss.detach())
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.reference_errors is None:
            raise RuntimeError("autoencoder novelty scorer is not fitted")
        errors = self._reconstruction_errors(features)
        return np.searchsorted(self.reference_errors, errors, side="right") / len(
            self.reference_errors
        )

    def _reconstruction_errors(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.mean is None or self.scale is None:
            raise RuntimeError("autoencoder novelty scorer is not fitted")
        if list(features.columns) != self.columns:
            raise ValueError("autoencoder feature columns do not match the fitted schema")
        raw = features.to_numpy(dtype=np.float32)
        tensor = torch.from_numpy((raw - self.mean) / self.scale)
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(tensor)
            return torch.mean((reconstruction - tensor) ** 2, dim=1).numpy()
