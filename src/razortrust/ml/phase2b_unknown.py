from __future__ import annotations

import numpy as np

from ..domain import HoldDecision

CONSENSUS_RULES = ("AE_ANY", "AE_BOTH", "TWO_OF_THREE")


def consensus_signal(
    rule: str,
    reconstruction_signal: np.ndarray,
    latent_signal: np.ndarray,
    isolation_signal: np.ndarray,
) -> np.ndarray:
    """Combine three novelty views without turning novelty into a fraud label."""
    reconstruction_signal = np.asarray(reconstruction_signal, dtype=bool)
    latent_signal = np.asarray(latent_signal, dtype=bool)
    isolation_signal = np.asarray(isolation_signal, dtype=bool)
    if not (reconstruction_signal.shape == latent_signal.shape == isolation_signal.shape):
        raise ValueError("novelty signal arrays must have identical shapes")
    if rule == "AE_ANY":
        return reconstruction_signal | latent_signal
    if rule == "AE_BOTH":
        return reconstruction_signal & latent_signal
    if rule == "TWO_OF_THREE":
        return (
            reconstruction_signal.astype(np.int8)
            + latent_signal.astype(np.int8)
            + isolation_signal.astype(np.int8)
        ) >= 2
    raise ValueError(f"unknown consensus rule: {rule}")


def apply_evidence_only_guard(
    base_actions: np.ndarray,
    novelty_signal: np.ndarray,
) -> np.ndarray:
    """Route novel RELEASE candidates to evidence; never escalate from novelty alone."""
    base_actions = np.asarray(base_actions, dtype=object)
    novelty_signal = np.asarray(novelty_signal, dtype=bool)
    if base_actions.shape != novelty_signal.shape:
        raise ValueError("base_actions and novelty_signal must have identical shapes")
    guarded = base_actions.copy()
    divert = novelty_signal & (base_actions == HoldDecision.RELEASE)
    guarded[divert] = HoldDecision.EVIDENCE_NEEDED
    return guarded
