from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..costs import DEFAULT_COST_MATRIX
from ..synthetic import TrueRiskState
from .dataset import model_matrix
from .modeling import IsolationForestScorer, _target_indices, _with_anomaly
from .splits import SplitManifest


def tune_xgboost(
    frame: pd.DataFrame,
    split_manifest: SplitManifest,
    *,
    n_trials: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune only on training merchants with stratified grouped folds and early stopping."""
    training = frame[frame["merchant_id"].isin(split_manifest.train_merchants)].reset_index(
        drop=True
    )
    labels = _target_indices(training)
    groups = training["merchant_id"].astype(str)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    matrix = np.asarray(DEFAULT_COST_MATRIX.matrix)

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }
        fold_scores: list[float] = []
        for fold_train, fold_validation in folds.split(training, labels, groups):
            train = training.iloc[fold_train]
            validation = training.iloc[fold_validation]
            scorer = IsolationForestScorer(seed=seed).fit(
                model_matrix(train),
                train["cohort"],
                train["true_risk_state"].astype(str) == TrueRiskState.LEGITIMATE,
            )
            x_train = _with_anomaly(model_matrix(train), train["cohort"], scorer)
            x_validation = _with_anomaly(model_matrix(validation), validation["cohort"], scorer)
            y_train = _target_indices(train)
            y_validation = _target_indices(validation)
            classifier = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=1500,
                early_stopping_rounds=75,
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                **parameters,
            )
            classifier.fit(
                x_train,
                y_train,
                sample_weight=compute_sample_weight(class_weight="balanced", y=y_train),
                eval_set=[(x_validation, y_validation)],
                verbose=False,
            )
            probabilities = classifier.predict_proba(x_validation)
            actions = np.argmin(probabilities @ matrix.T, axis=1)
            false_release = float(np.mean(actions[y_validation != 0] == 0))
            if false_release > 0.05:
                return 1_000 + false_release
            expected_cost = float(np.mean(matrix[actions, y_validation]))
            fold_scores.append(
                expected_cost + float(log_loss(y_validation, probabilities, labels=[0, 1, 2]))
            )
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)
