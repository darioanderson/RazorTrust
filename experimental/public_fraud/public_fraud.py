from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from razortrust.domain import StrictModel

ULB_FEATURES = ("Time", *(f"V{index}" for index in range(1, 29)), "Amount")
ULB_COLUMNS = (*ULB_FEATURES, "Class")


class DetectorMetric(StrictModel):
    name: str
    precision: float
    recall: float
    pr_auc: float
    threshold: float


class BusinessCostMetric(StrictModel):
    name: str
    threshold: float
    total_cost: float
    fraud_loss: float
    legitimate_delay_cost: float
    escalated_count: int


class PublicFraudReport(StrictModel):
    schema_version: str = "1.0"
    dataset: str
    dataset_sha256: str
    source_tree_sha256: str
    seed: int
    row_count: int
    fraud_count: int
    train_rows: int
    validation_rows: int
    sealed_test_rows: int
    supervised_pr_auc: float
    selection_partition: str
    validation_isolation_forest: DetectorMetric
    validation_autoencoders: list[DetectorMetric]
    isolation_forest: DetectorMetric
    autoencoders: list[DetectorMetric]
    selected_autoencoder: str | None
    selected_novelty_strategy: str
    autoencoder_metric_gate_passed: bool
    autoencoder_gate_passed: bool
    autoencoder_training: dict[str, dict[str, int | float | bool]]
    validation_supervised_cost: BusinessCostMetric
    validation_novelty_candidates: list[BusinessCostMetric]
    held_out_cost_gate_passed: bool
    research_decision: str
    promotion_eligible: bool = False
    production_action_eligible: bool = False
    model_artifacts_sha256: dict[str, str]
    supervised_cost: BusinessCostMetric
    disagreement_override_cost: BusinessCostMetric


def load_ulb_credit_card(path: str | Path) -> tuple[pd.DataFrame, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source, compression="infer")
    if tuple(frame.columns) != ULB_COLUMNS:
        raise ValueError("ULB dataset columns do not match Time, V1..V28, Amount, Class")
    values = frame.loc[:, list(ULB_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ULB dataset contains non-finite feature values")
    if (frame["Time"] < 0).any() or (frame["Amount"] < 0).any():
        raise ValueError("ULB Time and Amount must be non-negative")
    labels = set(frame["Class"].astype(int).unique())
    if labels != {0, 1}:
        raise ValueError("ULB Class must contain both binary labels 0 and 1")
    return frame.sort_values("Time", kind="stable").reset_index(drop=True), digest


def evaluate_public_fraud(
    path: str | Path,
    *,
    source_tree_sha256: str,
    output_dir: str | Path,
    seed: int = 42,
    estimators: int = 300,
    legitimate_fit_limit: int = 50_000,
) -> PublicFraudReport:
    frame, dataset_sha256 = load_ulb_credit_card(path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to replace immutable research run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train, validation, test = _chronological_split(frame)
    scaler = StandardScaler().fit(train.loc[:, list(ULB_FEATURES)])
    x_train = scaler.transform(train.loc[:, list(ULB_FEATURES)])
    x_validation = scaler.transform(validation.loc[:, list(ULB_FEATURES)])
    x_test = scaler.transform(test.loc[:, list(ULB_FEATURES)])
    y_train = train["Class"].to_numpy(dtype=int)
    y_validation = validation["Class"].to_numpy(dtype=int)
    y_test = test["Class"].to_numpy(dtype=int)

    positives = max(1, int(y_train.sum()))
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=estimators,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=1.0,
        scale_pos_weight=(len(y_train) - positives) / positives,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
    )
    classifier.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    validation_probability = classifier.predict_proba(x_validation)[:, 1]
    test_probability = classifier.predict_proba(x_test)[:, 1]
    supervised_validation_cost = optimize_business_threshold(
        y_validation,
        validation_probability,
        validation["Amount"].to_numpy(dtype=float),
    )
    supervised_threshold = supervised_validation_cost.threshold
    supervised_cost = business_cost(
        "xgboost",
        y_test,
        test_probability,
        test["Amount"].to_numpy(dtype=float),
        supervised_threshold,
    )

    rng = np.random.default_rng(seed)
    legitimate_train = x_train[y_train == 0]
    if len(legitimate_train) > legitimate_fit_limit:
        legitimate_train = legitimate_train[
            rng.choice(len(legitimate_train), legitimate_fit_limit, replace=False)
        ]
    isolation_forest = IsolationForest(
        n_estimators=estimators,
        max_samples=min(2048, len(legitimate_train)),
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(legitimate_train)
    validation_if = -isolation_forest.score_samples(x_validation)
    test_if = -isolation_forest.score_samples(x_test)
    if_threshold = float(np.quantile(validation_if[y_validation == 0], 0.995))
    validation_if_metric = detector_metric(
        "isolation_forest", y_validation, validation_if, if_threshold
    )
    if_metric = detector_metric("isolation_forest", y_test, test_if, if_threshold)

    architectures = {
        "autoencoder_shallow_4": (12, 4, 12),
        "autoencoder_deep_4": (24, 12, 4, 12, 24),
    }
    autoencoder_metrics: list[DetectorMetric] = []
    validation_autoencoder_metrics: list[DetectorMetric] = []
    autoencoder_models: dict[str, MLPRegressor] = {}
    autoencoder_training: dict[str, dict[str, int | float | bool]] = {}
    autoencoder_validation_scores: dict[str, np.ndarray] = {}
    autoencoder_test_scores: dict[str, np.ndarray] = {}
    for offset, (name, layers) in enumerate(architectures.items()):
        maximum_iterations = 200
        model = MLPRegressor(
            hidden_layer_sizes=layers,
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=512,
            learning_rate_init=0.001,
            max_iter=maximum_iterations,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed + offset,
        ).fit(legitimate_train, legitimate_train)
        validation_error = np.mean((model.predict(x_validation) - x_validation) ** 2, axis=1)
        test_error = np.mean((model.predict(x_test) - x_test) ** 2, axis=1)
        threshold = float(np.quantile(validation_error[y_validation == 0], 0.995))
        validation_autoencoder_metrics.append(
            detector_metric(name, y_validation, validation_error, threshold)
        )
        autoencoder_metrics.append(detector_metric(name, y_test, test_error, threshold))
        autoencoder_models[name] = model
        autoencoder_training[name] = {
            "iterations": int(model.n_iter_),
            "maximum_iterations": maximum_iterations,
            "converged_before_limit": bool(model.n_iter_ < maximum_iterations),
            "final_loss": float(model.loss_),
        }
        autoencoder_validation_scores[name] = validation_error
        autoencoder_test_scores[name] = test_error

    metric_passing = [
        metric
        for metric in validation_autoencoder_metrics
        if metric.precision >= validation_if_metric.precision
        and metric.recall >= validation_if_metric.recall
    ]

    validation_amounts = validation["Amount"].to_numpy(dtype=float)
    validation_candidates: list[
        tuple[str, DetectorMetric | None, BusinessCostMetric]
    ] = []
    validation_if_flag = validation_if >= if_threshold
    validation_if_probability = np.where(
        (validation_probability < supervised_threshold) & validation_if_flag,
        1.0,
        validation_probability,
    )
    validation_candidates.append(
        (
            "isolation_forest",
            None,
            business_cost(
                "validation_isolation_forest_override",
                y_validation,
                validation_if_probability,
                validation_amounts,
                supervised_threshold,
            ),
        )
    )
    for metric in metric_passing:
        combined_flag = validation_if_flag | (
            autoencoder_validation_scores[metric.name] >= metric.threshold
        )
        combined_probability = np.where(
            (validation_probability < supervised_threshold) & combined_flag,
            1.0,
            validation_probability,
        )
        validation_candidates.append(
            (
                f"isolation_forest+{metric.name}",
                metric,
                business_cost(
                    f"validation_isolation_forest_{metric.name}_override",
                    y_validation,
                    combined_probability,
                    validation_amounts,
                    supervised_threshold,
                ),
            )
        )

    best_strategy, selected, best_validation_cost = min(
        validation_candidates,
        key=lambda item: (item[2].total_cost, item[0]),
    )
    if best_validation_cost.total_cost >= supervised_validation_cost.total_cost:
        best_strategy = "supervised_only"
        selected = None

    novelty_flag = np.zeros_like(y_test, dtype=bool)
    if best_strategy != "supervised_only":
        novelty_flag = test_if >= if_threshold
    if selected is not None:
        novelty_flag |= autoencoder_test_scores[selected.name] >= selected.threshold
    overridden_probability = np.where(
        (test_probability < supervised_threshold) & novelty_flag,
        1.0,
        test_probability,
    )
    override_cost = business_cost(
        "xgboost_novelty_disagreement",
        y_test,
        overridden_probability,
        test["Amount"].to_numpy(dtype=float),
        supervised_threshold,
    )
    held_out_cost_gate_passed = (
        best_strategy != "supervised_only"
        and override_cost.total_cost <= supervised_cost.total_cost
    )
    if best_strategy == "supervised_only":
        research_decision = "NO_NOVELTY_OVERRIDE_SELECTED_ON_VALIDATION"
    elif held_out_cost_gate_passed:
        research_decision = "NOVELTY_OVERRIDE_RETAINS_RESEARCH_VALUE"
    else:
        research_decision = "NOVELTY_OVERRIDE_REJECTED_HELD_OUT_COST_REGRESSION"
    _write_precision_recall_plot(
        output / "novelty-pr-curves.png",
        y_test,
        {"Isolation Forest": test_if, **autoencoder_test_scores},
    )
    if selected is not None:
        _write_latent_plot(
            output / "autoencoder-latent-tsne.png",
            autoencoder_models[selected.name],
            x_test,
            y_test,
            seed=seed,
        )

    import skops.io as sio

    classifier.save_model(output / "xgboost.json")
    sio.dump(scaler, output / "scaler.skops")
    sio.dump(isolation_forest, output / "isolation-forest.skops")
    for name, model in autoencoder_models.items():
        sio.dump(model, output / f"{name}.skops")
    model_artifacts = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.suffix in {".json", ".skops"}
    }
    report = PublicFraudReport(
        dataset="ULB credit-card fraud (Kaggle mlg-ulb/creditcardfraud)",
        dataset_sha256=dataset_sha256,
        source_tree_sha256=source_tree_sha256,
        seed=seed,
        row_count=len(frame),
        fraud_count=int(frame["Class"].sum()),
        train_rows=len(train),
        validation_rows=len(validation),
        sealed_test_rows=len(test),
        supervised_pr_auc=round(float(average_precision_score(y_test, test_probability)), 8),
        selection_partition="chronological_validation_only",
        validation_isolation_forest=validation_if_metric,
        validation_autoencoders=validation_autoencoder_metrics,
        isolation_forest=if_metric,
        autoencoders=autoencoder_metrics,
        selected_autoencoder=selected.name if selected is not None else None,
        selected_novelty_strategy=best_strategy,
        autoencoder_metric_gate_passed=bool(metric_passing),
        autoencoder_gate_passed=selected is not None and held_out_cost_gate_passed,
        autoencoder_training=autoencoder_training,
        validation_supervised_cost=supervised_validation_cost,
        validation_novelty_candidates=[item[2] for item in validation_candidates],
        held_out_cost_gate_passed=held_out_cost_gate_passed,
        research_decision=research_decision,
        model_artifacts_sha256=model_artifacts,
        supervised_cost=supervised_cost,
        disagreement_override_cost=override_cost,
    )
    (output / "public-fraud-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def detector_metric(
    name: str, labels: np.ndarray, scores: np.ndarray, threshold: float
) -> DetectorMetric:
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & (labels == 1)))
    return DetectorMetric(
        name=name,
        precision=round(true_positive / max(1, int(predicted.sum())), 8),
        recall=round(true_positive / max(1, int((labels == 1).sum())), 8),
        pr_auc=round(float(average_precision_score(labels, scores)), 8),
        threshold=threshold,
    )


def optimize_business_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    *,
    legitimate_delay_rate: float = 0.01,
) -> BusinessCostMetric:
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.80, 0.9999, 200)))
    return min(
        (
            business_cost(
                "validation",
                labels,
                probabilities,
                amounts,
                float(threshold),
                legitimate_delay_rate=legitimate_delay_rate,
            )
            for threshold in candidates
        ),
        key=lambda metric: (metric.total_cost, metric.threshold),
    )


def business_cost(
    name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    legitimate_delay_rate: float = 0.01,
) -> BusinessCostMetric:
    escalated = probabilities >= threshold
    fraud_loss = float(amounts[(labels == 1) & ~escalated].sum())
    delay_cost = float(amounts[(labels == 0) & escalated].sum() * legitimate_delay_rate)
    return BusinessCostMetric(
        name=name,
        threshold=threshold,
        total_cost=round(fraud_loss + delay_cost, 2),
        fraud_loss=round(fraud_loss, 2),
        legitimate_delay_cost=round(delay_cost, 2),
        escalated_count=int(escalated.sum()),
    )


def _chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    partitions = (
        frame.iloc[:train_end],
        frame.iloc[train_end:validation_end],
        frame.iloc[validation_end:],
    )
    if any(set(partition["Class"].astype(int)) != {0, 1} for partition in partitions):
        raise ValueError("chronological train/validation/test partitions must contain both classes")
    return partitions


def _write_precision_recall_plot(
    path: Path, labels: np.ndarray, scores: dict[str, np.ndarray]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        axis.plot(recall, precision, label=name)
    axis.set(xlabel="Recall", ylabel="Precision", title="Novelty detectors on sealed ULB test")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_latent_plot(
    path: Path,
    model: MLPRegressor,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(seed)
    fraud_indices = np.flatnonzero(labels == 1)
    legitimate_indices = np.flatnonzero(labels == 0)
    legitimate_sample = rng.choice(
        legitimate_indices, min(2000 - len(fraud_indices), len(legitimate_indices)), replace=False
    )
    indices = np.concatenate([fraud_indices, legitimate_sample])
    latent = _bottleneck_activation(model, features[indices])
    embedding = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(indices) // 20)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(latent)
    figure, axis = plt.subplots(figsize=(7, 5))
    selected_labels = labels[indices]
    axis.scatter(
        embedding[selected_labels == 0, 0],
        embedding[selected_labels == 0, 1],
        s=8,
        alpha=0.25,
        label="Legitimate",
    )
    axis.scatter(
        embedding[selected_labels == 1, 0],
        embedding[selected_labels == 1, 1],
        s=18,
        alpha=0.8,
        label="Fraud",
    )
    axis.set(title="Selected autoencoder bottleneck (t-SNE)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _bottleneck_activation(model: MLPRegressor, features: np.ndarray) -> np.ndarray:
    hidden = tuple(int(width) for width in model.hidden_layer_sizes)
    bottleneck_layer = int(np.argmin(hidden))
    values = features
    for index in range(bottleneck_layer + 1):
        values = np.maximum(0.0, values @ model.coefs_[index] + model.intercepts_[index])
    return values


def report_summary(report: PublicFraudReport) -> dict[str, Any]:
    return report.model_dump(mode="json")
