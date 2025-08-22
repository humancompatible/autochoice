"""
main
====

Hydra-only MLflow init, dataset dispatch (adult|compas|custom), FLAML tuning,
MAPIE conformal prediction, and AIF360 fairness post/post metrics logging.

Usage:
    python main.py <dataset_name> <preprocessing_name> <postprocessing_name> [search_algo]

Examples:
    python main.py custom Reweighing EqOddsPostprocessing tpe
    python main.py adult Reweighing CalibratedEqOddsPostprocessing rand
    python main.py compas Reweighing RejectOptionClassification
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import mlflow
from flaml import AutoML
from hyperopt import fmin, tpe, rand, hp, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mapie.classification import MapieClassifier

from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import (
    EqOddsPostprocessing,
    CalibratedEqOddsPostprocessing,
    RejectOptionClassification,
)
from aif360.metrics import ClassificationMetric

from hydra import initialize, compose
from omegaconf import DictConfig

from data_helper import (
    init_mlflow_from_cfg,
    load_custom_dataset,
    load_openml_adult,
    load_compas_dataset,
)

# -------------------------
# GPU / environment info
# -------------------------
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"\U0001F680 GPU Available: {torch.cuda.get_device_name(0)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("⚠️ No GPU detected, using CPU.")

print("MLFLOW version: ")
print(mlflow.__version__)


# ---------------------------------------------------------------------------
# Fairness utilities
# ---------------------------------------------------------------------------

def apply_preprocessing(preprocessing_name: str, dataset: StandardDataset) -> StandardDataset:
    """Apply AIF360 preprocessing (currently Reweighing)."""
    if preprocessing_name == "Reweighing":
        return Reweighing(privileged_groups, unprivileged_groups).fit_transform(dataset)
    raise ValueError(f"Unsupported preprocessing method: {preprocessing_name}")


def apply_postprocessing(
    postprocessing_name: str,
    dataset_true: StandardDataset,
    dataset_pred: StandardDataset,
) -> StandardDataset:
    """Apply AIF360 postprocessing."""
    if postprocessing_name == "EqOddsPostprocessing":
        model = EqOddsPostprocessing(privileged_groups, unprivileged_groups)
    elif postprocessing_name == "CalibratedEqOddsPostprocessing":
        model = CalibratedEqOddsPostprocessing(privileged_groups, unprivileged_groups, cost_constraint="fnr")
    elif postprocessing_name == "RejectOptionClassification":
        model = RejectOptionClassification(
            privileged_groups, unprivileged_groups,
            low_class_thresh=0.01, high_class_thresh=0.99,
            num_class_thresh=100, num_ROC_margin=50,
            metric_name="Statistical parity difference", metric_ub=0.05, metric_lb=-0.05
        )
    else:
        raise ValueError(f"Unsupported postprocessing method: {postprocessing_name}")

    model.fit(dataset_true, dataset_pred)
    return model.predict(dataset_pred)


def log_fairness_metrics(
    dataset_before: StandardDataset,
    dataset_after: StandardDataset,
    prefix: str = "",
) -> None:
    """Compute and log fairness metrics to MLflow."""
    metric = ClassificationMetric(dataset_before, dataset_after, unprivileged_groups, privileged_groups)
    metrics = {
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "disparate_impact": metric.disparate_impact(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "average_odds_difference": metric.average_odds_difference(),
        "theil_index": metric.theil_index(),
    }
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}", float(v))
        print(f"{prefix}{k}: {v:.4f}")


# ---------------------------------------------------------------------------
# FLAML search space and objective
# ---------------------------------------------------------------------------

search_space = {
    "time_budget": hp.uniform("time_budget", 10, 30),
    "metric": hp.choice("metric", ["accuracy"]),
    "estimator_list": hp.choice("estimator_list", [["lgbm"]]),
}


def objective(params: Dict[str, Any]) -> float:
    """Hyperopt objective: train FLAML model, log metrics, fairness postprocess."""
    automl_settings = {
        "time_budget": int(params["time_budget"]),
        "task": "classification",
        "estimator_list": params["estimator_list"],
        "metric": params["metric"],
        "verbose": 0,
    }
    if USE_GPU:
        automl_settings["use_gpu"] = True

    automl = AutoML()
    automl.fit(X_train, y_train, **automl_settings)
    best_model = automl.model

    if best_model is None:
        raise RuntimeError("❌ AutoML failed to return a model.")

    start = time.time()
    _ = best_model.predict(X_test)
    automl_inference_time = time.time() - start

    acc = accuracy_score(y_test, best_model.predict(X_test))
    mlflow.log_metric("automl_inference_time", float(automl_inference_time))
    mlflow.log_metric("automl_accuracy", float(acc))

    print("\n📈 Model Comparison:")
    print(pd.DataFrame([["AutoML Optimized", acc, automl_inference_time]],
                       columns=["Model", "Accuracy", "Inference Time"]))

    try:
        mapie = MapieClassifier(estimator=best_model, method="score")
        mapie.fit(X_train, y_train.ravel())
        y_pred_mapie, _ = mapie.predict(X_test, alpha=0.1)
        coverage = float(np.mean(y_pred_mapie != -1))
        mlflow.log_metric("conformal_coverage", coverage)
    except Exception as e:
        print(f"❌ MAPIE failed: {repr(e)}")
        mlflow.log_metric("conformal_coverage", -1.0)

    pred_dataset = preprocessed_data.copy()
    pred_dataset.labels = best_model.predict(preprocessed_data.features).reshape(-1, 1)
    postprocessed_ds = apply_postprocessing(postprocessing_name, preprocessed_data, pred_dataset)
    log_fairness_metrics(preprocessed_data, postprocessed_ds, prefix="postprocessed_")

    return -acc


# ---------------------------------------------------------------------------
# Experiment runner and dataset dispatch
# ---------------------------------------------------------------------------

def run_experiment(preprocessing_name_arg: str, postprocessing_name_arg: str, search_algo: str) -> None:
    """Run a single MLflow experiment and log fairness metrics."""
    with mlflow.start_run():
        mlflow.log_param("preprocessing_algorithm", preprocessing_name_arg)
        mlflow.log_param("postprocessing_algorithm", postprocessing_name_arg)

        global preprocessed_data, X_train, X_test, y_train, y_test
        preprocessed_data = apply_preprocessing(preprocessing_name_arg, aif_data)

        X_train, X_test, y_train, y_test = train_test_split(
            preprocessed_data.features,
            preprocessed_data.labels.ravel(),
            test_size=0.2,
            random_state=42,
            stratify=preprocessed_data.labels.ravel() if len(np.unique(preprocessed_data.labels.ravel())) > 1 else None,
        )

        log_fairness_metrics(aif_data, aif_data, prefix="raw_")
        log_fairness_metrics(preprocessed_data, preprocessed_data, prefix="preprocessed_")

        trials = Trials()
        algo = tpe.suggest if search_algo == "tpe" else rand.suggest
        best_params = fmin(fn=objective, space=search_space, algo=algo, max_evals=1, trials=trials)

        for key, value in best_params.items():
            mlflow.log_param(key, str(value))


def load_dataset_dispatch(dataset_name: str, cfg: DictConfig) -> Tuple[StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """Dispatch dataset loading by name: custom | adult | compas."""
    name = dataset_name.lower()

    if name == "custom":
        dataset_path = cfg.get("data", {}).get("path", "/data/dataset1M.parquet")
        function_filter = cfg.get("data", {}).get("function_filter_value", "Legal")
        protected_attr = cfg.get("data", {}).get("protected_attribute", "experiences_no")
        protected_threshold = int(cfg.get("data", {}).get("protected_threshold", 2))

        _, aif, pg, ug, target_label = load_custom_dataset(
            dataset_path=dataset_path,
            function_filter_value=function_filter,
            target_from="automatch_score",
            target_bins=(0.6, 0.85, np.inf),
            target_label_name="target_class",
            protected_attribute=protected_attr,
            protected_threshold=protected_threshold,
        )
        return aif, pg, ug, target_label

    if name == "adult":
        info = load_openml_adult(
            random_seed=42,
            train_size=0.7,
            build_pipeline=False,
            return_aif360=True,
            protected_for_aif=cfg.get("data", {}).get("adult_protected_attr", "sex"),
            favorable_label_for_aif=cfg.get("data", {}).get("adult_favorable_label", ">50K"),
        )
        return info["aif_data"], info["privileged_groups"], info["unprivileged_groups"], "income_binary"

    if name == "compas":
        info = load_compas_dataset()
        return info

    raise ValueError(f"Unknown dataset '{dataset_name}'. Use one of: custom, adult, compas.")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python main.py <dataset_name> <preprocessing_name> <postprocessing_name> [search_algo]\n"
            "  dataset_name: custom | adult | compas\n"
            "  preprocessing_name: Reweighing\n"
            "  postprocessing_name: EqOddsPostprocessing | CalibratedEqOddsPostprocessing | RejectOptionClassification\n"
            "  search_algo: tpe | rand   (default: tpe)"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    preprocessing_name = sys.argv[2]
    postprocessing_name = sys.argv[3]
    search_algo = sys.argv[4] if len(sys.argv) > 4 else "tpe"

    with initialize(version_base=None, config_path="."):
        cfg: DictConfig = compose(config_name="config")

    init_mlflow_from_cfg(cfg)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    aif_data, privileged_groups, unprivileged_groups, TARGET_LABEL = load_dataset_dispatch(dataset_name, cfg)

    print(f"\n🎯 Using target label: '{TARGET_LABEL}'")
    print(f"🛡️ Privileged groups: {privileged_groups}")
    print(f"🛡️ Unprivileged groups: {unprivileged_groups}")

    run_experiment(preprocessing_name, postprocessing_name, search_algo)
