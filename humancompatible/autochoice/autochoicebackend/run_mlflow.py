"""
run_mlflow
====

Hydra-only MLflow init, dataset dispatch (adult|compas|custom), FLAML tuning,
MAPIE conformal prediction, AIF360 fairness pre/post metrics logging — plus:

- FACTS-only experiment for bias detection (dataset-agnostic).
- GLANCE experiment for global counterfactual actions (dataset-agnostic).
- DETECT experiment (MSD / L∞ histogram gap) for subgroup bias discovery.

Usage
-----
Regular pipeline (with pre/post-processing + FLAML):

    python run_mlflow.py <dataset_name> <preprocessing_name> <postprocessing_name> [search_algo]

FACTS bias scan only (skip pre/post): pass 'FACTS' as the 3rd arg:

    python run_mlflow.py <dataset_name> _ FACTS

GLANCE global actions only (skip pre/post): pass 'GLANCE' as the 3rd arg:

    python run_mlflow.py <dataset_name> _ GLANCE

DETECT (MSD / L∞) bias scan only (skip pre/post): pass 'DETECT' as the 3rd arg:

    python run_mlflow.py <dataset_name> _ DETECT
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple, Union

import mlflow
import mlflow.sklearn as msk
import numpy as np
import pandas as pd
import torch
from flaml import AutoML
from hyperopt import Trials, fmin, hp, rand, tpe
from mapie.classification import MapieClassifier
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from aif360.algorithms.postprocessing import (
    CalibratedEqOddsPostprocessing,
    EqOddsPostprocessing,
    RejectOptionClassification,
)
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric
from aif360.sklearn.detectors.facts import FACTS_bias_scan

# GLANCE (Global Actions)
from humancompatible.explain.glance.iterative_merges.iterative_merges import (  # type: ignore
    C_GLANCE,
    format_glance_output,
)

# DETECT (MSD & L∞ helpers)
from humancompatible.detect.helpers.utils import (  # type: ignore
    detect_and_score,
    evaluate_subgroup_discrepancy,
    signed_subgroup_discrepancy,
)
from humancompatible.detect.methods.msd.mapping_msd import (  # type: ignore
    subgroup_map_from_conjuncts_dataframe,
)

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from data_helper import (
    init_mlflow_from_cfg,
    load_compas_dataset,
    load_custom_dataset,
    load_openml_adult,
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
    """
    Apply AIF360 preprocessing.

    Parameters
    ----------
    preprocessing_name : str
        Currently only ``"Reweighing"`` is supported.
    dataset : StandardDataset
        Input dataset to transform.

    Returns
    -------
    StandardDataset
        Transformed dataset with instance weights.

    Raises
    ------
    ValueError
        If ``preprocessing_name`` is unknown.
    """
    if preprocessing_name == "Reweighing":
        return Reweighing(privileged_groups, unprivileged_groups).fit_transform(dataset)
    raise ValueError(f"Unsupported preprocessing method: {preprocessing_name}")


def apply_postprocessing(
    postprocessing_name: str,
    dataset_true: StandardDataset,
    dataset_pred: StandardDataset,
) -> StandardDataset:
    """
    Apply AIF360 post-processing mitigation.

    Parameters
    ----------
    postprocessing_name : str
        One of ``"EqOddsPostprocessing"``, ``"CalibratedEqOddsPostprocessing"``,
        or ``"RejectOptionClassification"``.
    dataset_true : StandardDataset
        The ground-truth dataset.
    dataset_pred : StandardDataset
        Model predictions on the same features.

    Returns
    -------
    StandardDataset
        Post-processed predictions consistent with the chosen mitigator.

    Raises
    ------
    ValueError
        If ``postprocessing_name`` is unknown.
    """
    if postprocessing_name == "EqOddsPostprocessing":
        model = EqOddsPostprocessing(privileged_groups, unprivileged_groups)
    elif postprocessing_name == "CalibratedEqOddsPostprocessing":
        model = CalibratedEqOddsPostprocessing(
            privileged_groups, unprivileged_groups, cost_constraint="fnr"
        )
    elif postprocessing_name == "RejectOptionClassification":
        model = RejectOptionClassification(
            privileged_groups,
            unprivileged_groups,
            low_class_thresh=0.01,
            high_class_thresh=0.99,
            num_class_thresh=100,
            num_ROC_margin=50,
            metric_name="Statistical parity difference",
            metric_ub=0.05,
            metric_lb=-0.05,
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
    """
    Compute and log fairness metrics to MLflow.

    Parameters
    ----------
    dataset_before : StandardDataset
        Reference dataset (ground truth or preprocessed).
    dataset_after : StandardDataset
        Dataset to evaluate (same features, different labels).
    prefix : str, optional
        Metric key prefix (e.g., ``"preprocessed_"``), by default ``""``.

    Notes
    -----
    Logs common classification fairness metrics from AIF360.
    """
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
# Small utilities
# ---------------------------------------------------------------------------

def _detect_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect categorical and numeric feature names.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix.

    Returns
    -------
    cat_cols : list of str
        Columns considered categorical (``object`` or ``category`` dtypes).
    num_cols : list of str
        Columns considered numeric (NumPy number subtypes).

    Notes
    -----
    This is a lightweight detector used both by FACTS/GLANCE and the baseline model.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols


def _ensure_interval_features_for_facts(
    X: pd.DataFrame,
    cfg: DictConfig
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[float]], List[str]]:
    """
    Convert configured continuous features to interval categories (FACTS-friendly).

    Parameters
    ----------
    X : pandas.DataFrame
        Raw features as a DataFrame.
    cfg : DictConfig
        Hydra configuration with an optional ``facts`` section.

    Returns
    -------
    Xc : pandas.DataFrame
        Updated features with specified columns converted to interval categories.
    cat_cols : list of str
        Detected categorical columns after conversion.
    bins_used_map : dict
        Mapping ``col -> bin edges`` used, finite-friendly for logging.
    created_cols : list of str
        Any synthesized interval columns.

    Configuration
    -------------
    facts.ensure_interval_cols : list[str], default ``["age"]``
        Names to coerce to interval categories if present.
    facts.create_missing_interval_cols : bool, default ``True``
        If a required column is missing (e.g. ``"age"``), synthesize it.
    facts.default_n_bins : int, default ``7``
        Number of quantile bins when edges are not provided.
    facts.interval_bins : dict[str, list[float]]
        Optional manual bin edges per column.
    """
    Xc = X.copy()
    facts_cfg = cfg.get("facts", {}) if hasattr(cfg, "get") else {}
    ensure_cols = list(facts_cfg.get("ensure_interval_cols", ["age"]))
    create_missing = bool(facts_cfg.get("create_missing_interval_cols", True))
    default_n_bins = int(facts_cfg.get("default_n_bins", 7))
    interval_bins = facts_cfg.get("interval_bins", {}) or {}

    bins_used_map: Dict[str, List[float]] = {}
    created_cols: List[str] = []

    for col in ensure_cols:
        if col in Xc.columns:
            s = Xc[col]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                if pd.api.types.is_datetime64_any_dtype(s):
                    s_num = pd.to_datetime(s, errors="coerce").view("int64")
                else:
                    s_num = pd.to_numeric(s, errors="coerce")

                bins = interval_bins.get(col)
                if not bins:
                    qs = np.linspace(0, 1, default_n_bins + 1)
                    edges = np.unique(np.nanquantile(s_num, qs))
                    if len(edges) < 2:
                        edges = np.array([-np.inf, np.inf])
                    bins = edges.tolist()

                bins_used_map[col] = [
                    float(x) if np.isfinite(x) else (1e18 if x == np.inf else -1e18) for x in bins
                ]
                Xc[col] = pd.cut(s_num, bins=bins).astype("category")
        else:
            if create_missing and col == "age":
                bins = interval_bins.get(col) or [-np.inf, 20, 30, 40, 50, 60, 70, np.inf]
                bins_used_map[col] = [
                    float(x) if np.isfinite(x) else (1e18 if x == np.inf else -1e18) for x in bins
                ]
                synth = pd.Series(np.full(len(Xc), 30), index=Xc.index)
                Xc[col] = pd.cut(synth, bins=bins).astype("category")
                created_cols.append(col)

    cat_cols = Xc.select_dtypes(include=["object", "category"]).columns.tolist()
    return Xc, cat_cols, bins_used_map, created_cols


# ---------------------------------------------------------------------------
# FLAML search space and objective
# ---------------------------------------------------------------------------

search_space = {
    "time_budget": hp.uniform("time_budget", 10, 30),
    "metric": hp.choice("metric", ["accuracy"]),
    "estimator_list": hp.choice("estimator_list", [["lgbm"]]),
}


def objective(params: Dict[str, Any]) -> float:
    """
    Hyperopt objective to train a FLAML model, log metrics, and run post-processing fairness.

    Parameters
    ----------
    params : dict
        Sampled hyperparameters for FLAML.

    Returns
    -------
    float
        Negative accuracy (Hyperopt minimizes).
    """
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
    print(
        pd.DataFrame(
            [["AutoML Optimized", acc, automl_inference_time]],
            columns=["Model", "Accuracy", "Inference Time"],
        )
    )

    # MAPIE Conformal Prediction (multiclass-friendly "score" method)
    try:
        mapie = MapieClassifier(estimator=best_model, method="score")
        mapie.fit(X_train, y_train.ravel())
        y_pred_mapie, _ = mapie.predict(X_test, alpha=0.1)
        coverage = float(np.mean(y_pred_mapie != -1))
        mlflow.log_metric("conformal_coverage", coverage)
    except Exception as e:
        print(f"❌ MAPIE failed: {repr(e)}")
        mlflow.log_metric("conformal_coverage", -1.0)

    # Post-processing fairness (AIF360)
    pred_dataset = preprocessed_data.copy()
    pred_dataset.labels = best_model.predict(preprocessed_data.features).reshape(-1, 1)
    postprocessed_ds = apply_postprocessing(postprocessing_name, preprocessed_data, pred_dataset)
    log_fairness_metrics(preprocessed_data, postprocessed_ds, prefix="postprocessed_")

    return -acc


# ---------------------------------------------------------------------------
# Experiment runner and dataset dispatch
# ---------------------------------------------------------------------------

def run_experiment(preprocessing_name_arg: str, postprocessing_name_arg: str, search_algo: str) -> None:
    """
    Run the baseline ML experiment with FLAML and AIF360 mitigation.

    Parameters
    ----------
    preprocessing_name_arg : str
        Preprocessing algorithm name (currently ``"Reweighing"``).
    postprocessing_name_arg : str
        Postprocessing algorithm name.
    search_algo : str
        Hyperopt search algorithm: ``"tpe"`` or ``"rand"``.

    Notes
    -----
    Logs model accuracy, MAPIE coverage, and pre/post fairness metrics to MLflow.
    """
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
            stratify=preprocessed_data.labels.ravel()
            if len(np.unique(preprocessed_data.labels.ravel())) > 1
            else None,
        )

        log_fairness_metrics(aif_data, aif_data, prefix="raw_")
        log_fairness_metrics(preprocessed_data, preprocessed_data, prefix="preprocessed_")

        trials = Trials()
        algo = tpe.suggest if search_algo == "tpe" else rand.suggest
        best_params = fmin(fn=objective, space=search_space, algo=algo, max_evals=1, trials=trials)

        for key, value in best_params.items():
            mlflow.log_param(key, str(value))


def load_dataset_dispatch(
    dataset_name: str, cfg: DictConfig
) -> Tuple[StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """
    Load and return an AIF360 dataset by name.

    Parameters
    ----------
    dataset_name : str
        One of ``"custom"``, ``"adult"``, ``"compas"``.
    cfg : DictConfig
        Hydra configuration.

    Returns
    -------
    aif_data : StandardDataset
        AIF360 dataset object.
    privileged_groups : list of dict
        Privileged group spec(s).
    unprivileged_groups : list of dict
        Unprivileged group spec(s).
    target_label : str
        The canonical target label name.

    Raises
    ------
    ValueError
        If ``dataset_name`` is unknown.
    """
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
        return (
            info["aif_data"],
            info["privileged_groups"],
            info["unprivileged_groups"],
            "income_binary",
        )

    if name == "compas":
        aif_ds, pg, ug, target_label = load_compas_dataset()
        return aif_ds, pg, ug, target_label

    raise ValueError(f"Unknown dataset '{dataset_name}'. Use one of: custom, adult, compas.")


# ---------------------------------------------------------------------------
# FACTS-only bias detection (dataset-agnostic)
# ---------------------------------------------------------------------------

def run_facts_experiment(dataset_name: str, cfg: DictConfig) -> None:
    """
    Run an AIF360 FACTS bias scan on a simple binary classifier.

    This is dataset-agnostic and will:
      * Convert configured continuous features to interval categories.
      * Binarize labels if needed.
      * Train a baseline logistic regression with one-hot encoding.
      * Run ``FACTS_bias_scan`` on held-out data and log ranked subgroups.

    Parameters
    ----------
    dataset_name : str
        Dataset key (``"custom"``, ``"adult"``, ``"compas"``).
    cfg : DictConfig
        Hydra configuration (may contain a ``facts`` section).

    Logs
    ----
    MLflow params:
        Protected attribute, favorable value, any synthesized columns, bin edges per column.
    MLflow metrics:
        ``facts_unfairness_rank_k`` per ranked subgroup.
    MLflow artifacts:
        ``facts_top_groups.json`` with the complete ranking payload.
    """
    # Load AIF dataset and convert back to pandas
    aif, _, _, _ = load_dataset_dispatch(dataset_name, cfg)
    df, attrs = aif.convert_to_dataframe(de_dummy_code=True)
    label_name = attrs["label_names"][0]

    # Choose protected attribute for scanning
    facts_cfg = cfg.get("facts", {}) if hasattr(cfg, "get") else {}
    pattrs = attrs.get("protected_attribute_names") or []
    prot_attr_name = facts_cfg.get("protected_attribute") or (pattrs[0] if pattrs else None)
    if prot_attr_name is None or prot_attr_name not in df.columns:
        raise ValueError(
            "FACTS requires a 'protected_attribute' present in the dataframe. "
            "Set cfg.facts.protected_attribute to a valid column name."
        )

    # Features & labels (binarize if needed)
    X_raw = df.drop(columns=[label_name])
    y_raw = df[label_name]

    if y_raw.nunique() == 2:
        try:
            y = y_raw.astype(int)
            favorable_val_logged = "as-is-binary"
        except Exception:
            vals = list(y_raw.unique())
            y = (y_raw == vals[0]).astype(int)
            favorable_val_logged = f"binary-mapped:{vals[0]}=1"
    else:
        favorable_val = facts_cfg.get("favorable_label_value")
        if favorable_val is None:
            try:
                favorable_val = pd.to_numeric(y_raw, errors="coerce").max()
            except Exception:
                favorable_val = y_raw.value_counts().idxmax()
        y = (y_raw == favorable_val).astype(int)
        favorable_val_logged = str(favorable_val)

    # Make features FACTS-compatible
    X, cat_cols, bins_used_map, created_cols = _ensure_interval_features_for_facts(X_raw, cfg)

    # Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline model
    preproc = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    clf = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1500))])

    # Temporarily disable sklearn autolog (Interval dtype vs schema)
    _reenable_autolog = False
    if (cfg.get("mlflow", {}) or {}).get("autolog", True):
        _reenable_autolog = True
        msk.autolog(disable=True)

    try:
        with mlflow.start_run(run_name=f"FACTS_{dataset_name}"):
            # Train model
            clf.fit(X_tr, y_tr)

            # FACTS knobs
            facts_metric = facts_cfg.get("metric", "equal-effectiveness")
            top_k = int(facts_cfg.get("top_count", 5))
            min_supp = float(facts_cfg.get("freq_itemset_min_supp", 0.10))
            feats_allowed = facts_cfg.get("feats_allowed_to_change", None)
            feats_frozen = facts_cfg.get("feats_not_allowed_to_change", None)
            feature_weights = facts_cfg.get("feature_weights", {}) or {}

            # Log setup
            mlflow.log_param("facts_metric", facts_metric)
            mlflow.log_param("facts_top_count", top_k)
            mlflow.log_param("facts_min_support", min_supp)
            mlflow.log_param("facts_protected_attr", prot_attr_name)
            mlflow.log_param("facts_favorable_value_for_binarization", favorable_val_logged)
            if created_cols:
                mlflow.log_param("facts_synth_interval_cols", ",".join(created_cols))
            for col, edges in bins_used_map.items():
                mlflow.log_param(f"facts_bins.{col}", str(edges))

            # Run scan
            results = FACTS_bias_scan(
                X=X_te,
                clf=clf,
                prot_attr=prot_attr_name,
                metric=facts_metric,
                categorical_features=cat_cols,
                freq_itemset_min_supp=min_supp,
                feature_weights=feature_weights,
                feats_allowed_to_change=feats_allowed,
                feats_not_allowed_to_change=feats_frozen,
                top_count=top_k,
                verbose=True,
                print_recourse_report=False,
            )

            # Log ranked groups
            biased_groups: List[Dict[str, Any]] = []
            for rank, (group_desc, value) in enumerate(results, start=1):
                mlflow.log_metric(f"facts_unfairness_rank_{rank}", float(value))
                for k, v in group_desc.items():
                    mlflow.log_param(f"facts_group_{rank}.{k}", str(v))
                biased_groups.append({"rank": rank, "unfairness": float(value), "group": group_desc})

            # Artifact for UI
            with open("facts_top_groups.json", "w") as f:
                json.dump(
                    {
                        "dataset": dataset_name,
                        "metric": facts_metric,
                        "protected_attr": prot_attr_name,
                        "groups": biased_groups,
                    },
                    f,
                    indent=2,
                )
            mlflow.log_artifact("facts_top_groups.json")

            print("\n🔎 FACTS — Top biased subgroups:")
            for g in biased_groups:
                print(f"  #{g['rank']:02d}  unfairness={g['unfairness']:.4f}  group={g['group']}")
    finally:
        if _reenable_autolog:
            msk.autolog()  # restore for the rest of the pipeline


# ---------------------------------------------------------------------------
# GLANCE-only global actions (dataset-agnostic)
# ---------------------------------------------------------------------------

def _binarize_labels_for_glance(y_raw: pd.Series, cfg: DictConfig) -> Tuple[pd.Series, str]:
    """
    Binarize labels for GLANCE if necessary.

    Parameters
    ----------
    y_raw : pandas.Series
        Original labels.
    cfg : DictConfig
        Hydra configuration (may contain a ``glance`` section with
        ``favorable_label_value``).

    Returns
    -------
    y : pandas.Series
        Binary labels (0/1).
    favorable_val_logged : str
        Description of how binarization was performed.
    """
    if y_raw.nunique() == 2:
        try:
            return y_raw.astype(int), "as-is-binary"
        except Exception:
            vals = list(y_raw.unique())
            return (y_raw == vals[0]).astype(int), f"binary-mapped:{vals[0]}=1"

    gl_cfg = cfg.get("glance", {}) if hasattr(cfg, "get") else {}
    favorable_val = gl_cfg.get("favorable_label_value")
    if favorable_val is None:
        try:
            favorable_val = pd.to_numeric(y_raw, errors="coerce").max()
        except Exception:
            favorable_val = y_raw.value_counts().idxmax()
    return (y_raw == favorable_val).astype(int), str(favorable_val)


def _coerce_categoricals_for_glance(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Ensure feature dtypes are GLANCE-friendly, returning lists of categorical/numeric columns.

    Parameters
    ----------
    X : pandas.DataFrame
        Raw features.

    Returns
    -------
    Xc : pandas.DataFrame
        Possibly coerced features (e.g., object/category preserved).
    categorical : list of str
        Names of categorical columns.
    numeric : list of str
        Names of numeric columns.
    """
    Xc = X.copy()
    cat_cols, num_cols = _detect_feature_types(Xc)
    return Xc, cat_cols, num_cols


def _serialize_actions_for_artifact(global_actions: Any) -> Union[List[Dict[str, Any]], Any]:
    """
    Convert ``global_actions`` result to a JSON-serializable structure where possible.

    Parameters
    ----------
    global_actions : Any
        Payload returned by ``C_GLANCE.global_actions()``.

    Returns
    -------
    json_like : Any
        A structure safe to dump with ``json.dump``, if feasible; otherwise the original.
    """
    try:
        if isinstance(global_actions, pd.DataFrame):
            return global_actions.to_dict(orient="records")
        if isinstance(global_actions, (list, tuple)):
            def _to_dict(x):
                if isinstance(x, pd.Series):
                    return x.to_dict()
                if isinstance(x, dict):
                    return x
                return {"value": str(x)}
            return [_to_dict(x) for x in global_actions]
        if isinstance(global_actions, dict):
            out = {}
            for k, v in global_actions.items():
                if isinstance(v, pd.Series):
                    out[k] = v.to_dict()
                elif isinstance(v, pd.DataFrame):
                    out[k] = v.to_dict(orient="records")
                else:
                    out[k] = v
            return out
    except Exception:
        pass
    return global_actions


def run_glance_experiment(dataset_name: str, cfg: DictConfig) -> None:
    """
    Run GLANCE (C-GLANCE iterative merges) to derive **global counterfactual actions**.

    The procedure is dataset-agnostic and mirrors the upstream example:
      1. Load an AIF dataset and convert to pandas.
      2. Binarize labels if needed (configurable favorable value).
      3. Train a simple baseline classifier (OneHot + LogisticRegression).
      4. Fit ``C_GLANCE`` with the training split.
      5. Explain the held-out group and log cumulative effectiveness/cost.
      6. Log cluster-wise statistics and global actions as MLflow artifacts.

    Parameters
    ----------
    dataset_name : str
        Dataset key (``"custom"``, ``"adult"``, ``"compas"``).
    cfg : DictConfig
        Hydra configuration with an optional ``glance`` section.

    Logs
    ----
    MLflow params:
        Initial/final clusters, local CF generator, action-choice algorithm, favorable value.
    MLflow metrics:
        ``glance_total_effectiveness``, ``glance_total_cost``.
    MLflow artifacts:
        ``glance_clusters.json`` (per-cluster stats) and ``glance_global_actions.json`` (final actions).
    """
    # --- Load & prepare ---
    aif, _, _, _ = load_dataset_dispatch(dataset_name, cfg)
    df, attrs = aif.convert_to_dataframe(de_dummy_code=True)
    label_name = attrs["label_names"][0]

    X_raw = df.drop(columns=[label_name])
    y_raw = df[label_name]

    y, favorable_val_logged = _binarize_labels_for_glance(y_raw, cfg)
    X, cat_cols, num_cols = _coerce_categoricals_for_glance(X_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline predictive model for GLANCE
    preproc = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    base_clf = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1500))])
    base_clf.fit(X_tr, y_tr)

    # --- Read GLANCE config ---
    gl_cfg = cfg.get("glance", {}) if hasattr(cfg, "get") else {}

    initial_clusters = int(gl_cfg.get("initial_clusters", 100))
    final_clusters = int(gl_cfg.get("final_clusters", 10))
    num_local_cfs = int(gl_cfg.get("num_local_counterfactuals", 5))
    heuristic_weights = tuple(gl_cfg.get("heuristic_weights", (0.5, 0.5)))  # (cost, effect)
    alternative_merges = bool(gl_cfg.get("alternative_merges", True))
    random_seed = int(gl_cfg.get("random_seed", 13))
    clustering_method = gl_cfg.get("clustering_method", "KMeans")
    cf_generator = gl_cfg.get("cf_generator", "Dice")
    cluster_action_choice_algo = gl_cfg.get("cluster_action_choice_algo", "max-eff")

    # Local-method-specific optional knobs
    nns__n_scalars = gl_cfg.get("nns__n_scalars")
    rs__n_most_important = gl_cfg.get("rs__n_most_important")
    rs__n_categorical_most_frequent = gl_cfg.get("rs__n_categorical_most_frequent")
    lowcost__action_threshold = gl_cfg.get("lowcost__action_threshold")
    lowcost__num_low_cost = gl_cfg.get("lowcost__num_low_cost")
    min_cost_eff_thres__effectiveness_threshold = gl_cfg.get("min_cost_eff_thres__effectiveness_threshold")
    min_cost_eff_thres_combinations__num_min_cost = gl_cfg.get("min_cost_eff_thres_combinations__num_min_cost")
    eff_thres_hybrid__max_n_actions_full_combinations = gl_cfg.get("eff_thres_hybrid__max_n_actions_full_combinations")

    # --- Fit GLANCE ---
    glance = C_GLANCE(
        model=base_clf,
        initial_clusters=initial_clusters,
        final_clusters=final_clusters,
        num_local_counterfactuals=num_local_cfs,
        heuristic_weights=heuristic_weights,  # (weight_cost, weight_effectiveness)
        alternative_merges=alternative_merges,
        random_seed=random_seed,
        verbose=True,
    )

    glance.fit(
        X=X_tr,
        y=y_tr,
        train_dataset=X_tr,
        feat_to_vary=gl_cfg.get("feat_to_vary", "all"),
        numeric_features_names=num_cols,
        categorical_features_names=cat_cols,
        clustering_method=clustering_method,
        cf_generator=cf_generator,
        cluster_action_choice_algo=cluster_action_choice_algo,
        nns__n_scalars=nns__n_scalars,
        rs__n_most_important=rs__n_most_important,
        rs__n_categorical_most_frequent=rs__n_categorical_most_frequent,
        lowcost__action_threshold=lowcost__action_threshold,
        lowcost__num_low_cost=lowcost__num_low_cost,
        min_cost_eff_thres__effectiveness_threshold=min_cost_eff_thres__effectiveness_threshold,
        min_cost_eff_thres_combinations__num_min_cost=min_cost_eff_thres_combinations__num_min_cost,
        eff_thres_hybrid__max_n_actions_full_combinations=eff_thres_hybrid__max_n_actions_full_combinations,
    )

    # --- Explain held-out group ---
    total_effectiveness, total_cost = glance.explain_group(X_te)

    # Try to format cluster-wise outputs if available on the object
    clusters_stats = getattr(glance, "cluster_results", None)
    if clusters_stats is not None:
        formatted = format_glance_output(clusters_stats, categorical_columns=cat_cols)
    else:
        formatted = {}

    # Global actions (if exposed)
    try:
        global_actions = glance.global_actions()
    except Exception:
        global_actions = None

    # --- Log to MLflow ---
    with mlflow.start_run(run_name=f"GLANCE_{dataset_name}"):
        # Params
        mlflow.log_param("glance_initial_clusters", initial_clusters)
        mlflow.log_param("glance_final_clusters", final_clusters)
        mlflow.log_param("glance_num_local_counterfactuals", num_local_cfs)
        mlflow.log_param("glance_heuristic_weights", str(heuristic_weights))
        mlflow.log_param("glance_alternative_merges", alternative_merges)
        mlflow.log_param("glance_random_seed", random_seed)
        mlflow.log_param("glance_clustering_method", clustering_method)
        mlflow.log_param("glance_cf_generator", cf_generator)
        mlflow.log_param("glance_action_choice_algo", cluster_action_choice_algo)

        # Metrics
        mlflow.log_metric("glance_total_effectiveness", float(total_effectiveness))
        mlflow.log_metric("glance_total_cost", float(total_cost))

        # Artifacts
        with open("glance_clusters.json", "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "cluster_stats": formatted if formatted else (clusters_stats or {}),
                    "categorical_columns": cat_cols,
                    "numeric_columns": num_cols,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact("glance_clusters.json")

        if global_actions is not None:
            ga_json = _serialize_actions_for_artifact(global_actions)
            with open("glance_global_actions.json", "w") as f:
                json.dump({"global_actions": ga_json}, f, indent=2)
            mlflow.log_artifact("glance_global_actions.json")

    print(f"\n🌐 GLANCE — total_effectiveness={total_effectiveness:.4f}, total_cost={total_cost:.4f}")
    if clusters_stats is not None:
        print("Top-level cluster stats logged to MLflow artifact 'glance_clusters.json'.")


# ---------------------------------------------------------------------------
# DETECT-only (MSD / L∞ histogram gap) — dataset-agnostic
# ---------------------------------------------------------------------------

def _choose_protected_and_continuous(
    df: pd.DataFrame,
    attrs: Dict[str, Any],
    cfg: DictConfig,
    label_name: str,
) -> Tuple[List[str], List[str]]:
    """
    Choose protected attributes and continuous-protected columns for DETECT.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataframe (features + label).
    attrs : dict
        Metadata returned by ``StandardDataset.convert_to_dataframe``.
    cfg : DictConfig
        Hydra configuration (may contain a ``detect`` section).
    label_name : str
        Name of the label column (to be excluded).

    Returns
    -------
    protected_list : list of str
        Protected attribute columns to audit.
    continuous_list : list of str
        Subset of ``protected_list`` treated as continuous (binned by DETECT).

    Notes
    -----
    - If ``cfg.detect.protected_attributes`` is provided, it is used.
    - Else fall back to ``attrs['protected_attribute_names']``.
    - Continuous features are inferred as numeric dtypes among the protected list,
      unless ``cfg.detect.continuous_attributes`` overrides.
    """
    dcfg = cfg.get("detect", {}) if hasattr(cfg, "get") else {}
    explicit = dcfg.get("protected_attributes")
    if explicit:
        protected_list = [c for c in explicit if c in df.columns and c != label_name]
    else:
        protected_list = [c for c in (attrs.get("protected_attribute_names") or []) if c in df.columns]
    if not protected_list:
        raise ValueError(
            "DETECT requires at least one protected attribute. "
            "Set cfg.detect.protected_attributes or ensure the AIF dataset exposes them."
        )

    cont_override = dcfg.get("continuous_attributes")
    if cont_override:
        continuous_list = [c for c in cont_override if c in protected_list]
    else:
        continuous_list = [c for c in protected_list if pd.api.types.is_numeric_dtype(df[c])]
    return protected_list, continuous_list


def _pretty_rule(rule: List[Tuple[int, Any]]) -> str:
    """
    Convert a DETECT rule list into a human-readable string.

    Parameters
    ----------
    rule : list[tuple[int, Any]]
        The rule as returned by ``detect_and_score`` (index, Bin) pairs.

    Returns
    -------
    str
        Human-readable conjunction such as ``"Race = Blue AND Age in [30,40)"``.

    Notes
    -----
    The ``Bin`` objects implement ``__str__``; we print each conjunct with AND.
    """
    try:
        return " AND ".join(str(cond) for _, cond in (rule or [])) or "(empty subgroup)"
    except Exception:
        return str(rule)


def run_detect_experiment(dataset_name: str, cfg: DictConfig) -> None:
    """
    Run HumanCompatible.Detect to find the most biased subgroup and score it.

    The procedure mirrors the quick-start usage and helper API:
      1. Load an AIF dataset, convert to pandas.
      2. Select protected attributes (from config or dataset metadata).
      3. Call ``detect_and_score`` with method ``"MSD"`` (or ``"l_inf"``).
      4. Build a boolean mask of the subgroup and re-evaluate signed/absolute gaps.
      5. Log result and rule to MLflow + save rule artifact.

    Parameters
    ----------
    dataset_name : str
        Dataset key (``"custom"``, ``"adult"``, ``"compas"``).
    cfg : DictConfig
        Hydra configuration with an optional ``detect`` section:

        - ``method``: ``"MSD"`` (default) or ``"l_inf"``.
        - ``seed``: random seed for subsampling/solver (optional).
        - ``n_samples``: cap on rows for subsampling (default: 1_000_000).
        - ``method_kwargs``: dict with MSD knobs (e.g., ``time_limit``, ``n_min``, ``solver``).
        - ``protected_attributes``: override list of protected columns (optional).
        - ``continuous_attributes``: subset of protected treated as continuous (optional).

    Logs
    ----
    MLflow params:
        method, protected_list, continuous_list, n_samples, seed, method_kwargs.
    MLflow metrics:
        ``detect_value`` (MSD or L∞), ``detect_abs_delta``, ``detect_signed_delta``,
        ``detect_support``, ``detect_support_frac``.
    MLflow artifact:
        ``detect_rule.json`` with both raw and human-readable rule.
    """
    # --- Load dataset & pick columns ---
    aif, _, _, _ = load_dataset_dispatch(dataset_name, cfg)
    df, attrs = aif.convert_to_dataframe(de_dummy_code=True)
    label_name = attrs["label_names"][0]

    protected_list, continuous_list = _choose_protected_and_continuous(df, attrs, cfg, label_name)

    # --- Prepare inputs for detect_and_score ---
    X = df[protected_list]
    y = df[label_name]

    dcfg = cfg.get("detect", {}) if hasattr(cfg, "get") else {}
    method = str(dcfg.get("method", "MSD"))
    n_samples = int(dcfg.get("n_samples", 1_000_000))
    seed = dcfg.get("seed", None)
    method_kwargs = dcfg.get("method_kwargs", None)

    # --- Run detection ---
    # detect_and_score returns (rule, value); rule is list[(feature_index, Bin)]
    rule, value = detect_and_score(
        X=X,
        y=y,
        protected_list=protected_list,
        continuous_list=continuous_list,
        fp_map=None,
        seed=seed,
        n_samples=n_samples,
        method=method,
        method_kwargs=method_kwargs,
    )

    pretty = _pretty_rule(rule)

    # Build subgroup mask & recompute signed/absolute deltas
    mask = subgroup_map_from_conjuncts_dataframe(rule, X)
    abs_delta = float(evaluate_subgroup_discrepancy(mask, y.to_numpy().astype(bool)))
    signed_delta = float(signed_subgroup_discrepancy(mask, y.to_numpy().astype(bool)))
    support = int(mask.sum())
    support_frac = float(support / len(mask)) if len(mask) else 0.0

    # --- Log to MLflow ---
    with mlflow.start_run(run_name=f"DETECT_{dataset_name}"):
        mlflow.log_param("detect_method", method)
        mlflow.log_param("detect_protected_list", ",".join(protected_list))
        mlflow.log_param("detect_continuous_list", ",".join(continuous_list))
        mlflow.log_param("detect_n_samples", n_samples)
        if seed is not None:
            mlflow.log_param("detect_seed", seed)
        if method_kwargs:
            mlflow.log_param("detect_method_kwargs", json.dumps(method_kwargs))

        mlflow.log_metric("detect_value", float(value))
        mlflow.log_metric("detect_abs_delta", abs_delta)
        mlflow.log_metric("detect_signed_delta", signed_delta)
        mlflow.log_metric("detect_support", support)
        mlflow.log_metric("detect_support_frac", support_frac)

        # Persist the rule
        raw_rule = [
            {"feature_index": int(idx), "bin": str(binop)} for idx, binop in (rule or [])
        ]
        with open("detect_rule.json", "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "protected_list": protected_list,
                    "continuous_list": continuous_list,
                    "value": float(value),
                    "abs_delta": abs_delta,
                    "signed_delta": signed_delta,
                    "support": support,
                    "support_frac": support_frac,
                    "rule_human": pretty,
                    "rule_raw": raw_rule,
                },
                f,
                indent=2,
            )
        mlflow.log_artifact("detect_rule.json")

    print(f"\n🧭 DETECT — method={method} value={value:.4f}")
    print(f"Subgroup: {pretty}  (support={support}, signed_delta={signed_delta:.4f})")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def _load_cfg() -> DictConfig:
    """
    Load Hydra config from ``.`` or ``./humancompatible``.

    Returns
    -------
    DictConfig
        Composed Hydra configuration.

    Raises
    ------
    FileNotFoundError
        If no ``config.yaml`` is found in either search location.
    """
    GlobalHydra.instance().clear()
    if os.path.exists("config.yaml"):
        with initialize(version_base=None, config_path="."):
            return compose(config_name="config")
    elif os.path.exists(os.path.join("humancompatible", "config.yaml")):
        with initialize(version_base=None, config_path="humancompatible"):
            return compose(config_name="config")
    raise FileNotFoundError(
        "config.yaml not found in '.' or 'humancompatible/'. "
        "Place your single config file in one of those locations."
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python run_mlflow.py <dataset_name> <preprocessing_name> <postprocessing_name> [search_algo]\n"
            "  dataset_name: custom | adult | compas\n"
            "  preprocessing_name: Reweighing | _ (ignored for FACTS/GLANCE/DETECT)\n"
            "  postprocessing_name: EqOddsPostprocessing | CalibratedEqOddsPostprocessing | RejectOptionClassification | FACTS | GLANCE | DETECT\n"
            "  search_algo: tpe | rand   (default: tpe)\n\n"
            "Examples:\n"
            "  python run_mlflow.py custom Reweighing EqOddsPostprocessing tpe\n"
            "  python run_mlflow.py adult _ FACTS\n"
            "  python run_mlflow.py compas _ GLANCE\n"
            "  python run_mlflow.py adult _ DETECT\n"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    preprocessing_name = sys.argv[2]
    postprocessing_name = sys.argv[3] if len(sys.argv) > 3 else "EqOddsPostprocessing"
    search_algo = sys.argv[4] if len(sys.argv) > 4 else "tpe"

    cfg: DictConfig = _load_cfg()

    # MLflow init from Hydra config
    init_mlflow_from_cfg(cfg)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # FACTS-only path
    if postprocessing_name.strip().upper() == "FACTS":
        run_facts_experiment(dataset_name, cfg)
        sys.exit(0)

    # GLANCE-only path
    if postprocessing_name.strip().upper() == "GLANCE":
        run_glance_experiment(dataset_name, cfg)
        sys.exit(0)

    # DETECT-only path (MSD / L∞)
    if postprocessing_name.strip().upper() == "DETECT":
        run_detect_experiment(dataset_name, cfg)
        sys.exit(0)

    # Regular path with mitigation + FLAML
    aif_data, privileged_groups, unprivileged_groups, TARGET_LABEL = load_dataset_dispatch(dataset_name, cfg)
    print(f"\n🎯 Using target label: '{TARGET_LABEL}'")
    print(f"🛡️ Privileged groups: {privileged_groups}")
    print(f"🛡️ Unprivileged groups: {unprivileged_groups}")

    run_experiment(preprocessing_name, postprocessing_name, search_algo)
