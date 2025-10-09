"""
main
====

Hydra-only MLflow init, dataset dispatch (adult|compas|custom), FLAML tuning,
MAPIE conformal prediction, and AIF360 fairness pre/post metrics logging.

This version also adds:
- a FACTS-only experiment for bias detection (dataset-agnostic preparation).
- a REPAIR toolkit experiment on the Adult dataset (sex as protected attribute),
  with methods: "origin", "barycentre", and "partial" (adapted from the
  03_adult-sex.ipynb example).

Usage
-----
Regular pipeline (with pre/post-processing + FLAML):

    python run_mlflow.py <dataset_name> <preprocessing_name> <postprocessing_name> [search_algo]

FACTS bias scan only (skip pre/post): pass 'FACTS' as the 3rd arg:

    python run_mlflow.py <dataset_name> _ FACTS

REPAIR toolkit: pass 'REPAIR' as the 3rd arg and the method as the 4th:

    python run_mlflow.py adult _ REPAIR <origin|barycentre|partial> [partial_theta]

Examples
--------
    python run_mlflow.py custom Reweighing EqOddsPostprocessing tpe
    python run_mlflow.py adult _ FACTS
    python run_mlflow.py compas _ FACTS
    python run_mlflow.py adult _ REPAIR origin
    python run_mlflow.py adult _ REPAIR barycentre
    python run_mlflow.py adult _ REPAIR partial 1e-3
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

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

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from data_helper import (
    init_mlflow_from_cfg,
    load_compas_dataset,
    load_custom_dataset,
    load_openml_adult,
    # NEW (used by REPAIR path)
    prepare_adult_for_repair,
    choose_x_for_repair,
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
        Currently only "Reweighing" is supported.
    dataset : StandardDataset
        Input dataset to transform.

    Returns
    -------
    StandardDataset
        Transformed dataset with instance weights.
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
    Apply AIF360 postprocessing.

    Parameters
    ----------
    postprocessing_name : str
        One of "EqOddsPostprocessing", "CalibratedEqOddsPostprocessing",
        or "RejectOptionClassification".
    dataset_true : StandardDataset
        The ground-truth dataset.
    dataset_pred : StandardDataset
        Model predictions on the same features.

    Returns
    -------
    StandardDataset
        Postprocessed predictions.
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
        Dataset used as reference (ground truth or preprocessed).
    dataset_after : StandardDataset
        Dataset to evaluate (same features, different labels).
    prefix : str, optional
        Prefix for MLflow metric keys (e.g., "preprocessed_"), by default "".
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
# FLAML search space and objective
# ---------------------------------------------------------------------------

search_space = {
    "time_budget": hp.uniform("time_budget", 10, 30),
    "metric": hp.choice("metric", ["accuracy"]),
    "estimator_list": hp.choice("estimator_list", [["lgbm"]]),
}


def objective(params: Dict[str, Any]) -> float:
    """
    Hyperopt objective: train FLAML model, log metrics, and fairness postprocess.

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

    # MAPIE Conformal Prediction (multiclass-friendly score method)
    try:
        mapie = MapieClassifier(estimator=best_model, method="score")
        mapie.fit(X_train, y_train.ravel())
        y_pred_mapie, _ = mapie.predict(X_test, alpha=0.1)
        coverage = float(np.mean(y_pred_mapie != -1))
        mlflow.log_metric("conformal_coverage", coverage)
    except Exception as e:
        print(f"❌ MAPIE failed: {repr(e)}")
        mlflow.log_metric("conformal_coverage", -1.0)

    # Postprocessing fairness
    pred_dataset = preprocessed_data.copy()
    pred_dataset.labels = best_model.predict(preprocessed_data.features).reshape(-1, 1)
    postprocessed_ds = apply_postprocessing(postprocessing_name, preprocessed_data, pred_dataset)
    log_fairness_metrics(preprocessed_data, postprocessed_ds, prefix="postprocessed_")

    return -acc


# ---------------------------------------------------------------------------
# Experiment runner and dataset dispatch (regular path)
# ---------------------------------------------------------------------------

def run_experiment(preprocessing_name_arg: str, postprocessing_name_arg: str, search_algo: str) -> None:
    """
    Run a single MLflow experiment and log fairness metrics.

    Parameters
    ----------
    preprocessing_name_arg : str
        Preprocessing algorithm name (currently "Reweighing").
    postprocessing_name_arg : str
        Postprocessing algorithm name.
    search_algo : str
        Hyperopt search algorithm: "tpe" or "rand".
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
    Dispatch dataset loading by name.

    Parameters
    ----------
    dataset_name : str
        One of "custom", "adult", "compas".
    cfg : DictConfig
        Hydra config.

    Returns
    -------
    aif_data : StandardDataset
    privileged_groups : list[dict]
    unprivileged_groups : list[dict]
    target_label : str
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

def _ensure_interval_features_for_facts(
    X: pd.DataFrame,
    cfg: DictConfig
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[float]], List[str]]:
    """
    Make features FACTS-compatible in a dataset-agnostic way.
    [...]
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


def run_facts_experiment(dataset_name: str, cfg: DictConfig) -> None:
    """
    Run an AIF360 FACTS bias scan on a simple binary classifier trained on raw features.
    [...]
    """
    # Load AIF dataset and convert back to pandas (preserves categories)
    aif, _, _, _ = load_dataset_dispatch(dataset_name, cfg)
    df, attrs = aif.convert_to_dataframe(de_dummy_code=True)
    label_name = attrs["label_names"][0]

    facts_cfg = cfg.get("facts", {}) if hasattr(cfg, "get") else {}
    pattrs = attrs.get("protected_attribute_names") or []
    prot_attr_name = facts_cfg.get("protected_attribute") or (pattrs[0] if pattrs else None)
    if prot_attr_name is None or prot_attr_name not in df.columns:
        raise ValueError(
            "FACTS requires a 'protected_attribute' present in the dataframe. "
            "Set cfg.facts.protected_attribute to a valid column name."
        )

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

    X, cat_cols, bins_used_map, created_cols = _ensure_interval_features_for_facts(X_raw, cfg)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    clf = Pipeline([("prep", preproc), ("clf", LogisticRegression(max_iter=1500))])

    _reenable_autolog = False
    if (cfg.get("mlflow", {}) or {}).get("autolog", True):
        _reenable_autolog = True
        msk.autolog(disable=True)

    try:
        with mlflow.start_run(run_name=f"FACTS_{dataset_name}"):
            clf.fit(X_tr, y_tr)

            facts_metric = facts_cfg.get("metric", "equal-effectiveness")
            top_k = int(facts_cfg.get("top_count", 5))
            min_supp = float(facts_cfg.get("freq_itemset_min_supp", 0.10))
            feats_allowed = facts_cfg.get("feats_allowed_to_change", None)
            feats_frozen = facts_cfg.get("feats_not_allowed_to_change", None)
            feature_weights = facts_cfg.get("feature_weights", {}) or {}

            mlflow.log_param("facts_metric", facts_metric)
            mlflow.log_param("facts_top_count", top_k)
            mlflow.log_param("facts_min_support", min_supp)
            mlflow.log_param("facts_protected_attr", prot_attr_name)
            mlflow.log_param("facts_favorable_value_for_binarization", favorable_val_logged)
            if created_cols:
                mlflow.log_param("facts_synth_interval_cols", ",".join(created_cols))
            for col, edges in bins_used_map.items():
                mlflow.log_param(f"facts_bins.{col}", str(edges))

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

            biased_groups: List[Dict[str, Any]] = []
            for rank, (group_desc, value) in enumerate(results, start=1):
                mlflow.log_metric(f"facts_unfairness_rank_{rank}", float(value))
                for k, v in group_desc.items():
                    mlflow.log_param(f"facts_group_{rank}.{k}", str(v))
                biased_groups.append({"rank": rank, "unfairness": float(value), "group": group_desc})

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
            msk.autolog()  # restore for the rest of your pipeline


# ---------------------------------------------------------------------------
# REPAIR toolkit (Adult dataset; origin/barycentre/partial)
# ---------------------------------------------------------------------------

def run_repair_experiment_adult(
    method: str,
    cfg: DictConfig,
    *,
    partial_theta: Optional[float] = None,
) -> None:
    """
    Run the humancompatible.repair post-processing experiment on the Adult dataset
    with sex as the protected attribute (S: Male=1, Female=0), adapted from
    examples/03_adult-sex.ipynb.

    Parameters
    ----------
    method : {"origin", "barycentre", "partial"}
        Repair strategy to evaluate.
    cfg : DictConfig
        Hydra config; optional keys under `repair`:
          - var_list: list[str]  (default ["age","capital-gain","capital-loss","education-num"])
          - x_list: list[str]    (default auto-select; else enforce)
          - tv_threshold: float  (default 0.1) for auto-select
          - K: int               (default 200)
          - e: float             (default 0.01)
          - favorable_label: int (default 1)
          - thresh: float|"auto" (default 0.05)
          - n_splits: int        (default 5) random re-splits
          - linspace_range: (low, high) for auto threshold search (default (0.1,0.9))
          - theta_auto: float    (default 1e-3) for auto threshold probe
    partial_theta : float, optional
        If method == "partial", the repair parameter; if None, we’ll try a grid
        [1e-2, 1e-3, 1e-4] like in the notebook.
    """
    # Lazy import to avoid making this a hard dependency unless used
    from sklearn.ensemble import RandomForestClassifier
    from humancompatible.repair.postprocess.proj_postprocess import Projpostprocess

    rep_cfg = cfg.get("repair", {}) if hasattr(cfg, "get") else {}
    var_list = list(rep_cfg.get("var_list", ["age", "capital-gain", "capital-loss", "education-num"]))
    tv_threshold = float(rep_cfg.get("tv_threshold", 0.10))
    K = int(rep_cfg.get("K", 200))
    e = float(rep_cfg.get("e", 0.01))
    favorable_label = int(rep_cfg.get("favorable_label", 1))
    thresh = rep_cfg.get("thresh", 0.05)  # can be float or 'auto'
    n_splits = int(rep_cfg.get("n_splits", 5))
    linspace_range = tuple(rep_cfg.get("linspace_range", (0.1, 0.9)))
    theta_auto = float(rep_cfg.get("theta_auto", 1e-3))

    # Prepare Adult data in the exact format the toolkit expects
    messy, X_np, y_np = prepare_adult_for_repair(
        var_list=var_list,
        protected_attr="sex",
    )

    # Auto-select axes for repair, unless user fixes them
    x_list_cfg = rep_cfg.get("x_list")
    if x_list_cfg:
        x_list = list(x_list_cfg)
        tv_dist = {}  # skip computation if user fixed axes
    else:
        x_list, tv_dist = choose_x_for_repair(var_list, messy, tv_threshold)
        if not x_list:
            # Fallback to the notebook's pair
            x_list = ["age", "education-num"]

    # Log setup
    with mlflow.start_run(run_name=f"REPAIR_adult_{method}"):
        mlflow.log_param("repair_method", method)
        mlflow.log_param("repair_var_list", ",".join(var_list))
        mlflow.log_param("repair_x_list", ",".join(x_list))
        mlflow.log_param("repair_K", K)
        mlflow.log_param("repair_epsilon", e)
        mlflow.log_param("repair_favorable_label", favorable_label)
        mlflow.log_param("repair_thresh", str(thresh))
        mlflow.log_param("repair_n_splits", n_splits)
        mlflow.log_param("repair_linspace_range", str(linspace_range))
        mlflow.log_param("repair_theta_auto", theta_auto)
        if tv_dist:
            # store as a compact string
            mlflow.log_param("repair_tv_dist", {k: float(v) for k, v in tv_dist.items()})

        # Accumulate per-split rows exactly like the notebook’s "report" DataFrame
        report = pd.DataFrame(
            columns=["DI", "f1 macro", "f1 micro", "f1 weighted", "TV distance", "method"]
        )

        var_dim = len(var_list)
        # NOTE: X_np has columns: var_list + ['S','W']; y_np is Y
        rng = np.random.default_rng(42)

        # Build either single-parameter list or small grid for "partial"
        if method == "partial":
            if partial_theta is not None:
                partial_grid = [float(partial_theta)]
            else:
                partial_grid = [1e-2, 1e-3, 1e-4]
        else:
            partial_grid = []

        for split_idx in range(n_splits):
            # random split ~40% test then 30% of the remaining for val (not used by toolkit here)
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size=0.4, random_state=int(rng.integers(0, 1_000_000))
            )

            clf = RandomForestClassifier(max_depth=5, random_state=split_idx).fit(
                X_train[:, :var_dim], y_train
            )

            projpost = Projpostprocess(
                X_test,
                y_test,
                x_list,
                var_list,
                clf,
                K,
                e,
                thresh,
                favorable_label,
                linspace_range=linspace_range,
                theta=theta_auto,
            )

            if method in ("origin", "barycentre"):
                row = projpost.postprocess(method)
                report = pd.concat([report, row], ignore_index=True)
            elif method == "partial":
                for p in partial_grid:
                    row = projpost.postprocess("partial", para=float(p))
                    report = pd.concat([report, row], ignore_index=True)
            else:
                raise ValueError("Unknown repair method. Use one of: origin, barycentre, partial.")

        # Persist the raw table and log summary metrics
        out_csv = f"repair_report_adult_{method}.csv"
        report.to_csv(out_csv, index=False)
        mlflow.log_artifact(out_csv)

        # Aggregate by method string (partial has e.g. 'partial_0.001')
        agg = (
            report.groupby("method")[["DI", "f1 macro", "f1 micro", "f1 weighted", "TV distance"]]
            .mean()
            .reset_index()
        )
        agg_out = f"repair_summary_adult_{method}.csv"
        agg.to_csv(agg_out, index=False)
        mlflow.log_artifact(agg_out)

        # Also log the best method (by DI) as metrics for the run
        best_idx = agg["DI"].idxmax()
        best_row = agg.loc[best_idx]
        mlflow.log_metric("repair_DI_best", float(best_row["DI"]))
        mlflow.log_metric("repair_f1_macro_best", float(best_row["f1 macro"]))
        mlflow.log_metric("repair_f1_micro_best", float(best_row["f1 micro"]))
        mlflow.log_metric("repair_f1_weighted_best", float(best_row["f1 weighted"]))
        mlflow.log_metric("repair_TV_distance_best", float(best_row["TV distance"]))
        mlflow.log_param("repair_best_method_string", str(best_row["method"]))

        print("\n🧩 REPAIR — summary (means by method):")
        print(agg)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

def _load_cfg() -> DictConfig:
    """
    Load Hydra config, trying current directory first, then "./humancompatible".

    Returns
    -------
    DictConfig
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
            "  preprocessing_name: Reweighing | _ (ignored for FACTS/REPAIR)\n"
            "  postprocessing_name: EqOddsPostprocessing | CalibratedEqOddsPostprocessing | RejectOptionClassification | FACTS | REPAIR\n"
            "  search_algo: tpe | rand   (default: tpe)  (only used in the regular path)\n\n"
            "FACTS:\n"
            "  python run_mlflow.py adult _ FACTS\n"
            "REPAIR (Adult-only):\n"
            "  python run_mlflow.py adult _ REPAIR <origin|barycentre|partial> [partial_theta]\n\n"
            "Examples:\n"
            "  python run_mlflow.py custom Reweighing EqOddsPostprocessing tpe\n"
            "  python run_mlflow.py adult _ FACTS\n"
            "  python run_mlflow.py compas _ FACTS\n"
            "  python run_mlflow.py adult _ REPAIR origin\n"
            "  python run_mlflow.py adult _ REPAIR partial 1e-3\n"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    preprocessing_name = sys.argv[2]
    postprocessing_name = sys.argv[3] if len(sys.argv) > 3 else "EqOddsPostprocessing"

    cfg: DictConfig = _load_cfg()

    # MLflow init from Hydra config
    init_mlflow_from_cfg(cfg)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # FACTS-only path (skip mitigation + FLAML)
    if postprocessing_name.strip().upper() == "FACTS":
        run_facts_experiment(dataset_name, cfg)
        sys.exit(0)

    # REPAIR path (Adult dataset only; sex protected attr)
    if postprocessing_name.strip().upper() == "REPAIR":
        if dataset_name.lower() != "adult":
            raise ValueError("The REPAIR path here is implemented for the Adult dataset only.")
        repair_method = sys.argv[4] if len(sys.argv) > 4 else "origin"
        partial_theta = float(sys.argv[5]) if (len(sys.argv) > 5 and repair_method == "partial") else None
        run_repair_experiment_adult(repair_method, cfg, partial_theta=partial_theta)
        sys.exit(0)

    # Regular path (mitigation + FLAML)
    aif_data, privileged_groups, unprivileged_groups, TARGET_LABEL = load_dataset_dispatch(dataset_name, cfg)
    print(f"\n🎯 Using target label: '{TARGET_LABEL}'")
    print(f"🛡️ Privileged groups: {privileged_groups}")
    print(f"🛡️ Unprivileged groups: {unprivileged_groups}")

    search_algo = sys.argv[4] if len(sys.argv) > 4 else "tpe"
    run_experiment(preprocessing_name, postprocessing_name, search_algo)
