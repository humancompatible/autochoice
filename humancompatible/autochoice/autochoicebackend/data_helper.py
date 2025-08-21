"""
data_loading
============

Hydra-driven MLflow initialization and dataset loaders used by both the
training pipeline and the Voilá frontend.

Public API:
- init_mlflow_from_cfg
- mlflow_client_from_cfg
- load_custom_dataset
- load_openml_adult
- load_compas_dataset
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from aif360.datasets import StandardDataset
from omegaconf import DictConfig

try:
    # Optional convenience loader for Adult
    from aif360.sklearn.datasets.openml_datasets import fetch_adult
except Exception:  # pragma: no cover
    fetch_adult = None  # type: ignore

__all__ = [
    "init_mlflow_from_cfg",
    "mlflow_client_from_cfg",
    "load_custom_dataset",
    "load_openml_adult",
    "load_compas_dataset",
]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _convert_to_standard_dataset(
    df: pd.DataFrame,
    target_label_name: str,
    favorable_classes: List[Any],
    protected_is_privileged_col: str = "is_privileged",
    categorical_infer: bool = True,
    scores_name: str = "",
) -> StandardDataset:
    """Convert a pandas DataFrame into an AIF360 StandardDataset.

    Assumes a binary protected indicator column exists (default: ``is_privileged`` 0/1).

    Args:
        df: Input DataFrame including the binary protected indicator column.
        target_label_name: Name of the target/label column.
        favorable_classes: Values considered favorable for the target.
        protected_is_privileged_col: Column name of 0/1 protected indicator.
        categorical_infer: If True, encode object/category columns to integer codes.
        scores_name: Optional scores column name.

    Returns:
        StandardDataset: AIF360 dataset ready for fairness analysis.
    """
    if target_label_name not in df.columns:
        raise KeyError(f"Target column '{target_label_name}' not found.")
    if protected_is_privileged_col not in df.columns:
        raise KeyError(f"Protected indicator '{protected_is_privileged_col}' not found.")

    selected_features = df.columns.tolist()
    df_clean = df.dropna(subset=[target_label_name, protected_is_privileged_col])

    categorical_features: List[str] = []
    if categorical_infer:
        categorical_features = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_features:
            df_clean[col] = df_clean[col].astype("category").cat.codes

    if df_clean[target_label_name].dtype.name in ("object", "category"):
        df_clean[target_label_name] = df_clean[target_label_name].astype("category").cat.codes
    df_clean[target_label_name] = df_clean[target_label_name].astype(int)

    return StandardDataset(
        df=df_clean,
        label_name=target_label_name,
        favorable_classes=favorable_classes,
        scores_name=scores_name,
        protected_attribute_names=[protected_is_privileged_col],
        privileged_classes=[[1]],
        categorical_features=categorical_features,
        features_to_keep=selected_features,
    )


def load_custom_dataset(
    dataset_path: str = "/data/dataset1M.parquet",
    function_filter_value: str = "Legal",
    target_from: str = "automatch_score",
    target_bins: Tuple[float, float, float] = (0.6, 0.85, np.inf),
    target_label_name: str = "target_class",
    protected_attribute: str = "experiences_no",
    protected_threshold: int = 2,
) -> Tuple[pd.DataFrame, StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """Load and prepare the custom parquet dataset.

    Steps:
      1) Read schema, exclude heavy/unused columns, ensure ``job.function`` present.
      2) Filter rows where ``job.function == function_filter_value``.
      3) Create multiclass target from ``automatch_score`` via ``target_bins``.
      4) Create ``is_privileged`` from ``protected_attribute`` threshold.
      5) Convert to AIF360 StandardDataset (favorable class = 2).

    Returns:
        (processed_df, aif_data, privileged_groups, unprivileged_groups, target_label_name)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

    df_schema = pd.read_parquet(dataset_path, engine="pyarrow", columns=None)
    all_columns = df_schema.columns.tolist()
    exclude = [
        "skills",
        "job.remote",
        "job.id",
        "job.account_id",
        "applicant_id",
        "job.education",
        "job.account_name",
    ]
    include_columns = [c for c in all_columns if c not in exclude]
    if "job.function" not in include_columns:
        include_columns.append("job.function")

    data = pd.read_parquet(dataset_path, columns=include_columns)
    data.columns = data.columns.str.strip().str.lower()

    if "job.function" not in data.columns:
        raise KeyError("Column 'job.function' not found in dataset.")
    data = data[data["job.function"] == function_filter_value].drop(columns=["job.function"])

    if target_from not in data.columns:
        raise KeyError(f"Column '{target_from}' not found in dataset.")
    bins = [-np.inf, target_bins[0], target_bins[1], target_bins[2]]
    data[target_label_name] = pd.cut(data[target_from], bins=bins, labels=[0, 1, 2]).astype(int)

    if protected_attribute not in data.columns:
        raise KeyError(f"Protected attribute '{protected_attribute}' not found in dataset.")
    data["is_privileged"] = (data[protected_attribute] > protected_threshold).astype(int)

    aif_data = _convert_to_standard_dataset(
        df=data,
        target_label_name=target_label_name,
        favorable_classes=[2],
        protected_is_privileged_col="is_privileged",
        categorical_infer=True,
        scores_name="",
    )

    privileged_groups = [{"is_privileged": 1}]
    unprivileged_groups = [{"is_privileged": 0}]
    return data, aif_data, privileged_groups, unprivileged_groups, target_label_name

def load_openml_adult(
    random_seed: int = 42,
    train_size: float = 0.7,
    build_pipeline: bool = True,
    return_aif360: bool = True,
    protected_for_aif: str = "sex",
    favorable_label_for_aif: str = ">50K",
) -> Dict[str, Any]:
    """Load the OpenML Adult dataset with optional AIF360 dataset and sklearn pipeline."""
    if fetch_adult is None:
        raise RuntimeError("AIF360 sklearn OpenML loaders are unavailable. Install the appropriate extras.")

    X, y, sample_weight = fetch_adult()
    df = X.copy()
    df["income"] = y

    drop_cols = ["income"]
    if return_aif360:
        drop_cols.append(protected_for_aif)
    df = df.dropna(subset=drop_cols)

    y_series = df["income"]
    X_df = df.drop(columns=["income"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, train_size=train_size, random_state=random_seed, stratify=y_series
    )

    categorical_features = X_df.select_dtypes(include=["object", "category"]).columns.to_list()
    onehot = ColumnTransformer(
        transformers=[("one-hot-encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough",
    )

    result: Dict[str, Any] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "categorical_features": categorical_features,
    }

    if build_pipeline:
        model = Pipeline(
            steps=[
                ("one-hot-encoder", onehot),
                ("clf", LogisticRegression(max_iter=1500)),
            ]
        )
        result["pipeline"] = model

    if return_aif360:
        if protected_for_aif not in df.columns:
            raise KeyError(f"Protected attribute '{protected_for_aif}' not in Adult dataset.")

        if protected_for_aif == "sex":
            protected_indicator = (df["sex"].astype(str).str.lower() == "male").astype(int)
        else:
            top_cat = df[protected_for_aif].astype(str).mode().iloc[0]
            protected_indicator = (df[protected_for_aif].astype(str) == top_cat).astype(int)

        df_aif = df.copy()
        df_aif["is_privileged"] = protected_indicator
        df_aif["income_binary"] = (df_aif["income"] == favorable_label_for_aif).astype(int)

        aif_data = _convert_to_standard_dataset(
            df=df_aif.drop(columns=["income"]),
            target_label_name="income_binary",
            favorable_classes=[1],
            protected_is_privileged_col="is_privileged",
            categorical_infer=True,
            scores_name="",
        )
        privileged_groups = [{"is_privileged": 1}]
        unprivileged_groups = [{"is_privileged": 0}]

        result.update(
            {
                "aif_data": aif_data,
                "privileged_groups": privileged_groups,
                "unprivileged_groups": unprivileged_groups,
            }
        )

    return result

def load_compas_dataset() -> Tuple[StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """Load the COMPAS dataset via AIF360."""
    try:
        from aif360.datasets import CompasDataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("AIF360 CompasDataset unavailable. Install AIF360 with datasets extras.") from e

    compas = CompasDataset()
    if "sex" in compas.protected_attribute_names:
        privileged_groups = [{"sex": 1}]
        unprivileged_groups = [{"sex": 0}]
    else:
        privileged_groups = [{"race": 1}]
        unprivileged_groups = [{"race": 0}]

    target_label = compas.label_names[0] if compas.label_names else "two_year_recid"
    return compas, privileged_groups, unprivileged_groups, target_label



def init_mlflow_from_cfg(cfg: DictConfig) -> None:
    """Initialize MLflow strictly from a Hydra/OmegaConf config.

    Expected YAML structure under ``mlflow``:
      tracking_uri: "http://mlflow:5000"
      registry_uri: null
      experiment_name: "legal-matching"
      autolog: true
      flavor: "sklearn"   # or "all", "pytorch", "xgboost", ...
      env: { ... }        # optional env vars
    """
    if not hasattr(cfg, "mlflow"):
        raise RuntimeError("Config missing 'mlflow' section.")

    mlcfg = cfg.mlflow

    env: Optional[Mapping[str, Any]] = mlcfg.get("env")
    if env:
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    tracking_uri = mlcfg.get("tracking_uri")
    if not tracking_uri:
        raise RuntimeError("cfg.mlflow.tracking_uri is required.")
    mlflow.set_tracking_uri(tracking_uri)

    registry_uri = mlcfg.get("registry_uri")
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    experiment_name = mlcfg.get("experiment_name")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    if mlcfg.get("autolog", True):
        flavor = mlcfg.get("flavor", "sklearn")
        try:
            if flavor == "sklearn":
                import mlflow.sklearn as mls; mls.autolog()
            elif flavor == "pytorch":
                import mlflow.pytorch as mlpt; mlpt.autolog()
            elif flavor == "xgboost":
                import mlflow.xgboost as mlx; mlx.autolog()
            elif flavor == "lightgbm":
                import mlflow.lightgbm as mll; mll.autolog()
            elif flavor == "catboost":
                import mlflow.catboost as mlc; mlc.autolog()
            elif flavor == "fastai":
                import mlflow.fastai as mlf; mlf.autolog()
            elif flavor == "transformers":
                import mlflow.transformers as mlt; mlt.autolog()
            elif flavor == "all":
                mlflow.autolog()
            else:
                mlflow.autolog()
        except Exception:
            mlflow.autolog()


def mlflow_client_from_cfg(cfg: DictConfig) -> MlflowClient:
    """Create an MlflowClient that matches Hydra config."""
    init_mlflow_from_cfg(cfg)
    return MlflowClient(
        tracking_uri=cfg.mlflow.tracking_uri,
        registry_uri=cfg.mlflow.get("registry_uri"),
    )
