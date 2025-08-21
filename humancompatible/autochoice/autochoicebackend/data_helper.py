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
