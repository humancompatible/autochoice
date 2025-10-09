"""
data_helper
===========

Hydra-friendly helpers for:

- MLflow initialization from config
- Dataset loaders:
  * load_custom_dataset(...)  -> build multiclass label from automatch_score, add is_privileged
  * load_openml_adult(...)    -> OpenML Adult; add is_privileged from sex; build income_binary
  * load_compas_dataset(...)  -> COMPAS; add is_privileged from race ('Caucasian'); label two_year_recid
- Converting pandas DataFrames into AIF360 StandardDataset safely (no MultiIndex issues)

All functions include Sphinx-friendly docstrings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from aif360.datasets import CompasDataset, StandardDataset
from aif360.sklearn.datasets.openml_datasets import fetch_adult


# ---------------------------------------------------------------------------
# MLflow init
# ---------------------------------------------------------------------------

def init_mlflow_from_cfg(cfg: DictConfig) -> None:
    """
    Initialize MLflow using a Hydra config.

    Expected config structure
    -------------------------
    cfg.mlflow.tracking_uri : str
        MLflow tracking server, e.g. ``"http://192.168.1.151:5000"``.
    cfg.mlflow.registry_uri : Optional[str]
        Model registry URI (may be ``null``).
    cfg.mlflow.experiment_name : str
        Name of the experiment to use/create.
    cfg.mlflow.autolog : bool
        If true, enable MLflow autolog for sklearn.
    cfg.mlflow.flavor : str
        MLflow autolog flavor, e.g. ``"sklearn"``.
    cfg.mlflow.env : Dict[str, str]
        Extra environment variables to export (e.g., S3/TLS settings).

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    """
    mlflow_cfg = cfg.get("mlflow", {}) if hasattr(cfg, "get") else {}
    # export env first
    for k, v in (mlflow_cfg.get("env") or {}).items():
        if v is not None:
            os.environ[str(k)] = str(v)

    tracking_uri = mlflow_cfg.get("tracking_uri")
    registry_uri = mlflow_cfg.get("registry_uri")
    experiment_name = mlflow_cfg.get("experiment_name", "Default")
    autolog_enabled = bool(mlflow_cfg.get("autolog", True))
    flavor = mlflow_cfg.get("flavor", "sklearn")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    # ensure experiment exists
    mlflow.set_experiment(experiment_name)

    # autolog (sklearn flavor commonly used here)
    if autolog_enabled and flavor.lower() == "sklearn":
        import mlflow.sklearn as ml_sklearn

        ml_sklearn.autolog()


# ---------------------------------------------------------------------------
# AIF360 conversion utility (MultiIndex-safe)
# ---------------------------------------------------------------------------

def _convert_to_standard_dataset(
    df: pd.DataFrame,
    target_label_name: str,
    favorable_classes: List[Any],
    protected_is_privileged_col: str = "is_privileged",
    categorical_infer: bool = True,
    scores_name: str = "",
) -> StandardDataset:
    """
    Convert a pandas DataFrame into an AIF360 StandardDataset.

    This function ensures the index is flattened (no MultiIndex) to avoid
    pandas → AIF360 issues when converting index to string.

    Parameters
    ----------
    df : pd.DataFrame
        Input features + labels + protected indicator column.
    target_label_name : str
        Name of the label column in ``df``.
    favorable_classes : list
        Values of ``target_label_name`` considered favorable.
    protected_is_privileged_col : str, default="is_privileged"
        Name of the column that is 1 for privileged, 0 for unprivileged.
    categorical_infer : bool, default=True
        If True, any object/category columns are encoded with categorical codes.
    scores_name : str, default=""
        Optional scores column name (unused here).

    Returns
    -------
    StandardDataset
        AIF360 dataset with numeric features and labels.
    """
    if target_label_name not in df.columns:
        raise KeyError(f"Target column '{target_label_name}' not found.")
    if protected_is_privileged_col not in df.columns:
        raise KeyError(f"Protected indicator '{protected_is_privileged_col}' not found.")

    df_clean = df.dropna(subset=[target_label_name, protected_is_privileged_col]).copy()

    # 🔧 Flatten index to avoid MultiIndex -> astype(str) error in AIF360
    if isinstance(df_clean.index, pd.MultiIndex) or not isinstance(df_clean.index, pd.RangeIndex):
        df_clean.reset_index(drop=True, inplace=True)

    # Encode categoricals to integer codes
    categorical_features: List[str] = []
    if categorical_infer:
        categorical_features = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_features:
            df_clean[col] = df_clean[col].astype("category").cat.codes

    # Labels must be integers
    if df_clean[target_label_name].dtype.name in ("object", "category", "bool"):
        # If label is boolean or category, map to ints
        if df_clean[target_label_name].dtype.name == "bool":
            df_clean[target_label_name] = df_clean[target_label_name].astype(int)
        else:
            df_clean[target_label_name] = df_clean[target_label_name].astype("category").cat.codes
    df_clean[target_label_name] = df_clean[target_label_name].astype(int)

    selected_features = df_clean.columns.tolist()

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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_custom_dataset(
    dataset_path: str,
    *,
    function_filter_value: str = "Legal",
    target_from: str = "automatch_score",
    target_bins: Tuple[float, float, float] = (0.6, 0.85, float("inf")),
    target_label_name: str = "target_class",
    protected_attribute: str = "experiences_no",
    protected_threshold: int = 2,
) -> Tuple[pd.DataFrame, StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """
    Load and prepare your custom parquet dataset for AIF360.

    Steps
    -----
    - Select columns (exclude heavy/text), keep ``job.function`` for filtering
    - Filter to ``job.function == function_filter_value`` and drop that column
    - Build multiclass label from ``target_from`` using ``target_bins``
    - Add ``is_privileged`` = 1 if ``protected_attribute > protected_threshold`` else 0
    - Convert to AIF360 StandardDataset (favorable class is the top bin)

    Parameters
    ----------
    dataset_path : str
        Absolute path to the parquet file (mounted in container at ``/data``).
    function_filter_value : str, default="Legal"
    target_from : str, default="automatch_score"
    target_bins : tuple, default=(0.6, 0.85, inf)
        Right-open cuts for three classes: ``(-inf, b0]``, ``(b0, b1]``, ``(b1, inf)``.
    target_label_name : str, default="target_class"
    protected_attribute : str, default="experiences_no"
    protected_threshold : int, default=2

    Returns
    -------
    data : pd.DataFrame
        Cleaned dataframe with label and ``is_privileged``.
    aif_data : StandardDataset
        AIF360 dataset.
    privileged_groups : list[dict]
        ``[{"is_privileged": 1}]``
    unprivileged_groups : list[dict]
        ``[{"is_privileged": 0}]``
    target_label_name : str
        Name of the label column.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

    # Determine included columns
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

    # Filter to a specific function and drop the column
    if "job.function" in data.columns:
        data = data[data["job.function"] == function_filter_value].copy()
        data.drop(columns=["job.function"], inplace=True)
    else:
        raise KeyError("Column 'job.function' not found in dataset.")

    # Build multiclass label from target_from
    if target_from not in data.columns:
        raise KeyError(f"Column '{target_from}' not found in dataset.")
    b0, b1, b2 = target_bins
    data[target_label_name] = pd.cut(
        data[target_from],
        bins=[-np.inf, b0, b1, b2],
        labels=[0, 1, 2],
    ).astype(int)

    print("\n🎯 Using multiclass target from 'automatch_score':")
    print("  ➤ Class 0: Low match (< 0.6)")
    print("  ➤ Class 1: Medium match (0.6 - 0.85)")
    print("  ➤ Class 2: High match (> 0.85)")

    # Protected indicator
    if protected_attribute not in data.columns:
        raise KeyError(f"Protected attribute '{protected_attribute}' not found.")
    data["is_privileged"] = (data[protected_attribute] > protected_threshold).astype(int)

    print(f"\n🛡️ Using '{protected_attribute}' as protected attribute.")
    print(f"   ➔ Privileged group: {protected_attribute} > {protected_threshold}")
    print(f"   ➔ Unprivileged group: {protected_attribute} <= {protected_threshold}")

    # Convert to AIF360
    aif_data = _convert_to_standard_dataset(
        df=data,
        target_label_name=target_label_name,
        favorable_classes=[2],  # top bin is favorable
        protected_is_privileged_col="is_privileged",
        categorical_infer=True,
    )
    privileged_groups = [{"is_privileged": 1}]
    unprivileged_groups = [{"is_privileged": 0}]
    return data, aif_data, privileged_groups, unprivileged_groups, target_label_name


def load_openml_adult(
    *,
    random_seed: int = 42,
    train_size: float = 0.7,
    build_pipeline: bool = False,
    return_aif360: bool = True,
    protected_for_aif: str = "sex",
    favorable_label_for_aif: str = ">50K",
) -> Dict[str, Any]:
    """
    Load the OpenML *Adult* dataset using AIF360's helper, and optionally build AIF360 dataset.

    Parameters
    ----------
    random_seed : int, default=42
    train_size : float, default=0.7
        Not used here directly but kept for parity with examples.
    build_pipeline : bool, default=False
        Reserved for user code that wants a sklearn Pipeline.
    return_aif360 : bool, default=True
        If True, also return an AIF360 StandardDataset.
    protected_for_aif : str, default="sex"
        Protected attribute to derive ``is_privileged`` from (``Male`` considered privileged).
    favorable_label_for_aif : str, default=">50K"
        Favorable income group.

    Returns
    -------
    info : dict
        Keys:
          - "X" : pd.DataFrame
          - "y" : pd.Series
          - "aif_data" : StandardDataset (if return_aif360)
          - "privileged_groups" / "unprivileged_groups"
    """
    X, y, sample_weight = fetch_adult()
    df = X.copy()
    df["income"] = y

    # A simple "cleaning" step: ensure categories stay as strings (no encoding yet)
    # Add protected indicator
    if protected_for_aif not in df.columns:
        raise KeyError(f"Protected attribute '{protected_for_aif}' not found in Adult dataset.")
    df["is_privileged"] = (df[protected_for_aif].astype(str).str.lower() == "male").astype(int)

    # Binary target: 1 if favorable label, else 0
    df["income_binary"] = (df["income"] == favorable_label_for_aif).astype(int)

    info: Dict[str, Any] = {"X": df.drop(columns=["income_binary"]), "y": df["income_binary"]}

    if return_aif360:
        aif_df = df.copy()
        aif_data = _convert_to_standard_dataset(
            df=aif_df,
            target_label_name="income_binary",
            favorable_classes=[1],
            protected_is_privileged_col="is_privileged",
            categorical_infer=True,
        )
        info["aif_data"] = aif_data
        info["privileged_groups"] = [{"is_privileged": 1}]
        info["unprivileged_groups"] = [{"is_privileged": 0}]

    return info


def load_compas_dataset() -> Tuple[StandardDataset, List[Dict[str, int]], List[Dict[str, int]], str]:
    """
    Load COMPAS via AIF360 and convert to a StandardDataset with a unified ``is_privileged`` flag.

    Notes
    -----
    - Protected group is derived from ``race == 'Caucasian'`` (privileged).
    - Label ``two_year_recid`` is favorable when 0 (did **not** recidivate).

    Returns
    -------
    aif_data : StandardDataset
    privileged_groups : list[dict]
    unprivileged_groups : list[dict]
    target_label_name : str
    """
    compas = CompasDataset()
    df, attrs = compas.convert_to_dataframe(de_dummy_code=True)
    label_name = attrs["label_names"][0]  # usually 'two_year_recid'

    # Define is_privileged from race
    if "race" not in df.columns:
        raise KeyError("Expected 'race' in COMPAS dataset.")
    df["is_privileged"] = (df["race"].astype(str) == "Caucasian").astype(int)

    # Ensure label is int; favorable = 0 (no recidivism)
    df[label_name] = pd.to_numeric(df[label_name], errors="coerce").fillna(1).astype(int)

    aif_data = _convert_to_standard_dataset(
        df=df,
        target_label_name=label_name,
        favorable_classes=[0],
        protected_is_privileged_col="is_privileged",
        categorical_infer=True,
    )
    privileged_groups = [{"is_privileged": 1}]
    unprivileged_groups = [{"is_privileged": 0}]
    return aif_data, privileged_groups, unprivileged_groups, label_name


# ---------------------------------------------------------------------------
# Convenience for "repair" experiments
# ---------------------------------------------------------------------------

def get_repair_ready_dataframe(
    dataset_name: str = "adult",
    *,
    protected_attr: Optional[str] = None,
    favorable_label_for_aif: str = ">50K",
) -> Tuple[pd.DataFrame, str, str]:
    """
    Return a pandas DataFrame + names useful for the 'repair' toolkit experiments.

    By default this returns the OpenML Adult dataset with:
      - label column: 'income_binary' (1 if income == '>50K', else 0)
      - protected attribute column: 'sex' (string values kept as-is)
      - also includes a convenience boolean/indicator column 'is_privileged' (Male=1)

    Parameters
    ----------
    dataset_name : {'adult', 'compas', 'custom'}, default='adult'
        Currently optimized for 'adult'. Other datasets will be returned in
        their AIF360-to-DataFrame form with the label column preserved.
    protected_attr : str, optional
        Name of the protected attribute column to expose in the returned DataFrame.
        If None, defaults to:
           - 'sex' for 'adult'
           - 'race' for 'compas'
           - 'is_privileged' for 'custom' (since the raw protected attribute is domain-specific)
    favorable_label_for_aif : str, default='>50K'
        Only used for 'adult' to define 'income_binary'.

    Returns
    -------
    df : pd.DataFrame
        DataFrame including the label column and a human-readable protected attribute column.
    label_col : str
        Name of the label column in `df`.
    protected_col : str
        Name of the protected attribute column in `df`.
    """
    name = (dataset_name or "adult").lower()

    if name == "adult":
        X, y, _ = fetch_adult()
        df = X.copy()
        df["income"] = y
        prot_col = protected_attr or "sex"
        if prot_col not in df.columns:
            raise KeyError(f"Protected attribute '{prot_col}' not found in Adult dataset.")
        df["is_privileged"] = (df[prot_col].astype(str).str.lower() == "male").astype(int)
        df["income_binary"] = (df["income"] == favorable_label_for_aif).astype(int)
        return df, "income_binary", prot_col

    if name == "compas":
        compas = CompasDataset()
        df, attrs = compas.convert_to_dataframe(de_dummy_code=True)
        label_name = attrs["label_names"][0]
        prot_col = protected_attr or "race"
        if prot_col not in df.columns:
            raise KeyError(f"Protected attribute '{prot_col}' not found in COMPAS dataset.")
        # unify helper column
        df["is_privileged"] = (df["race"].astype(str) == "Caucasian").astype(int)
        # ensure int label
        df[label_name] = pd.to_numeric(df[label_name], errors="coerce").fillna(1).astype(int)
        return df, label_name, prot_col

    if name == "custom":
        # For custom data, we cannot reload parquet here generically;
        # downstream code should pass through load_custom_dataset and operate on that df.
        raise NotImplementedError(
            "get_repair_ready_dataframe('custom'): please load your DataFrame first with "
            "load_custom_dataset(...) and use the returned DataFrame directly."
        )

    raise ValueError("dataset_name must be one of: 'adult', 'compas', 'custom'.")
