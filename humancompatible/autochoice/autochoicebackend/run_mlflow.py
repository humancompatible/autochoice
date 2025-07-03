import os
import sys
import mlflow
import numpy as np
import pandas as pd
import torch
from flaml import AutoML
from hyperopt import fmin, tpe, rand, hp, Trials
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing, CalibratedEqOddsPostprocessing, RejectOptionClassification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier
from sklearn.ensemble import RandomForestClassifier
import time

#  GPU Check
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("No GPU detected, using CPU.")


# IP and PORT of MLFlow Tracking Server
IP = ""
PORT = "5000"

# MLflow Setup
MLFLOW_TRACKING_URI = f"http://{IP}:{PORT}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.sklearn.autolog()

# Load Dataset
DATASET_PATH = "/data/datasetfile.csv"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")
data = pd.read_csv(DATASET_PATH)

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Show column types and unique values
print("\n Data Types and Categories:")
for col in data.columns:
    dtype = data[col].dtype
    col_type = "categorical" if dtype == object else "numerical"
    print(f"{col}: {dtype} ({col_type})")

#  Define target label and protected attribute
TARGET_LABEL = "job.remote"
PROTECTED_ATTRIBUTE = "experiences_no"
PROTECTED_THRESHOLD = data[PROTECTED_ATTRIBUTE].median()

print(f"\n Using '{PROTECTED_ATTRIBUTE}' as protected attribute (numerical).")
print(f"   ➤ Privileged group: {PROTECTED_ATTRIBUTE} <= {PROTECTED_THRESHOLD}")
print(f"   ➤ Unprivileged group: {PROTECTED_ATTRIBUTE} > {PROTECTED_THRESHOLD}")

def convert_to_standard_dataset(df, target_label_name, scores_name=""):
    protected_attributes = [PROTECTED_ATTRIBUTE]
    selected_features = df.columns.tolist()

    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_features:
        df[col] = df[col].astype("category").cat.codes

    if df[target_label_name].dtype == bool:
        df[target_label_name] = df[target_label_name].astype(int)

    df["is_privileged"] = (df[PROTECTED_ATTRIBUTE] <= PROTECTED_THRESHOLD).astype(int)
    protected_attributes = ["is_privileged"]

    dataset = StandardDataset(
        df=df,
        label_name=target_label_name,
        favorable_classes=[1],
        scores_name=scores_name,
        protected_attribute_names=protected_attributes,
        privileged_classes=[[1]],
        categorical_features=categorical_features,
        features_to_keep=selected_features + ["is_privileged"]
    )

    if scores_name == "":
        dataset.scores = dataset.labels.copy()
    return dataset

#  Convert
aif_data = convert_to_standard_dataset(data, target_label_name=TARGET_LABEL)

#  Redefine groups
privileged_groups = [{"is_privileged": 1}]
unprivileged_groups = [{"is_privileged": 0}]

def apply_preprocessing(preprocessing_name, dataset):
    if preprocessing_name == "Reweighing":
        return Reweighing(privileged_groups, unprivileged_groups).fit_transform(dataset)
    raise ValueError(f"Unsupported preprocessing method: {preprocessing_name}")

def apply_postprocessing(postprocessing_name, dataset_true, dataset_pred):
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

def log_fairness_metrics(dataset_before, dataset_after, prefix=""):
    from aif360.metrics import ClassificationMetric
    metric = ClassificationMetric(dataset_before, dataset_after, unprivileged_groups, privileged_groups)
    metrics = {
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "disparate_impact": metric.disparate_impact(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "average_odds_difference": metric.average_odds_difference(),
        "theil_index": metric.theil_index(),
    }
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}", v)
        print(f"{prefix}{k}: {v:.4f}")

search_space = {
    'time_budget': hp.uniform('time_budget', 1, 3),
    'metric': hp.choice('metric', ['accuracy']),
    'estimator_list': hp.choice('estimator_list', [['lgbm', 'xgboost']])
}

def objective(params):
    automl_settings = {
        "time_budget": int(params['time_budget']),
        "task": "classification",
        "estimator_list": params['estimator_list'],
        "metric": params['metric'],
        "verbose": 0
    }
    if USE_GPU:
        automl_settings["use_gpu"] = True

    automl = AutoML()
    automl.fit(X_train, y_train, **automl_settings)
    best_model = automl.model

    #  Inference time measurement
    start = time.time()
    _ = best_model.predict(X_test)
    automl_inference_time = time.time() - start

    acc = accuracy_score(y_test, best_model.predict(X_test))

    #  Log and compare
    result_df = pd.DataFrame([
        ["Baseline RandomForest", baseline_acc, baseline_inference_time],
        ["AutoML Optimized", acc, automl_inference_time]
    ], columns=["Model", "Accuracy", "Inference Time"])

    print("\n Model Comparison:")
    print(result_df)
    mlflow.log_metric("automl_inference_time", automl_inference_time)
    mlflow.log_metric("automl_accuracy", acc)
    mlflow.log_artifact(result_df.to_csv("model_comparison.csv", index=False))

    mapie = MapieClassifier(estimator=best_model, method="score")
    mapie.fit(X_train, y_train)
    y_pred_mapie, _ = mapie.predict(X_test, alpha=0.1)
    conformal_coverage = np.mean(y_pred_mapie != -1)
    mlflow.log_metric("conformal_coverage", conformal_coverage)

    pred_dataset = preprocessed_data.copy()
    pred_dataset.labels = best_model.predict(preprocessed_data.features).reshape(-1, 1)
    postprocessed_data = apply_postprocessing(postprocessing_name, preprocessed_data, pred_dataset)
    log_fairness_metrics(preprocessed_data, postprocessed_data, prefix="postprocessed_")

    return -acc

def run_experiment(preprocessing_name, postprocessing_name, search_algo):
    with mlflow.start_run():
        mlflow.log_param("preprocessing_algorithm", preprocessing_name)
        mlflow.log_param("postprocessing_algorithm", postprocessing_name)

        global preprocessed_data, X_train, X_test, y_train, y_test, baseline_acc, baseline_inference_time
        preprocessed_data = apply_preprocessing(preprocessing_name, aif_data)

        X_train, X_test, y_train, y_test = train_test_split(
            preprocessed_data.features, preprocessed_data.labels.ravel(), test_size=0.2, random_state=42
        )

        #  Baseline model
        baseline_rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
        baseline_rf.fit(X_train, y_train)
        start = time.time()
        _ = baseline_rf.predict(X_test)
        baseline_inference_time = time.time() - start
        baseline_acc = accuracy_score(y_test, baseline_rf.predict(X_test))
        mlflow.log_metric("baseline_inference_time", baseline_inference_time)
        mlflow.log_metric("baseline_accuracy", baseline_acc)

        log_fairness_metrics(aif_data, aif_data, prefix="raw_")
        log_fairness_metrics(preprocessed_data, preprocessed_data, prefix="preprocessed_")

        trials = Trials()
        algo = tpe.suggest if search_algo == "tpe" else rand.suggest
        best_params = fmin(fn=objective, space=search_space, algo=algo, max_evals=1, trials=trials)

        for key, value in best_params.items():
            mlflow.log_param(key, str(value))

if __name__ == "__main__":
    preprocessing_name = sys.argv[1]
    postprocessing_name = sys.argv[2]
    search_algo = sys.argv[3] if len(sys.argv) > 3 else "tpe"
    run_experiment(preprocessing_name, postprocessing_name, search_algo)
