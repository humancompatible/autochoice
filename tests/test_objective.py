# test_objective.py
import numpy as np
import pytest
from conftest import FakeStdDataset

def test_objective_happy_path(monkeypatch, mlflow_spy, import_module):
    mod = import_module

    # Globals required by objective()
    mod.postprocessing_name = "EqOddsPostprocessing"
    mod.preprocessed_data = FakeStdDataset(np.array([[0], [1], [0], [1]]), np.array([0, 1, 0, 1]))

    mod.X_train = np.array([[0], [1]])
    mod.y_train = np.array([0, 1])
    mod.X_test = np.array([[0], [1]])
    mod.y_test = np.array([0, 1])

    class PerfectModel:
        def predict(self, X):  # perfect predictions
            return X.ravel().astype(int)

    class DummyAutoML:
        def __init__(self): self.model = None
        def fit(self, X, y, **settings): self.model = PerfectModel()

    monkeypatch.setattr(mod, "AutoML", DummyAutoML)

    class DummyMapie:
        def __init__(self, estimator, method="score"): pass
        def fit(self, X, y): pass
        def predict(self, X, alpha=0.1):
            return np.zeros((len(X),)), None
    monkeypatch.setattr(mod, "MapieClassifier", DummyMapie)

    monkeypatch.setattr(mod, "apply_postprocessing", lambda *a, **k: "POST_OK")

    called = {"n": 0}
    def fake_log_fairness(*a, **k): called["n"] += 1
    monkeypatch.setattr(mod, "log_fairness_metrics", fake_log_fairness)

    val = mod.objective({"time_budget": 10, "estimator_list": ["lgbm"], "metric": "accuracy"})
    assert pytest.approx(val, rel=1e-9) == -1.0
    assert "automl_accuracy" in mlflow_spy.metrics
    assert called["n"] == 1
