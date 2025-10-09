# conftest.py
import importlib
import numpy as np
import pandas as pd
import pytest

MODULE_UNDER_TEST = "run_mlflow"  # change if your file is named differently

# ---------- Minimal stand-ins ----------

class DummyMLflowRun:
    def __enter__(self, *a, **k): return self
    def __exit__(self, *a, **k): return False

class MLflowSpy:
    """Capture mlflow params/metrics/artifacts without a live server."""
    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.tracking_uri = "file:///tmp/mlruns"
        self._autolog_disabled = False

    # mlflow-like API
    def start_run(self, *a, **k): return DummyMLflowRun()
    def log_param(self, k, v): self.params[k] = v
    def log_metric(self, k, v): self.metrics[k] = v
    def log_artifact(self, path): self.artifacts.append(path)
    def get_tracking_uri(self): return self.tracking_uri

class FakeStdDataset:
    """Minimal AIF360-like container used across tests."""
    def __init__(self, X, y):
        self.features = np.asarray(X)
        self.labels = np.asarray(y).reshape(-1, 1)

    def copy(self):
        return FakeStdDataset(self.features.copy(), self.labels.copy())

# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    # Avoid GPU branches
    monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)

@pytest.fixture
def mlflow_spy(monkeypatch):
    """Patch the module's mlflow + msk.autolog to a spy object."""
    mod = importlib.import_module(MODULE_UNDER_TEST)
    spy = MLflowSpy()
    monkeypatch.setattr(mod, "mlflow", spy, raising=True)

    class _MSK:
        def autolog(self, disable=False):
            spy._autolog_disabled = disable
    monkeypatch.setattr(mod, "msk", _MSK(), raising=True)
    return spy

@pytest.fixture
def import_module():
    """Reload the module to re-evaluate globals cleanly per test file when needed."""
    import importlib
    mod = importlib.import_module(MODULE_UNDER_TEST)
    importlib.reload(mod)
    return mod
