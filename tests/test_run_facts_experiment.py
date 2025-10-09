# test_run_facts_experiment.py
import json
import os
import numpy as np
import pandas as pd
from conftest import MLflowSpy  # only for typing/awareness (fixture provided elsewhere)

class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)

def test_run_facts_experiment_happy(monkeypatch, tmp_path, mlflow_spy, import_module):
    mod = import_module

    # Dispatcher returns AIF dataset with convert_to_dataframe()
    def _convert_to_dataframe(de_dummy_code=True):
        df = pd.DataFrame({
            "age": [21, 35, 52, 28, 44, 60],
            "sex": ["M", "F", "F", "M", "F", "M"],
            "label": [0, 1, 1, 0, 1, 0],
        })
        attrs = {"label_names": ["label"], "protected_attribute_names": ["sex"]}
        return df, attrs

    class _AIF:
        def convert_to_dataframe(self, **k): return _convert_to_dataframe()

    monkeypatch.setattr(mod, "load_dataset_dispatch",
                        lambda name, cfg: (_AIF(), [{"sex": 1}], [{"sex": 0}], "label"))

    # Replace sklearn components with trivial versions
    class TrivialClf:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)

    class FakePipeline:
        def __init__(self, steps): self._clf = TrivialClf()
        def fit(self, X, y): return self._clf.fit(X, y)
        def predict(self, X): return self._clf.predict(X)

    class FakeCT:
        def __init__(self, *a, **k): pass
    class FakeOHE:
        def __init__(self, *a, **k): pass
    class FakeLR:
        def __init__(self, *a, **k): pass

    monkeypatch.setattr(mod, "Pipeline", lambda steps: FakePipeline(steps))
    monkeypatch.setattr(mod, "ColumnTransformer", FakeCT)
    monkeypatch.setattr(mod, "OneHotEncoder", FakeOHE)
    monkeypatch.setattr(mod, "LogisticRegression", FakeLR)

    # FACTS returns two biased groups
    monkeypatch.setattr(
        mod,
        "FACTS_bias_scan",
        lambda **k: [({"sex=F": True, "age<=30": True}, 0.42),
                     ({"sex=M": True, "age>50": True}, 0.33)]
    )

    cfg = _Cfg({"facts": {"metric": "equal-effectiveness",
                          "top_count": 2,
                          "freq_itemset_min_supp": 0.1}})

    mod.run_facts_experiment("adult", cfg)

    # Assertions
    assert mlflow_spy.params["facts_metric"] == "equal-effectiveness"
    assert "facts_unfairness_rank_1" in mlflow_spy.metrics
    assert any(p.endswith("facts_top_groups.json") for p in mlflow_spy.artifacts)

    assert os.path.exists("facts_top_groups.json")
    with open("facts_top_groups.json") as f:
        data = json.load(f)
        assert data["dataset"] == "adult"
        assert len(data["groups"]) == 2
