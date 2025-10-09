# test_fairness_logging.py
from conftest import FakeStdDataset

def test_log_fairness_metrics_logs(monkeypatch, mlflow_spy, import_module):
    mod = import_module

    class DummyCM:
        def __init__(self, *a, **k): pass
        def statistical_parity_difference(self): return 0.1
        def disparate_impact(self): return 0.9
        def equal_opportunity_difference(self): return -0.2
        def average_odds_difference(self): return 0.05
        def theil_index(self): return 0.01

    monkeypatch.setattr(mod, "ClassificationMetric", DummyCM)
    ds = FakeStdDataset([[0], [1]], [0, 1])

    mod.log_fairness_metrics(ds, ds, prefix="pre_")
    assert mlflow_spy.metrics["pre_statistical_parity_difference"] == 0.1
    assert mlflow_spy.metrics["pre_disparate_impact"] == 0.9
    assert mlflow_spy.metrics["pre_equal_opportunity_difference"] == -0.2
    assert mlflow_spy.metrics["pre_average_odds_difference"] == 0.05
    assert mlflow_spy.metrics["pre_theil_index"] == 0.01
