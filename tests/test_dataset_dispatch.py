# test_dataset_dispatch.py
from conftest import FakeStdDataset

class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)

def test_load_dataset_dispatch_all_paths(monkeypatch, import_module):
    mod = import_module

    def fake_load_custom_dataset(**k):
        aif = FakeStdDataset([[0]], [0])
        pg, ug = [{"p": 1}], [{"p": 0}]
        return None, aif, pg, ug, "target_class"
    monkeypatch.setattr(mod, "load_custom_dataset", fake_load_custom_dataset)

    def fake_load_openml_adult(**k):
        return {
            "aif_data": FakeStdDataset([[0]], [0]),
            "privileged_groups": [{"sex": 1}],
            "unprivileged_groups": [{"sex": 0}],
        }
    monkeypatch.setattr(mod, "load_openml_adult", fake_load_openml_adult)

    def fake_load_compas_dataset():
        return FakeStdDataset([[0]], [0]), [{"r": 1}], [{"r": 0}], "two_year_recid"
    monkeypatch.setattr(mod, "load_compas_dataset", fake_load_compas_dataset)

    cfg = _Cfg({"data": {
        "path": "/tmp/x.parquet",
        "function_filter_value": "Legal",
        "protected_attribute": "experiences_no",
        "protected_threshold": 2
    }})

    aif, pg, ug, lbl = mod.load_dataset_dispatch("custom", cfg)
    assert lbl == "target_class"

    aif, pg, ug, lbl = mod.load_dataset_dispatch("adult", cfg)
    assert lbl == "income_binary"

    aif, pg, ug, lbl = mod.load_dataset_dispatch("compas", cfg)
    assert lbl == "two_year_recid"

    import pytest
    with pytest.raises(ValueError):
        mod.load_dataset_dispatch("unknown", cfg)
