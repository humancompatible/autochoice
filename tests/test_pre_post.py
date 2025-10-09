# test_pre_post.py
import pytest
from conftest import FakeStdDataset

def test_apply_preprocessing_reweighing(monkeypatch, import_module):
    mod = import_module
    mod.privileged_groups = [{"sex": 1}]
    mod.unprivileged_groups = [{"sex": 0}]

    called = {"fit_transform": 0}

    class DummyReweighing:
        def __init__(self, pg, ug): pass
        def fit_transform(self, ds):
            called["fit_transform"] += 1
            return "REWEIGHED"

    monkeypatch.setattr(mod, "Reweighing", DummyReweighing)
    out = mod.apply_preprocessing("Reweighing", FakeStdDataset([[0]], [0]))
    assert out == "REWEIGHED"
    assert called["fit_transform"] == 1

def test_apply_preprocessing_unsupported(import_module):
    mod = import_module
    with pytest.raises(ValueError):
        mod.apply_preprocessing("UnknownAlgo", FakeStdDataset([[0]], [0]))

def test_apply_postprocessing_variants(monkeypatch, import_module):
    mod = import_module
    mod.privileged_groups = [{"sex": 1}]
    mod.unprivileged_groups = [{"sex": 0}]

    chosen = {"cls": None}

    class _Base:
        def __init__(self, *a, **k): pass
        def fit(self, a, b): pass
        def predict(self, ds): return "POST_" + chosen["cls"]

    class EqOdds(_Base): pass
    class CalEqOdds(_Base): pass
    class ROC(_Base): pass

    def wrap(clsname, cls):
        def ctor(*a, **k):
            chosen["cls"] = clsname
            return cls()
        return ctor

    monkeypatch.setattr(mod, "EqOddsPostprocessing", wrap("EqOddsPostprocessing", EqOdds))
    monkeypatch.setattr(mod, "CalibratedEqOddsPostprocessing", wrap("CalibratedEqOddsPostprocessing", CalEqOdds))
    monkeypatch.setattr(mod, "RejectOptionClassification", wrap("RejectOptionClassification", ROC))

    ds_true = FakeStdDataset([[0]], [0])
    ds_pred = FakeStdDataset([[0]], [0])

    assert mod.apply_postprocessing("EqOddsPostprocessing", ds_true, ds_pred) == "POST_EqOddsPostprocessing"
    assert mod.apply_postprocessing("CalibratedEqOddsPostprocessing", ds_true, ds_pred) == "POST_CalibratedEqOddsPostprocessing"
    assert mod.apply_postprocessing("RejectOptionClassification", ds_true, ds_pred) == "POST_RejectOptionClassification"

    with pytest.raises(ValueError):
        mod.apply_postprocessing("Nope", ds_true, ds_pred)
