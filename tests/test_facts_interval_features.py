# test_facts_interval_features.py
import numpy as np
import pandas as pd

class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)

def test_ensure_interval_features_bins_and_synthesis(import_module):
    mod = import_module

    # Existing numeric 'age' -> binned categorical
    X = pd.DataFrame({"age": [18, 25, 37, 61], "sex": ["M", "F", "F", "M"]})
    cfg = _Cfg({"facts": {"ensure_interval_cols": ["age"], "default_n_bins": 3}})
    Xc, cats, bins_map, created = mod._ensure_interval_features_for_facts(X, cfg)
    assert "age" in Xc.columns and pd.api.types.is_categorical_dtype(Xc["age"])
    assert "age" in bins_map and isinstance(bins_map["age"], list)
    assert created == []
    assert "sex" in cats and "age" in cats

    # Missing 'age' -> synthesized with provided bins
    X2 = pd.DataFrame({"sex": ["M", "F", "F", "M"]})
    cfg2 = _Cfg({"facts": {
        "ensure_interval_cols": ["age"],
        "create_missing_interval_cols": True,
        "interval_bins": {"age": [-np.inf, 20, 30, 40, np.inf]},
    }})
    Xc2, cats2, bins_map2, created2 = mod._ensure_interval_features_for_facts(X2, cfg2)
    assert "age" in Xc2.columns and pd.api.types.is_categorical_dtype(Xc2["age"])
    assert created2 == ["age"]
    assert "age" in cats2
