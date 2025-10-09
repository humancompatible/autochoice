# test_load_cfg.py
import pytest

def test__load_cfg_not_found(monkeypatch, import_module):
    mod = import_module
    monkeypatch.setattr("os.path.exists", lambda p: False)
    with pytest.raises(FileNotFoundError):
        mod._load_cfg()
