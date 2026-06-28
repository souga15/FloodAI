"""
Tests for floodai.config — including a regression test that specifically
guards against reintroducing the WPT-D-26-00166 Cell 23 issue (a
training-data-inclusive bootstrap reported as if it were a held-out metric).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from floodai.config import ConfigError, load_config


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    base = {
        "experiment": {"id": "test", "random_seed": 42, "output_dir": "results/test"},
        "basins": {
            "b1": {"label": "Basin 1", "bbox": {"lat_min": 0, "lat_max": 1, "lon_min": 0, "lon_max": 1},
                   "n_points_target": 10, "tier_prior": 1},
        },
        "split": {"train_years": [2017, 2018], "val_years": [2019], "test_years": [2020]},
        "reporting": {
            "headline_metric_source": "test_set_only",
            "forbid_train_inclusive_headline": True,
            "bootstrap": {"enabled": True, "n_resamples": 1000, "resample_from": "test_set_only"},
        },
    }
    for section, vals in overrides.items():
        base[section].update(vals)
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(base, f)
    return path


class TestConfigValidation:
    def test_valid_config_loads(self, tmp_path):
        path = _write_config(tmp_path, {})
        cfg = load_config(path)
        assert cfg.experiment_id == "test"
        assert "b1" in cfg.basins

    def test_overlapping_split_years_rejected(self, tmp_path):
        path = _write_config(tmp_path, {"split": {"train_years": [2017, 2018], "val_years": [2018], "test_years": [2020]}})
        with pytest.raises(ConfigError, match="overlap"):
            load_config(path)

    def test_zero_basins_rejected(self, tmp_path):
        path = _write_config(tmp_path, {})
        data = yaml.safe_load(path.read_text())
        data["basins"] = {}
        path.write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="zero basins"):
            load_config(path)

    def test_train_inclusive_bootstrap_combination_is_rejected(self, tmp_path):
        """Regression test for the exact failure mode found in
        WPT-D-26-00166 Cell 23: a config that claims to forbid train-inclusive
        headline metrics, but configures the bootstrap to resample from the
        full (training-inclusive) dataset anyway, must fail loudly at config
        load time -- not silently produce an inflated number downstream."""
        path = _write_config(tmp_path, {
            "reporting": {
                "headline_metric_source": "test_set_only",
                "forbid_train_inclusive_headline": True,
                "bootstrap": {"enabled": True, "n_resamples": 1000, "resample_from": "full_dataset"},
            }
        })
        with pytest.raises(ConfigError, match="train-inclusive"):
            load_config(path)

    def test_missing_required_section_raises(self, tmp_path):
        path = _write_config(tmp_path, {})
        data = yaml.safe_load(path.read_text())
        del data["split"]
        path.write_text(yaml.dump(data))
        with pytest.raises(ConfigError, match="Missing required"):
            load_config(path)
