"""
Configuration loading and validation for FloodAI.

Design intent: every numeric/string parameter that can affect a reported
result must originate from config/config.yaml, never be hard-coded in a
module. This loader fails loudly (raises) rather than silently defaulting,
because a silent default is exactly the kind of thing that produces an
unreproducible result six months later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("floodai.config")


class ConfigError(Exception):
    """Raised when config.yaml is missing required keys or has invalid values."""


@dataclass(frozen=True)
class BasinConfig:
    key: str
    label: str
    bbox: dict[str, float]
    n_points_target: int
    tier_prior: int


@dataclass(frozen=True)
class SplitConfig:
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]

    def __post_init__(self) -> None:
        all_years = self.train_years + self.val_years + self.test_years
        if len(all_years) != len(set(all_years)):
            raise ConfigError(
                "split.train_years / val_years / test_years overlap. "
                "A year must belong to exactly one split to avoid temporal leakage."
            )


@dataclass(frozen=True)
class ReportingConfig:
    headline_metric_source: str
    forbid_train_inclusive_headline: bool
    bootstrap_enabled: bool
    bootstrap_n_resamples: int
    bootstrap_resample_from: str

    def __post_init__(self) -> None:
        if self.forbid_train_inclusive_headline and self.bootstrap_resample_from != "test_set_only":
            raise ConfigError(
                "reporting.forbid_train_inclusive_headline is True but "
                "reporting.bootstrap.resample_from is not 'test_set_only'. "
                "This combination would silently re-introduce a train-inclusive "
                "headline number, which is the exact issue this framework was "
                "built to avoid. Fix config.yaml."
            )


@dataclass(frozen=True)
class FloodAIConfig:
    raw: dict[str, Any] = field(repr=False)
    experiment_id: str
    random_seed: int
    output_dir: Path
    basins: dict[str, BasinConfig]
    split: SplitConfig
    reporting: ReportingConfig

    @property
    def all_years(self) -> list[int]:
        return self.split.train_years + self.split.val_years + self.split.test_years


def load_config(path: str | Path) -> FloodAIConfig:
    """Load and validate config.yaml. Raises ConfigError on any structural problem."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    try:
        exp = raw["experiment"]
        basins_raw = raw["basins"]
        split_raw = raw["split"]
        rep_raw = raw["reporting"]
    except KeyError as e:
        raise ConfigError(f"Missing required top-level config section: {e}") from e

    basins = {
        key: BasinConfig(
            key=key,
            label=b["label"],
            bbox=b["bbox"],
            n_points_target=b["n_points_target"],
            tier_prior=b["tier_prior"],
        )
        for key, b in basins_raw.items()
    }
    if len(basins) == 0:
        raise ConfigError("config.yaml defines zero basins; at least one is required.")

    split = SplitConfig(
        train_years=split_raw["train_years"],
        val_years=split_raw["val_years"],
        test_years=split_raw["test_years"],
    )

    reporting = ReportingConfig(
        headline_metric_source=rep_raw["headline_metric_source"],
        forbid_train_inclusive_headline=rep_raw["forbid_train_inclusive_headline"],
        bootstrap_enabled=rep_raw["bootstrap"]["enabled"],
        bootstrap_n_resamples=rep_raw["bootstrap"]["n_resamples"],
        bootstrap_resample_from=rep_raw["bootstrap"]["resample_from"],
    )

    cfg = FloodAIConfig(
        raw=raw,
        experiment_id=exp["id"],
        random_seed=exp["random_seed"],
        output_dir=Path(exp["output_dir"]),
        basins=basins,
        split=split,
        reporting=reporting,
    )

    logger.info(
        "Loaded config '%s': %d basins, seed=%d, train=%s val=%s test=%s",
        cfg.experiment_id, len(cfg.basins), cfg.random_seed,
        split.train_years, split.val_years, split.test_years,
    )
    return cfg
