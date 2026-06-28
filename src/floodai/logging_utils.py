"""
Centralized logging and run-provenance tracking.

Every pipeline run writes:
    1. A timestamped log file under <output_dir>/logs/
    2. A run_manifest.json capturing: config hash, package versions, git
       commit (if available), random seed, start/end time.

This is what makes "every experiment must be logged" an enforced behavior
rather than a documentation claim.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(output_dir: Path, run_name: str) -> logging.Logger:
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger("floodai")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logging initialized -> %s", log_path)
    return logger


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]


def write_run_manifest(
    output_dir: Path,
    run_name: str,
    config_path: Path,
    random_seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON manifest capturing everything needed to reproduce this run."""
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "config_sha256_16": _config_hash(config_path),
        "random_seed": random_seed,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import xgboost, sklearn, numpy, pandas
        manifest["package_versions"] = {
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        }
    except ImportError:
        manifest["package_versions"] = "unavailable (import failed at manifest time)"

    if extra:
        manifest["extra"] = extra

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = manifest_dir / f"{run_name}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return path
