"""Drift baseline — capture and persist the training-time distribution.

The baseline is computed once on the assembled training data, saved to
`models/drift_baseline.json`, and loaded at API startup. Subsequent batches
are compared against it.

The baseline captures, per column:
  - numeric:     mean, std, percentiles, a sample of values for KS testing
  - categorical: counts and proportions per category, plus n_total

The numeric value sample is bounded (default 1000 rows) so the JSON stays
small. KS testing on 1000 reference values is statistically adequate.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.storage.paths import MODELS_DIR

logger = logging.getLogger(__name__)


BASELINE_PATH = MODELS_DIR / "drift_baseline.json"
DEFAULT_NUMERIC_FEATURES = (
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes (years)",
    "Hormonal Contraceptives (years)", "IUD (years)",
    "STDs (number)", "host_prs", "strain_carcinogenicity",
)
DEFAULT_CATEGORICAL_FEATURES = ("assigned_hpv_strain", "matched_super_pop")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(int(p)): float(np.percentile(values, p))
        for p in (0, 10, 25, 50, 75, 90, 100)
    }


def _capture_numeric(series: pd.Series, sample_size: int = 1000) -> dict[str, Any]:
    """Per-numeric-feature stats: mean, std, missing rate, percentiles, sample."""
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) == 0:
        return {"n": 0, "missing_rate": 1.0}

    # Subsample for the KS-test reference if too large
    if len(vals) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(vals), size=sample_size, replace=False)
        sample = vals[idx]
    else:
        sample = vals

    return {
        "n": int(len(vals)),
        "missing_rate": float(series.isna().mean()),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "percentiles": _percentiles(vals),
        "values_sample": sample.tolist(),
    }


def _capture_categorical(series: pd.Series) -> dict[str, Any]:
    """Per-categorical-feature stats: counts, proportions, unique categories."""
    vals = series.dropna().astype(str)
    counts = Counter(vals)
    total = sum(counts.values())
    if total == 0:
        return {"n": 0, "missing_rate": 1.0}
    return {
        "n": int(total),
        "missing_rate": float(series.isna().mean()),
        "value_counts": dict(counts),
        "proportions": {k: v / total for k, v in counts.items()},
        "unique_count": len(counts),
    }


def capture_baseline(
    df: pd.DataFrame,
    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES,
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES,
    source_label: str = "training",
) -> dict[str, Any]:
    """Build the baseline dict from a DataFrame.

    Only columns present in df are recorded; others are skipped with a log line.
    """
    num: dict[str, dict] = {}
    for col in numeric_features:
        if col in df.columns:
            num[col] = _capture_numeric(df[col])
        else:
            logger.debug("baseline: numeric column %s not in df", col)

    cat: dict[str, dict] = {}
    for col in categorical_features:
        if col in df.columns:
            cat[col] = _capture_categorical(df[col])
        else:
            logger.debug("baseline: categorical column %s not in df", col)

    return {
        "version": "0.1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_label": source_label,
        "n_records": int(len(df)),
        "numeric_features": num,
        "categorical_features": cat,
    }


def save_baseline(baseline: dict[str, Any], path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, default=str))
    logger.info("Saved drift baseline to %s (%d numeric, %d categorical features)",
                path, len(baseline.get("numeric_features", {})),
                len(baseline.get("categorical_features", {})))


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning("No drift baseline at %s — call capture+save first", path)
        return None
    return json.loads(path.read_text())


def main() -> None:
    """CLI: capture baseline from the assembled patient table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    from src.storage.files import require_file
    from src.storage.paths import PROCESSED_DIR

    src = PROCESSED_DIR / "patients_assembled.parquet"
    require_file(
        src,
        hint_command="python -m src.features.build_patient",
        description="Assembled patient table from day-1+2 feature engineering",
    )
    df = pd.read_parquet(src)
    baseline = capture_baseline(df, source_label="patients_assembled")
    save_baseline(baseline)

    print(f"\nDrift baseline captured from {len(df)} patients.")
    print(f"  Numeric features:     {len(baseline['numeric_features'])}")
    print(f"  Categorical features: {len(baseline['categorical_features'])}")
    print(f"  Saved to: {BASELINE_PATH}\n")
    print("Sample of categorical baseline:")
    for col, info in baseline["categorical_features"].items():
        top3 = dict(sorted(info["proportions"].items(),
                           key=lambda kv: -kv[1])[:3])
        print(f"  {col:25s} top-3 = {top3}")


if __name__ == "__main__":
    main()
