#!/usr/bin/env python3
"""Generate the pickled fixture file tests/fixtures/precomputed_forecasts.pkl.

Usage:
    python scripts/generate_fixture_pkl.py

This produces a small pickle (~2 KB) containing sample forecast DataFrames
that tests can load for quick validation without running any model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _make_forecast_output(horizon: int, seed: int = 42) -> dict:
    """Build a dict with the same structure as BaseForecaster.predict()."""
    rng = np.random.default_rng(seed)
    mean = 100 + np.arange(horizon) * 0.5 + rng.normal(0, 0.1, horizon)
    std = 1.0 + np.abs(rng.normal(0, 0.2, horizon))
    index = pd.RangeIndex(horizon)

    simple = pd.DataFrame({"mean": mean, "std": std}, index=index)

    quantile_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    from scipy.stats import norm
    quantiles = {}
    for q in quantile_vals:
        z = norm.ppf(q)
        quantiles[f"q_{q}"] = pd.DataFrame(
            mean + z * std, index=index, columns=["forecast"]
        )

    return {"simple": simple, "quantiles": quantiles}


def _make_evaluation_output() -> dict:
    """Return a metrics dict matching compute_metrics() output."""
    return {
        "MASE": 0.85,
        "sMAPE": 1.45,
        "CRPS": 0.72,
        "NMAE": 0.015,
        "OWA": 0.52,
        "Coverage": 0.88,
        "MAE": 1.23,
        "RMSE": 1.56,
        "MRAE": 0.62,
    }


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "horizon_10": _make_forecast_output(10),
        "horizon_24": _make_forecast_output(24),
        "horizon_48": _make_forecast_output(48),
        "metrics": _make_evaluation_output(),
    }

    path = FIXTURES_DIR / "precomputed_forecasts.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = path.stat().st_size / 1024
    keys = list(data.keys())
    print(f"Generated {path} ({size_kb:.1f} KB)")
    print(f"Top-level keys: {keys}")


if __name__ == "__main__":
    main()
