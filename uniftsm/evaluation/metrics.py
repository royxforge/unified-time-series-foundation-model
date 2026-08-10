"""Forecasting metrics for rigorous evaluation.

Implements scale-dependent, percentage, and scaled error metrics
commonly used in the M-competitions and TSFM literature.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray | None = None,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    The MASE normalises the MAE by the in-sample MAE of a naive
    (seasonal) random walk forecast.  Values < 1 indicate the
    forecast outperforms the naive baseline.

    Args:
        y_true: Ground truth values.
        y_pred: Forecast values.
        y_insample: In-sample training data for scaling.  If ``None``,
            uses ``y_true`` with a lag of ``seasonality``.
        seasonality: Seasonal period. 1 = non-seasonal naive.

    Returns:
        MASE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mae = np.mean(np.abs(y_true - y_pred))

    if y_insample is not None:
        y_insample = np.asarray(y_insample, dtype=np.float64).ravel()
        naive_errors = np.abs(np.diff(y_insample, n=seasonality))
    else:
        naive_errors = np.abs(np.diff(y_true, n=seasonality))

    naive_mae = np.mean(naive_errors)
    if naive_mae < 1e-8:
        return 0.0 if mae < 1e-8 else 1e6

    return float(mae / naive_mae)


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Expressed as a percentage.  Lower is better.

    Args:
        y_true: Ground truth values.
        y_pred: Forecast values.

    Returns:
        sMAPE value (percentage).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-8
    if not mask.any():
        return 0.0
    return float(200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))


def crps(
    y_true: np.ndarray,
    samples: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> float:
    """Continuous Ranked Probability Score.

    Measures probabilistic forecast quality.  Lower is better.
    Computed from either Monte Carlo samples or a Gaussian
    parametric assumption.

    Args:
        y_true: Ground truth values.
        samples: Monte Carlo samples, shape ``(n_samples, horizon)``.
        mean: Forecast mean (used if ``samples`` is ``None``).
        std: Forecast std (used if ``samples`` is ``None``).

    Returns:
        CRPS value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    if samples is not None:
        samples = np.asarray(samples, dtype=np.float64)
        # CRPS via empirical CDF approximation
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            obs = y_true[i]
            samps = samples[:, i]
            scores[i] = np.mean(np.abs(samps - obs)) - 0.5 * np.mean(
                np.abs(samps[:, None] - samps[None, :])
            )
        return float(np.mean(scores))

    if mean is not None and std is not None:
        mean = np.asarray(mean, dtype=np.float64).ravel()
        std = np.asarray(std, dtype=np.float64).ravel()
        z = (y_true - mean) / (std + 1e-8)
        score = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        return float(np.mean(score))

    raise ValueError("Either 'samples' or ('mean' and 'std') must be provided.")


def nmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_insample: np.ndarray,
) -> float:
    """Normalised Mean Absolute Error (MAE / mean of in-sample data).

    Args:
        y_true: Ground truth values.
        y_pred: Forecast values.
        y_insample: In-sample training data.

    Returns:
        NMAE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_insample = np.asarray(y_insample, dtype=np.float64).ravel()
    scale = np.mean(np.abs(y_insample))
    if scale < 1e-8:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def owa(
    mase_value: float,
    smape_value: float,
    mase_naive: float = 1.0,
    smape_naive: float = 100.0,
) -> float:
    """Overall Weighted Average (from M4 competition).

    Combines MASE and sMAPE into a single score, weighted equally.
    Values < 1 indicate better-than-naive performance.

    Args:
        mase_value: MASE of the forecast.
        smape_value: sMAPE of the forecast.
        mase_naive: MASE of the naive baseline (default 1.0).
        smape_naive: sMAPE of the naive baseline (default 100.0).

    Returns:
        OWA value.
    """
    return float((mase_value / mase_naive + smape_value / smape_naive) / 2.0)


def mrae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> float:
    """Mean Relative Absolute Error.

    Ratio of the forecast MAE to the baseline MAE.  Values < 1
    indicate improvement over the baseline.

    Args:
        y_true: Ground truth values.
        y_pred: Forecast values.
        y_pred_baseline: Baseline forecast values.

    Returns:
        MRAE value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=np.float64).ravel()

    forecast_ae = np.abs(y_true - y_pred)
    baseline_ae = np.abs(y_true - y_pred_baseline)
    non_zero = baseline_ae > 1e-8
    if not non_zero.any():
        return 1.0
    return float(np.mean(forecast_ae[non_zero] / baseline_ae[non_zero]))


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage of prediction intervals.

    Proportion of true values that fall within the interval [lower, upper].

    Args:
        y_true: Ground truth values.
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        Coverage proportion in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    inside = np.logical_and(y_true >= lower, y_true <= upper)
    return float(np.mean(inside))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    y_insample: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    """Compute all evaluation metrics at once.

    Args:
        y_true: Ground truth.
        y_pred: Point forecasts.
        y_insample: In-sample data (for MASE, NMAE).
        samples: MC samples (for CRPS).
        mean: Forecast mean (for CRPS if ``samples`` absent).
        std: Forecast std (for CRPS if ``samples`` absent).
        lower: Lower interval bounds (for coverage).
        upper: Upper interval bounds (for coverage).
        seasonality: Seasonal period for MASE.

    Returns:
        Dictionary of metric name → value.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    results: dict[str, float] = {}

    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        results["MASE"] = mase(y_true, y_pred, y_insample, seasonality)
        results["sMAPE"] = smape(y_true, y_pred)
        if y_insample is not None:
            results["NMAE"] = nmae(y_true, y_pred, y_insample)
            results["OWA"] = owa(results["MASE"], results["sMAPE"])

    if samples is not None or (mean is not None and std is not None):
        results["CRPS"] = crps(y_true, samples=samples, mean=mean, std=std)

    if lower is not None and upper is not None:
        results["Coverage"] = coverage(y_true, lower, upper)

    if y_pred is not None:
        results["MAE"] = float(np.mean(np.abs(y_true - y_pred)))
        results["RMSE"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return results
