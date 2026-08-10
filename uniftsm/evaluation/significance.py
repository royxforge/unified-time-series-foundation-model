"""Statistical significance tests for forecast comparison.

Implements the Diebold-Mariano test, Wilcoxon signed-rank test,
and paired bootstrap test for rigorous model comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm, wilcoxon


def diebold_mariano_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
    one_sided: bool = False,
) -> dict[str, Any]:
    """Diebold-Mariano test for predictive accuracy comparison.

    Tests the null hypothesis that two forecasts have equal predictive
    accuracy.  A negative (positive) DM statistic with p-value < 0.05
    indicates model 1 (model 2) is significantly better.

    Args:
        errors_model1: Forecast errors (y_true - y_pred) for model 1.
        errors_model2: Forecast errors (y_true - y_pred) for model 2.
        h: Forecast horizon (for autocorrelation correction).
        loss: Loss function — ``"mse"`` or ``"mae"``.
        one_sided: If ``True``, test model1 < model2.  Two-sided otherwise.

    Returns:
        Dictionary with ``"dm_stat"``, ``"p_value"``, ``"h"``, and
        ``"significant_at_5pct"`` keys.
    """
    e1 = np.asarray(errors_model1, dtype=np.float64).ravel()
    e2 = np.asarray(errors_model2, dtype=np.float64).ravel()

    if len(e1) != len(e2):
        raise ValueError("Error arrays must have the same length.")

    # Loss differential
    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss function: '{loss}'")

    n = len(d)
    mean_d = np.mean(d)

    # Variance with HAC (Newey-West) correction
    gamma_0 = np.var(d, ddof=1)
    if h > 1 and n > h:
        # Autocovariance at lags 1..h-1
        gamma = np.zeros(h - 1)
        for j in range(1, h):
            gamma[j - 1] = np.mean(d[:-j] * d[j:]) - mean_d**2
        var_d = (gamma_0 + 2 * np.sum((1 - np.arange(1, h) / h) * gamma)) / n
    else:
        var_d = gamma_0 / n

    if var_d <= 0:
        var_d = 1e-8

    dm_stat = mean_d / np.sqrt(var_d)

    # Two-sided p-value
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    if one_sided:
        p_value = 1 - norm.cdf(dm_stat)

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "h": h,
        "significant_at_5pct": bool(p_value < 0.05),
        "model1_better": bool(dm_stat < 0),
    }


def wilcoxon_signed_rank_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test (non-parametric paired test).

    Tests whether the median difference between two model's errors is
    non-zero.  Does not assume normality.

    Args:
        errors_model1: Forecast errors for model 1.
        errors_model2: Forecast errors for model 2.

    Returns:
        Dictionary with ``"statistic"``, ``"p_value"``, and
        ``"significant_at_5pct"`` keys.
    """
    e1 = np.asarray(errors_model1, dtype=np.float64).ravel()
    e2 = np.asarray(errors_model2, dtype=np.float64).ravel()

    loss_diff = np.abs(e1) - np.abs(e2)
    # Remove zeros (tied pairs)
    loss_diff = loss_diff[loss_diff != 0]

    if len(loss_diff) < 2:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_5pct": False,
            "insufficient_data": True,
        }

    stat, p_value = wilcoxon(loss_diff)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_5pct": bool(p_value < 0.05),
        "insufficient_data": False,
    }


def paired_bootstrap_test(
    errors_model1: np.ndarray,
    errors_model2: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test for forecast comparison.

    Resamples the loss differences with replacement and computes the
    proportion of times the sign flips.

    Args:
        errors_model1: Forecast errors for model 1.
        errors_model2: Forecast errors for model 2.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with ``"p_value"`` and ``"significant_at_5pct"`` keys.
    """
    e1 = np.asarray(errors_model1, dtype=np.float64).ravel()
    e2 = np.asarray(errors_model2, dtype=np.float64).ravel()
    n = len(e1)

    loss_diff = np.abs(e1) - np.abs(e2)
    observed_mean = np.mean(loss_diff)

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_mean = np.mean(loss_diff[indices])
        if abs(bootstrap_mean) >= abs(observed_mean):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_bootstrap + 1)

    return {
        "p_value": float(p_value),
        "significant_at_5pct": bool(p_value < 0.05),
        "observed_mean_diff": float(observed_mean),
        "model1_better": bool(observed_mean < 0),
    }
