"""
Closed-form petrophysical baseline models for permeability prediction.

Baselines implemented:
- Log-linear porosity-permeability regression:
    log10(k) = a + b * phi
- Timur-style regression (SWT used as Swirr proxy):
    log10(k) = log_a + b * log10(phi) - c * log10(Swirr)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(values) -> np.ndarray:
    """Convert values to a 1D float numpy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _derive_fill_value(
    values: np.ndarray,
    *,
    require_positive: bool,
    eps: float,
    fallback: float,
) -> float:
    """Derive a robust fill value from valid training observations."""
    valid = np.isfinite(values)
    if require_positive:
        valid &= values > eps
    if np.any(valid):
        return float(np.nanmedian(values[valid]))
    return float(fallback)


def _prepare_feature(
    values,
    *,
    fill_value: float,
    require_positive: bool,
    eps: float,
) -> tuple[np.ndarray, int]:
    """
    Coerce a feature vector to finite values using training-derived fill values.

    Returns
    -------
    (prepared, n_imputed)
        prepared: finite feature vector safe for baseline equations
        n_imputed: number of values replaced by fill_value
    """
    arr = _as_float_array(values)
    invalid = ~np.isfinite(arr)
    if require_positive:
        invalid |= arr <= eps

    if np.any(invalid):
        arr = arr.copy()
        arr[invalid] = fill_value

    if require_positive:
        arr = np.maximum(arr, eps)

    return arr, int(np.sum(invalid))


def fit_log_linear_baseline(
    phi_train,
    y_train_log,
    *,
    phi_name: str = "phi",
) -> dict[str, Any]:
    """
    Fit log-linear permeability baseline: log10(k) = a + b * phi.
    """
    y = _as_float_array(y_train_log)
    phi_raw = _as_float_array(phi_train)
    phi_fill = _derive_fill_value(
        phi_raw,
        require_positive=False,
        eps=1e-10,
        fallback=0.0,
    )
    phi, n_imputed = _prepare_feature(
        phi_raw,
        fill_value=phi_fill,
        require_positive=False,
        eps=1e-10,
    )

    X = np.column_stack([np.ones_like(phi), phi])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(beta[0]), float(beta[1])

    return {
        "baseline_type": "log_linear",
        "phi_name": phi_name,
        "intercept": intercept,
        "slope": slope,
        "phi_fill_value": float(phi_fill),
        "train_n": int(len(y)),
        "train_n_imputed_phi": int(n_imputed),
    }


def predict_log_linear_baseline(phi_values, params: dict[str, Any]) -> np.ndarray:
    """
    Predict log10(k) from a fitted log-linear baseline model.
    """
    phi, _ = _prepare_feature(
        phi_values,
        fill_value=float(params["phi_fill_value"]),
        require_positive=False,
        eps=1e-10,
    )
    return float(params["intercept"]) + float(params["slope"]) * phi


def fit_timur_baseline(
    phi_train,
    swirr_train,
    y_train_log,
    *,
    phi_name: str = "phi",
    swirr_name: str = "swirr",
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Fit Timur-style baseline with linear least squares in transformed space.

    Uses:
        log10(k) = log_a + b * log10(phi) + c * (-log10(swirr))
    which is equivalent to:
        log10(k) = log_a + b * log10(phi) - c * log10(swirr)
    """
    y = _as_float_array(y_train_log)
    phi_raw = _as_float_array(phi_train)
    swirr_raw = _as_float_array(swirr_train)

    phi_fill = _derive_fill_value(
        phi_raw,
        require_positive=True,
        eps=eps,
        fallback=0.10,
    )
    swirr_fill = _derive_fill_value(
        swirr_raw,
        require_positive=True,
        eps=eps,
        fallback=0.50,
    )

    phi, phi_imputed = _prepare_feature(
        phi_raw,
        fill_value=phi_fill,
        require_positive=True,
        eps=eps,
    )
    swirr, swirr_imputed = _prepare_feature(
        swirr_raw,
        fill_value=swirr_fill,
        require_positive=True,
        eps=eps,
    )

    x_phi = np.log10(phi)
    x_neg_log_swirr = -np.log10(swirr)
    X = np.column_stack([np.ones_like(x_phi), x_phi, x_neg_log_swirr])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept, b_phi, c_swirr = float(beta[0]), float(beta[1]), float(beta[2])

    return {
        "baseline_type": "timur",
        "phi_name": phi_name,
        "swirr_name": swirr_name,
        "eps": float(eps),
        "intercept_loga": intercept,
        "b_phi": b_phi,
        "c_swirr": c_swirr,
        "d_log_swirr": -c_swirr,
        "phi_fill_value": float(phi_fill),
        "swirr_fill_value": float(swirr_fill),
        "train_n": int(len(y)),
        "train_n_imputed_phi": int(phi_imputed),
        "train_n_imputed_swirr": int(swirr_imputed),
    }


def predict_timur_baseline(
    phi_values,
    swirr_values,
    params: dict[str, Any],
) -> np.ndarray:
    """
    Predict log10(k) from a fitted Timur-style baseline model.
    """
    eps = float(params.get("eps", 1e-8))
    phi, _ = _prepare_feature(
        phi_values,
        fill_value=float(params["phi_fill_value"]),
        require_positive=True,
        eps=eps,
    )
    swirr, _ = _prepare_feature(
        swirr_values,
        fill_value=float(params["swirr_fill_value"]),
        require_positive=True,
        eps=eps,
    )

    return (
        float(params["intercept_loga"])
        + float(params["b_phi"]) * np.log10(phi)
        + float(params["c_swirr"]) * (-np.log10(swirr))
    )
