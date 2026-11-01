"""Flory Scaling Exponent Estimation from Ensemble Averages."""

import numpy as np
from typing import Sequence, Tuple


def estimate_flory_exponent(
    lengths: Sequence[int],
    mean_rg_sq_values: Sequence[float],
) -> Tuple[float, float]:
    """
    Fit Flory scaling relation: <R_g^2> ~ N^(2 * nu).
    ln(<R_g^2>) = 2 * nu * ln(N) + C
    Returns: (nu_flory, r_squared)
    """
    n_arr = np.asarray(lengths, dtype=np.float64)
    rg2_arr = np.asarray(mean_rg_sq_values, dtype=np.float64)

    if len(n_arr) < 2:
        raise ValueError("Need at least 2 data points to fit Flory scaling")

    log_n = np.log(n_arr)
    log_rg2 = np.log(rg2_arr)

    # Linear least squares
    slope, intercept = np.polyfit(log_n, log_rg2, 1)
    nu = slope / 2.0

    # R^2 determination coefficient
    fitted = slope * log_n + intercept
    ss_res = np.sum((log_rg2 - fitted) ** 2)
    ss_tot = np.sum((log_rg2 - np.mean(log_rg2)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    return float(nu), float(r_squared)
