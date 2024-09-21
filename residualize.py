"""Residualize a variable on a set of covariates, optionally within CZs."""

import pyfixest as pf
import numpy as np


def residualize(df, est_var, covariates, weighted=True, czvar="czname", within_cz=False):
    """Residualize `est_var` on `covariates`, optionally within CZs."""
    df = df.copy()
    se_var = est_var + "_se"

    if weighted:
        df["w"] = 1 / (df[se_var] ** 2)
    weights = "w" if weighted else None
    fmla = f"{est_var} ~ 1 + {' + '.join(covariates)}"

    if not within_cz:
        fit = pf.feols(fmla, weights=weights, data=df)
        residuals = fit.resid()
        fitted_values = fit.predict()
    else:
        # Residualize within each CZ
        residuals = np.full_like(df[est_var], np.nan)
        fitted_values = np.full_like(df[est_var], np.nan)
        for cz in df[czvar].unique():
            mask = df[czvar] == cz
            fit = pf.feols(fmla, weights=weights, data=df.loc[mask])
            residuals[mask] = fit.resid()
            fitted_values[mask] = fit.predict()

    return fitted_values, residuals
