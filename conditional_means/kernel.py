"""Implements kernel-based nonparametric regression estimators"""

import warnings

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()

nprobust = importr("nprobust")


def nprobust_lpbwselect(
    x,
    y,
    p=1,
    n_real=None,
    kernel="epa",
    bwselect="imse-dpi",
    full_return=False,
    **kwargs,
):
    """
    Wraps the nprobust::lpbwselect function for local polynomial bandwidth selection.
    Arguments:
    x: array-like
        The independent variable
    y: array-like
        The dependent variable
    p: int
        The polynomial order (odd order is preferred)
    n_real: int
        If using a subset of the data for bandwidth selection, then n_real is the total
        number of observations. The bandwidth is adjusted by (n_real/len(x)) ** exponent
    kernel: str
        The kernel to use for the local polynomial regression. Defaults to Epanechnikov.
    bwselect: str
        The bandwidth selection criterion. Defaults to imse-dpi. See Calonico et al (JSS,
        2019)
    full_return: bool
        If True, returns the bandwidth, kernel, and bias term. Otherwise, only returns the
        bandwidth and kernel
    """
    if p % 2 == 0:
        exponent = -1 / (5 + 2 * p)
        warnings.warn(
            "The polynomial order is even in local polynomial regression "
            "bandwidth selection assumes all points are interior and optimizes integrated MSE. "
            "However, due to boundary bias, the sup-norm error of the estimator may not "
            "converge at the optimal rate. "
            "See Calonico et al JSS 2019."
        )
    else:
        exponent = -1 / (3 + 2 * p)

    if len(x) > 40:
        obj = nprobust.lpbwselect(
            x=ro.FloatVector(x),
            y=ro.FloatVector(y),
            p=p,
            deriv=0,
            kernel=kernel,
            bwselect=bwselect,
            interior=True,
            **kwargs,
        )
        p, h, b = dict(obj.items())["bws"][0]
    else:
        raise ValueError(f"Too few observations for bandwidth selection. {x}")
        warnings.warn("Too few observations for bandwidth selection, using Silverman's rule.")
        h, b = 1.06 * np.std(x) * (len(x) + 3) ** (-1 / 5), 0

    if n_real is not None:
        h_const = h / (len(x) ** exponent)
        h = h_const * n_real**exponent

    if full_return:
        return (h, kernel, b)
    return (h, kernel)


def local_polynomial_smoothing(y, x, h, evals=None, p=1, effective_sample_size=False, kernel="epa"):
    """
    Vectorized local polynomial smoothing

    Arguments:
    y: array-like
        The dependent variable
    x: array-like
        The independent variable
    h: float
        The bandwidth
    evals: array-like
        Points at which to evaluate the fitted values. If None, defaults to x
    p: int
        The polynomial order
    effective_sample_size: bool
        If True, returns the effective sample size. That is, for a linear smoother yhat = Wy,
        the effective sample size is (1 / (W**2).sum(1)).mean(). See `compute_effective_sample_size`
    """
    kernels = {
        "epa": lambda x: np.maximum(0, (1 - x**2) * 3 / 4),
        "gau": lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi),
        "tri": lambda x: np.maximum(0, 1 - np.abs(x)),
        "uni": lambda x: np.where(np.abs(x) <= 1, 0.5, 0),
    }

    if evals is None:
        evals = x

    if kernel not in kernels:
        warnings.warn("Unknown kernel, using epanechnikov kernel instead.")
        kernel = "epa"

    xx = np.array([x**v for v in range(p + 1)]).T
    evals_xx = np.array([evals**v for v in range(p + 1)]).T
    xxt_matrices = xx[:, :, None] * xx[:, None, :]  # xxt_matrices[i] = xi xi'
    dist_mat = (evals[None, :] - x[:, None]) / h
    kernel_mat = kernels[kernel](dist_mat)

    # the dimensions are (sample point, evaluation point, d1, d2), summed along sample point
    summed_xxt_matrices = (kernel_mat[:, :, None, None] * xxt_matrices[:, None, :, :]).sum(0)
    try:
        inv_terms = np.linalg.inv(summed_xxt_matrices)
        singular = np.zeros(len(evals), dtype=bool)
    except np.linalg.LinAlgError:
        # On rare occasions, the matrix is singular, usually because the number of
        # in-bandwidth observations is too small. In this case, use the pseudo-inverse
        # and set the singular flag to True
        inv_terms = np.linalg.pinv(summed_xxt_matrices)
        singular = np.linalg.matrix_rank(summed_xxt_matrices) < p + 1

    # Average effective sample size calculation
    # (sample point, evaluation point, d1)
    weighted_x = kernel_mat[:, :, None] * xx[:, None, :]
    sum_sq_weight, eff_sample_size = None, None
    if effective_sample_size:
        sum_sq_weight, eff_sample_size = compute_effective_sample_size(
            evals_xx, inv_terms, weighted_x
        )

    # the dimensions are (sample point, evaluation point, d1), summed along sample point
    summed_xty_vectors = (weighted_x * y[:, None, None]).sum(0)
    coefs = (inv_terms * summed_xty_vectors[:, :, None]).sum(1)
    fitted = (coefs * evals_xx).sum(1)

    return (
        fitted,
        {
            "singular": singular,
            "effective_sample_size": eff_sample_size,
            "sum_squared_weight_vector": sum_sq_weight,
        },
    )


def compute_effective_sample_size(evals_xx, inv_terms, weighted_x):
    """
    Compute the average effective sample size, defined as the
    inverse of the sum of squared weights. That is, for a linear smoother yhat = Wy,
    the effective sample size is the (1 / (W**2).sum(1)).mean().

    The rationale is the following. Consider yh(j) = w_j'y where w_j'1 = 1.
    Its variance is sigma^2 w_j'w_j if the y's are homoskedastic. If we were averaging
    p terms, the variance would be sigma^2 / p. Hence p ~ 1/(w_j'w_j).
    """
    # (evaluation point, d1, d2)
    middle = (weighted_x[:, :, None, :] * weighted_x[:, :, :, None]).sum(0)
    sandwich = np.matmul(np.matmul(inv_terms, middle), inv_terms)

    # (evaluation point, d2) x (evaluation point, d2)
    sum_sq_weight = ((sandwich * evals_xx[:, :, None]).sum(1) * evals_xx).sum(1)
    effective_sample_size = (1 / sum_sq_weight).mean()
    return sum_sq_weight, effective_sample_size


def fast_bandwidth_select(x, y, truncate, frac=0.1, **kwargs):
    """Speed up bandwidth selection by only using a fraction of the data. It is often a
    good idea to truncate values of x that are large in this application (since the
    standard error data is skewed)"""

    # truncate to the `truncate` quantile and sample a fraction of the data
    idx = (x <= np.quantile(x, truncate)) & (np.random.RandomState(1231451).rand(len(y)) < frac)

    # select bandwidth using a subset of the data
    bw_obj = nprobust_lpbwselect(x=x[idx], y=y[idx], p=1, n_real=len(x), **kwargs)
    return bw_obj, idx


def ucb_fast(
    y,
    x,
    frac=0.1,
    kernel="epa",
    bwselect="imse-dpi",
    coverage=0.95,
    truncate=0.99,
    ngrid=100,
    **bw_kwargs,
):
    """
    A "poor man's" uniform confidence bands for the conditional mean by computing the max-t
    uniform confidence band for a finite number of grid points. This is a fast
    implementation wrapping the nprobust::lprobust function in R.
    """
    (h, kernel, b), _ = fast_bandwidth_select(
        x, y, truncate, frac=frac, kernel=kernel, bwselect=bwselect, full_return=True, **bw_kwargs
    )

    result = nprobust.lprobust(
        x=ro.FloatVector(x),
        y=ro.FloatVector(y),
        kernel=kernel,
        h=h,
        b=b,
        deriv=0,
        neval=ngrid,
        covgrid=True,
    )

    np_estimates = pd.DataFrame(
        result.rx["Estimate"][0],
        columns=["x", "h", "b", "N", "m_us", "tau_bc", "se_us", "se_rb"],
    )
    cov = np.array(result.rx["cov.rb"][0])
    point_estimates = np_estimates["tau_bc"].values
    corr_mat = cov / np.sqrt(np.diag(cov)[:, None]) / np.sqrt(np.diag(cov)[None, :])
    max_t_crit = np.quantile(
        np.abs(np.random.multivariate_normal(np.zeros(len(cov)), corr_mat, size=10000)).max(1),
        coverage,
    )
    return np_estimates, point_estimates, cov, max_t_crit


def local_linear_regression_fast(
    y,
    x,
    frac=0.1,
    kernel="epa",
    bwselect="imse-dpi",
    truncate=0.99,
    ngrid=500,
    effective_sample_size=False,
    **bw_kwargs,
):
    """
    Local linear regression with fast bandwidth selection.
    We compute the local linear regression at fixed grid points and interpolate linearly
    to obtain fitted values at all x values.
    """
    (optimal_bw, _), idx = fast_bandwidth_select(
        x, y, truncate, frac=frac, kernel=kernel, bwselect=bwselect, full_return=False, **bw_kwargs
    )

    if ngrid is not None:
        range_x = (x.min(), x.max())
        grid = np.linspace(*range_x, ngrid)
    else:
        grid = np.r_[[x.min()], sorted(x[idx]), [x.max()]]

    fitted_grid, meta = local_polynomial_smoothing(
        y,
        x,
        h=optimal_bw,
        evals=grid,
        p=1,
        kernel=kernel,
        effective_sample_size=effective_sample_size,
    )
    singular = meta["singular"]
    if singular.any():
        warnings.warn(
            f"Singular matrix encountered, dropping those grid points ({singular.mean()}) from interpolation."
        )

    # Linearly interpolate the grid values to obtain the fitted function at x values
    fitted_func = np.interp(x, grid[~singular], fitted_grid[~singular])

    if effective_sample_size:
        fitted_eff_sample_size = np.interp(
            x, grid[~singular], 1 / meta["sum_squared_weight_vector"][~singular]
        )  # Linearly interpolate grid points to obtain the effective sample size vector at x's
        return fitted_func, {
            "effective_sample_size": meta["effective_sample_size"],
            "effective_sample_size_vector": 1 / meta["sum_squared_weight_vector"],
            "fitted_effective_sample_size": fitted_eff_sample_size,
            "singular": singular,
        }
    else:
        return fitted_func


def local_linear_regression(
    y,
    x,
    kernel="epa",
    bwselect="imse-dpi",
    effective_sample_size=False,
    **bw_kwargs,
):
    """
    The standard local linear regression estimator
    """
    optimal_bw, _ = nprobust_lpbwselect(
        x=x, y=y, p=1, kernel=kernel, bwselect=bwselect, **bw_kwargs
    )
    fitted, meta = local_polynomial_smoothing(
        y, x, h=optimal_bw, p=1, kernel=kernel, effective_sample_size=effective_sample_size
    )
    singular = meta["singular"]
    eff_sample_size_vec = 1 / meta["sum_squared_weight_vector"]
    if singular.any():
        warnings.warn(f"Singular matrix encountered, linearly interpolating ({singular.mean()})")
        idx = x.argsort()
        fitted_sorted = fitted[idx][~singular]
        x_sorted = x[idx][~singular]
        fitted_func = np.interp(x, x_sorted, fitted_sorted)
        fitted_eff_sample_size = np.interp(x, x_sorted, eff_sample_size_vec[idx][~singular])
    else:
        fitted_func = fitted
        fitted_eff_sample_size = eff_sample_size_vec

    if effective_sample_size:
        return fitted_func, {
            "effective_sample_size": meta["effective_sample_size"],
            "effective_sample_size_vector": eff_sample_size_vec,
            "fitted_effective_sample_size": fitted_eff_sample_size,
        }
    else:
        return fitted_func


def local_linear_regression_conditional_moments(
    estimates,
    standard_errors,
    kernel="epa",
    bwselect="imse-dpi",
    fast=True,
    variance_fit_type="squared_residual",  # `squared_residual` or `fit_conditional` or `difference_of_squares`
    truncation_type="truncate",  # truncate or none
    **kwargs,
):
    """
    Fits conditional moments for `estimates` given `standard_errors` using local linear
    regression.
    1. Takes log(SE)
    2. Fits the conditional mean using LLR directly
    3. There are a few analogue estimators for Var(estimate | log SE) - SE^2
        a. `squared_residual`: Var(estimate | log SE) = E[(estimate - E(estimate | log
           SE))^2]
        b. `fit_conditional`: Fit E((Y-E(Y | sigma))^2 - sigma^2 | sigma)
        c. `difference_of_squares`: Var(estimate | log SE) = E(estimate^2 | sigma) -
           E(estimate | sigma)^2
        The default is `squared_residual`
    """
    regression = local_linear_regression_fast if fast else local_linear_regression
    log_se = np.log10(standard_errors)
    conditional_mean, _ = regression(
        estimates, log_se, kernel=kernel, bwselect=bwselect, effective_sample_size=True, **kwargs
    )

    if variance_fit_type not in {"squared_residual", "fit_conditional", "difference_of_squares"}:
        warnings.warn(
            f"variance_fit_type {variance_fit_type} not recognized, using `squared_residual`"
        )
        variance_fit_type = "squared_residual"

    if variance_fit_type == "fit_conditional":
        # Fit E((X-EX)^2 - sigma^2 | sigma)
        difference = (estimates - conditional_mean) ** 2 - standard_errors**2
        smoothed_cond_var, effective_sample_size_dict = regression(
            difference,
            log_se,
            kernel=kernel,
            bwselect=bwselect,
            effective_sample_size=True,
            **kwargs,
        )
    elif variance_fit_type == "squared_residual":
        # Use Var(X) = E[(X-EX)^2]
        squared_residuals = (estimates - conditional_mean) ** 2
        smoothed_squared_residual, effective_sample_size_dict = regression(
            squared_residuals,
            log_se,
            kernel=kernel,
            bwselect=bwselect,
            effective_sample_size=True,
            **kwargs,
        )
    else:
        # Use Var(X) = E(X^2) - (EX)^2
        squared = estimates**2
        smoothed_squared, effective_sample_size_dict = regression(
            squared, log_se, kernel=kernel, bwselect=bwselect, effective_sample_size=True, **kwargs
        )
        smoothed_squared_residual = smoothed_squared - conditional_mean**2

    effective_sample_size = effective_sample_size_dict["effective_sample_size"]

    if variance_fit_type == "fit_conditional":
        conditional_var = np.maximum(smoothed_cond_var, 0)
        smoothed_squared_residual = smoothed_cond_var + standard_errors**2
    else:
        conditional_var = np.maximum(smoothed_squared_residual - standard_errors**2, 0)

    if truncation_type is not None:
        conditional_var = truncate_conditional_var(
            conditional_var,
            standard_errors,
            truncation_type,
            smoothed_squared_residual,
            effective_sample_size,
        )
    conditional_std = np.sqrt(conditional_var)
    return conditional_mean, conditional_std


def truncate_conditional_var(
    conditional_var,
    standard_errors,
    truncation_type,
    smoothed_squared_residual,
    effective_sample_size,
):
    """Implement truncation of the conditional variance at zero in a data-driven manner"""
    if truncation_type is not None and truncation_type != "truncate":
        warnings.warn(f"Invalid truncation type `{truncation_type}`; Default to default truncation")
        truncation_type = "truncate"

    if truncation_type is not None:
        # Expression (1.6) in Estimation of
        # noncentrality parameters by Kubokawa, Robert, and Saleh
        # (Canadian Journal of Statistics, 1993)
        truncation_factor = 2 / (effective_sample_size + 2)
        truncation_point = truncation_factor * np.maximum(
            smoothed_squared_residual, standard_errors.min() ** 2
        )  # protect against the (exceedingly rare) case where smoothed_squared_residual < 0

        conditional_var = np.maximum(conditional_var, truncation_point)
    return conditional_var
