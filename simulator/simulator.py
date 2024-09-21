"""Methods for generating a draw from some calibrated prior distribution."""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import gamma
from empirical_bayes.ebmethods import default_v, independent_npmle, normalize_prior
from build_data import identifiers, load_data_for_outcome


def make_simulator(
    simulator_name,
    est_var,
    estimates,
    standard_errors,
    covariate_function,
    conditional_mean,
    conditional_std,
):
    generator_dict = {
        "weibull": weibull_prior_generator,
        "t_distribution": t_distribution_prior_generator,
        "npmle_by_bins": npmle_by_bins_prior_generator,
    }
    kwargs = {}

    if simulator_name == "npmle_by_bins":
        bin_type, priors = fit_npmle_by_bins(
            estimates,
            standard_errors,
            covariate_function,
            conditional_mean,
            conditional_std,
            nbins=20,
        )
        kwargs = {"priors": priors, "bin_type_vec": bin_type}

    if simulator_name.startswith("coupled_bootstrap"):
        if len(simulator_name.split("-")) > 1:
            split = float(simulator_name.split("-")[1])
        else:
            split = 0.9

        def simulator(seed):
            return coupled_bootstrap_prior_generator(
                seed, estimates, standard_errors, covariate_function, split=split
            )

        return simulator

    if simulator_name.startswith("proxy"):
        _, proxy, split = simulator_name.split("-")
        split = float(split)
        alt_est_var = est_var.replace("black_pooled", "white_pooled")
        merged_df, black_weight = compute_merged_df_and_proxy_weight(
            proxy, split, est_var, alt_est_var
        )

        def simulator(seed):
            return coupled_bootstrap_proxy_prior_generator(
                merged_df, est_var, alt_est_var, proxy, black_weight, seed=seed, split=split
            )

        return simulator

    def simulator(seed):
        return simulated_prior(
            estimates,
            standard_errors,
            covariate_function,
            conditional_mean,
            conditional_std,
            generator_dict[simulator_name],
            seed=seed,
            **kwargs
        )

    return simulator


def simulated_prior(
    estimates,
    standard_errors,
    covariate_function,
    conditional_mean,
    conditional_std,
    prior_for_normalized,
    seed=1241251,
    **kwargs
):
    rng = np.random.RandomState(seed)
    n = len(estimates)
    sampled_transformed_parameter = prior_for_normalized(rng, conditional_mean, **kwargs)
    sampled_parameter = conditional_mean + conditional_std * sampled_transformed_parameter
    sampled_noise = rng.randn(n) * standard_errors
    sampled_estimates = sampled_parameter + sampled_noise
    sampled_data = sampled_estimates + covariate_function

    # names starting with _ are information that feasible methods should not have access to
    sampled_obj = {
        "sampled_data": sampled_data,
        # "standard_errors": standard_errors,
        # "_sampled_estimates": sampled_estimates,
        "_sampled_residualized_parameter": sampled_parameter,
        # "_sampled_transformed_parameter": sampled_transformed_parameter,
        # "_covariate_function": covariate_function,
        # "_conditional_mean": conditional_mean,
        # "_conditional_std": conditional_std,
    }

    return sampled_obj


def weibull_prior_generator(rng, conditional_mean, **kwargs):
    """Conditional distribution is a Weibull distribution with shape
    parameter depending on conditional_mean."""
    n = len(conditional_mean)

    # the shape parameter ranges from 1/2 to 1 depending on conditional_mean
    shapes = 1 / 2 + 0.5 * (conditional_mean - conditional_mean.min()) / (
        conditional_mean.max() - conditional_mean.min()
    )
    weibulls = (-np.log(rng.rand(n))) ** (1 / shapes)

    # Normalize to have mean 0 and variance 1
    means = gamma(1 + 1 / shapes)
    variances = gamma(1 + 2 / shapes) - means**2
    sampled_transformed_parameter = (weibulls - means) / variances**0.5
    return sampled_transformed_parameter


def t_distribution_prior_generator(rng, conditional_mean, **kwargs):
    """Conditional distribution is a t-distribution with 4 degrees of freedom."""
    n = len(conditional_mean)

    # normalize to have variance 1
    sampled_transformed_parameter = rng.standard_t(df=4, size=n) / (4 / (4 - 2)) ** 0.5
    return sampled_transformed_parameter


def npmle_by_bins_prior_generator(
    rng,
    conditional_mean,
    priors=None,  # a list of tuples (loc, mass) for each bin
    bin_type_vec=None,  # a vector indicating which bin each observation belongs to
    **kwargs
):
    sampled_transformed_parameter = np.zeros_like(conditional_mean)
    bin_start, bin_end = bin_type_vec.min(), bin_type_vec.max()
    for t in range(bin_start, bin_end + 1):
        idx = bin_type_vec == t
        normalized_parameter_samples = rng.choice(
            priors[t][0], size=idx.sum(), p=priors[t][1], replace=True
        )
        sampled_transformed_parameter[idx] = normalized_parameter_samples

    return sampled_transformed_parameter


def coupled_bootstrap_prior_generator(
    seed, estimates, standard_errors, covariate_function, split=0.9, **kwargs
):
    """
    Simulate data by adding noise to the original estimates.
    Note that Y1 = Y + c * sigma Z and Y2 = Y - 1/c * sigma Z are
    conditionally independent given theta, if
    Y | theta ~ N(theta, sigma^2) and Z ~ N(0,1)

    We engineer a q-train (1-q)-test split by setting c = Sqrt[(1-q)/q].

    Observe that
      E[(µ(Y1) - Y2)^2] = E[(µ(Y1) - theta)^2] + (1/c)^2 sigma^2
    and
      E[a(Y1) Y2] = E[a(Y1) theta]
    """
    rng = np.random.RandomState(seed)

    data = estimates + covariate_function
    c = np.sqrt((1 - split) / split)
    noise = standard_errors * rng.randn(len(estimates))
    sampled_data = data + c * noise
    validation_data = data - 1 / c * noise

    return {
        "sampled_data": sampled_data,
        "standard_errors": standard_errors * (1 + c**2) ** 0.5,
        "_validation_data": validation_data,
        "_validation_standard_errors": standard_errors * (1 + (1 / c) ** 2) ** 0.5,
    }


def fit_npmle_by_bins(
    estimates, standard_errors, covariate_function, conditional_mean, conditional_std, nbins=20
):
    # bin standard errors into bins according to quantile
    bin_type = np.less_equal.outer(
        standard_errors,
        np.quantile(standard_errors, np.linspace(0, 1, nbins + 1)[1:]),
    ).sum(1)

    zs = (estimates - conditional_mean) / conditional_std
    nus = standard_errors / conditional_std
    priors = {}
    start, end = bin_type.min(), bin_type.max()

    for t in range(start, end + 1):
        idx = bin_type == t
        est = zs[idx].copy()
        se = nus[idx].copy()

        _, meta = independent_npmle(est, se, v=default_v(est.min(), est.max()))
        locs, masses = (
            meta["estimated_prior_location"],
            meta["estimated_prior_mass"],
        )
        normalized_locs, masses = normalize_prior(locs, masses)
        priors[t] = (normalized_locs, masses)
    return bin_type, priors


def compute_merged_df_and_proxy_weight(proxy, split, est_var, alt_est_var, **kwargs):
    assert "black_pooled" in est_var

    df = load_data_for_outcome(est_var)
    alt_df = load_data_for_outcome(alt_est_var)
    merged_df = df.merge(
        alt_df[[alt_est_var, alt_est_var + "_se"]], on=identifiers, how="left"
    ).sort_index()
    assert len(merged_df) == len(df)

    black_weight = (
        (merged_df["kid_black_pooled_blw_p50_n"] / merged_df["kid_pooled_pooled_blw_p50_n"])
        .clip(0, 1)
        .values
    )

    return merged_df, black_weight


def coupled_bootstrap_proxy_prior_generator(
    df, est_var, alt_est_var, proxy, black_weight, seed=1231451, split=0.9
):
    rng = np.random.RandomState(seed)
    c = np.sqrt((1 - split) / split)

    sample_inflation = (1 + c**2) ** 0.5
    validation_inflation = (1 + (1 / c) ** 2) ** 0.5

    noise_target = df[est_var + "_se"] * rng.randn(len(df))
    noise_proxy = df[alt_est_var + "_se"] * rng.randn(len(df))

    validation_data = (df[est_var] - 1 / c * noise_target).values
    validation_standard_errors = (df[est_var + "_se"] * validation_inflation).values

    noised_target = (df[est_var] + c * noise_target).values
    noised_proxy = (df[alt_est_var] + c * noise_proxy).values

    if proxy == "white":
        sampled_data = noised_proxy
        sampled_standard_errors = df[alt_est_var + "_se"] * sample_inflation
    else:
        assert proxy == "pooled", "proxy must be either 'white' or 'pooled'"
        sampled_data = noised_target * black_weight + noised_proxy * (1 - black_weight)
        sampled_standard_errors = (
            black_weight**2 * df[est_var + "_se"] ** 2
            + (1 - black_weight) ** 2 * df[alt_est_var + "_se"] ** 2
        ).values ** 0.5 * sample_inflation

    return {
        "sampled_data": sampled_data,
        "standard_errors": sampled_standard_errors,
        "_validation_data": np.where(np.isnan(sampled_data), np.nan, validation_data),
        "_validation_standard_errors": np.where(
            np.isnan(sampled_data), np.nan, validation_standard_errors
        ),
    }
