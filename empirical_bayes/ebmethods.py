import warnings

import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from scipy.special import softmax

numpy2ri.activate()
rebayes = importr("REBayes")


def weighted_mean(x, weights):
    return (x * weights).sum() / weights.sum()


def normalize_prior(locs, masses):
    """Given a discrete distribution, normalize it to have mean zero and variance 1"""
    masses = masses.clip(min=0)
    masses = masses / masses.sum()

    mean = locs @ masses
    variance = (locs - mean) ** 2 @ masses

    normalized_locs = (locs - mean) / np.sqrt(variance)
    return normalized_locs, masses


def glmix(estimates, standard_errors, v=500, **glmix_kwargs):
    """Wrapper for REBayes::GLmix"""
    mixture_fit = rebayes.GLmix(estimates, sigma=standard_errors, v=v, **glmix_kwargs)
    prior_location = mixture_fit.rx["x"][0]
    prior_mass = mixture_fit.rx["y"][0]

    posterior_means = mixture_fit.rx["dy"][0]

    status = mixture_fit.rx["status"][0][0]
    if status != "OPTIMAL":
        warnings.warn(f"Status is {status}")

    return (
        posterior_means,
        {
            "estimated_prior_location": prior_location,
            "estimated_prior_mass": prior_mass,
        },
        status,
    )


def independent_gaussian(estimates, standard_errors):
    precision_weights = 1 / (standard_errors**2)
    grand_mean = weighted_mean(estimates, precision_weights)
    grand_var = weighted_mean((estimates - grand_mean) ** 2 - standard_errors**2, precision_weights)

    shrinkage_factor = grand_var / (standard_errors**2 + grand_var)
    signal_proportion = shrinkage_factor.mean()

    posterior_means = grand_mean + shrinkage_factor * (estimates - grand_mean)
    return posterior_means, {
        "shrinkage_factor": shrinkage_factor,
        "signal_proportion": signal_proportion,
        "estimated_grand_mean": grand_mean,
        "estimated_grand_var": grand_var,
    }


def independent_npmle(estimates, standard_errors, **glmix_kwargs):
    _posterior_means, meta, status = glmix(estimates, standard_errors, **glmix_kwargs)
    if status != "OPTIMAL":
        warnings.warn(
            f"independent_npmle: Initial solve failed, status is {status}. Attempt with v = 1000."
        )
        glmix_kwargs["v"] = 1000
        _posterior_means, meta, status = glmix(
            estimates,
            standard_errors,
            **glmix_kwargs,
        )
        if status != "OPTIMAL":
            warnings.warn(f"independent_npmle: Second solve failed, status is {status}.")
        else:
            warnings.warn(
                "independent_npmle: Second solve succeeded, but quality may not be optimal."
            )
    return _posterior_means, meta


def default_v(t_stat_min, t_stat_max):
    """Default grid for CLOSE-NPMLE"""
    return np.r_[
        np.linspace(min(t_stat_min, -6.1), -6.1, 50),
        np.linspace(-6, 6, 400),
        np.linspace(6.1, max(t_stat_max, 8), 50),
    ]


def posterior_mean(prior_locs, prior_masses, estimates, standard_errors):
    """Given a discrete prior for a normal location, compute posterior means"""
    log_posterior = (
        -0.5 * (prior_locs[:, None] - estimates[None, :]) ** 2 / (standard_errors[None, :] ** 2)
        + np.log(prior_masses)[:, None]
    )
    posterior_mass = softmax(log_posterior, axis=0)
    posterior_means = (prior_locs[:, None] * posterior_mass).sum(axis=0)
    return posterior_means


def close_npmle(
    estimates,
    standard_errors,
    conditional_mean,
    conditional_std,
    truncation=1e-7,
    norm_prior=False,
    **glmix_kwargs,
):
    transformed_estimates = (estimates - conditional_mean) / conditional_std.clip(min=truncation)
    good_obs = conditional_std > 0
    transformed_standard_errors = standard_errors / conditional_std.clip(min=truncation)

    if "v" not in glmix_kwargs:
        t_stat = (transformed_estimates / transformed_standard_errors)[good_obs]
        glmix_kwargs["v"] = default_v(t_stat.min(), t_stat.max())

    _posterior_means, meta, status = glmix(
        transformed_estimates[good_obs],
        transformed_standard_errors[good_obs],
        **glmix_kwargs,
    )

    # Handle optimizer failure
    if status != "OPTIMAL":
        warnings.warn(
            f"close_npmle: Initial solve failed, status is {status}. Attempt with v = 1000."
        )
        glmix_kwargs["v"] = 1000
        _posterior_means, meta, status = glmix(
            transformed_estimates[good_obs],
            transformed_standard_errors[good_obs],
            **glmix_kwargs,
        )
        if status != "OPTIMAL":
            warnings.warn(f"close_npmle: Second solve failed, status is {status}.")
        else:
            warnings.warn("close_npmle: Second solve succeeded, but quality may not be optimal.")

    if norm_prior:
        # Impose that the prior mean and variance are 0 and 1
        locs, masses = (
            meta["estimated_prior_location"],
            meta["estimated_prior_mass"],
        ) = normalize_prior(meta["estimated_prior_location"], meta["estimated_prior_mass"])

        # Recalculate posterior mean
        _posterior_means = posterior_mean(
            locs,
            masses,
            transformed_estimates[good_obs],
            transformed_standard_errors[good_obs],
        )

    posterior_means_transformed_parameter = np.zeros_like(estimates)
    posterior_means_transformed_parameter[good_obs] = _posterior_means
    posterior_means = conditional_mean + conditional_std * posterior_means_transformed_parameter

    meta["good_obs"] = good_obs
    meta["conditional_mean"] = conditional_mean
    meta["conditional_std"] = conditional_std
    meta["posterior_mean_transformed_parameter"] = posterior_means_transformed_parameter
    return posterior_means, meta


def close_gaussian(estimates, standard_errors, conditional_mean, conditional_std, truncate=1e-10):
    conditional_var = conditional_std**2
    shrink_factor = conditional_var / (conditional_var + standard_errors**2)
    posterior_means = conditional_mean + shrink_factor * (estimates - conditional_mean)
    posterior_means_transformed_parameter = np.where(
        conditional_std > 0,
        (posterior_means - conditional_mean) / conditional_std.clip(min=truncate),
        0,
    )

    return posterior_means, {
        "shrinkage_factor": shrink_factor,
        "signal_proportion": shrink_factor.mean(),
        "conditional_mean": conditional_mean,
        "conditional_std": conditional_std,
        "posterior_mean_transformed_parameter": posterior_means_transformed_parameter,
    }
