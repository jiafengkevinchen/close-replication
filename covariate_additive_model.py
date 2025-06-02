import pandas as pd
from tqdm.auto import tqdm
from build_data import load_data_for_outcome, covariates
import numpy as np
from sklearn.preprocessing import SplineTransformer
from numpy.linalg import lstsq
from empirical_bayes.ebmethods import close_npmle, close_gaussian, independent_gaussian
from residualize import residualize
from conditional_means.kernel import local_linear_regression_conditional_moments
import click
from matplotlib import pyplot as plt


def load_and_reorder_data(est_var):
    """Load data and reorder by czname to match how results were saved"""
    sample = load_data_for_outcome(est_var)
    df = []
    for cz in sample["czname"].unique():
        idx = sample["czname"] == cz
        df.append(sample.loc[idx])
    df = pd.concat(df, axis=0).reset_index(drop=True)
    return df


est_var = "kfr_top20_black_pooled"
seed = 94301
SPLIT = 0.9
c = np.sqrt((1 - SPLIT) / SPLIT)


def compute_posterior_means_additive(est_var, seed):
    path = f"data/simulated_posterior_means/coupled_bootstrap-0.9/{est_var}/{seed}.feather"
    results = pd.read_feather(path)
    df = load_and_reorder_data(est_var)

    estimates = results["naive"].values
    standard_errors = df[f"{est_var}_se"].values * (1 + c**2) ** 0.5
    validation = results["validation"].values
    validation_se = results["validation_se"].values

    cts_covariates = [
        "par_rank_pooled_pooled_mean",
        "par_rank_black_pooled_mean",
        "poor_share2010",
        "share_black2010",
        "hhinc_mean2000",
        "ln_wage_growth_hs_grad",
        "frac_coll_plus2010",
        "log_kid_black_pooled_blw_p50_n",
        "log_kid_pooled_pooled_blw_p50_n",
    ]
    sample = df[
        cts_covariates + ["czname"] + ["kid_black_pooled_blw_p50_n", "kid_pooled_pooled_blw_p50_n"]
    ].copy()
    sample["estimates"] = estimates
    sample[est_var] = estimates
    sample["log_standard_errors"] = np.log10(standard_errors)
    sample["standard_errors"] = standard_errors
    sample[f"{est_var}_se"] = standard_errors

    spline = SplineTransformer(degree=3, n_knots=3, knots="quantile", include_bias=False)
    feature_matrix = spline.fit_transform(sample[cts_covariates + ["log_standard_errors"]])
    feature_matrix = np.c_[np.ones(len(feature_matrix)), feature_matrix]
    beta, _, _, _ = lstsq(feature_matrix, sample["estimates"], rcond=1e-6)

    # Fitting conditional means and variance using additive model with spline and least-squares
    conditional_mean = feature_matrix @ beta
    resid_sq_minus_s2 = (estimates - conditional_mean) ** 2 - standard_errors**2
    gamma, _, _, _ = lstsq(feature_matrix, resid_sq_minus_s2, rcond=1e-6)

    # Ensure the variance is positive by truncating at zero
    conditional_std = np.clip(feature_matrix @ gamma, a_min=0, a_max=None) ** 0.5

    # Compute posterior means for close_npmle and close_gaussian
    close_npmle_pm, meta_npmle = close_npmle(
        estimates, standard_errors, conditional_mean, conditional_std
    )
    close_gauss_pm, _ = close_gaussian(
        estimates, standard_errors, conditional_mean, conditional_std
    )

    feature_matrix_no_sigma = spline.fit_transform(sample[cts_covariates])
    feature_matrix_no_sigma = np.c_[np.ones(len(feature_matrix_no_sigma)), feature_matrix_no_sigma]
    beta_no_sigma, _, _, _ = lstsq(feature_matrix_no_sigma, sample["estimates"], rcond=1e-6)
    flexible_covariate_fn = feature_matrix_no_sigma @ beta_no_sigma

    independent_gaussian_pm, _ = independent_gaussian(
        estimates - flexible_covariate_fn, standard_errors
    )

    # Residualize estimates first, then compute posterior means, add back
    covariate_fn, residualized_est = residualize(
        sample,
        est_var,
        cts_covariates + ["kid_black_pooled_blw_p50_n", "kid_pooled_pooled_blw_p50_n"],
        weighted=False,
        within_cz=False,
    )

    conditional_mean_res, conditional_std_res = local_linear_regression_conditional_moments(
        residualized_est, standard_errors, fast=True
    )

    close_npmle_pm_res, _ = close_npmle(
        residualized_est, standard_errors, conditional_mean_res, conditional_std_res
    )
    close_gauss_pm_res, _ = close_gaussian(
        residualized_est, standard_errors, conditional_mean_res, conditional_std_res
    )

    # Collect the posterior means into a dataframe
    posterior_means = pd.DataFrame(
        {
            "close_npmle": close_npmle_pm,
            "close_gauss": close_gauss_pm,
            "close_npmle_res": close_npmle_pm_res + covariate_fn,
            "close_gauss_res": close_gauss_pm_res + covariate_fn,
            "indep_gauss_flexible": independent_gaussian_pm + flexible_covariate_fn,
            "naive": estimates,
            "validation": validation,
            "validation_se": validation_se,
        }
    )

    metadata = {"good_obs": meta_npmle["good_obs"].mean()}

    rank_table = rank_score(posterior_means)
    mse_table = (
        (
            posterior_means[
                ["close_npmle", "close_gauss", "close_npmle_res", "close_gauss_res", "naive"]
            ]
            - posterior_means["validation"].values[:, None]
        )
        ** 2
    ).mean()

    return posterior_means, rank_table, mse_table, metadata


def rank_score(posterior_means):
    """
    For each method, compute the mean of the validation variable within the top x%
    """
    assert "validation" in posterior_means.columns
    true_col = "validation"
    methods = posterior_means.drop(columns=[true_col, "validation_se"], axis=1).columns
    output = []
    for method_name in methods:
        # Sort by the posterior mean, largest first
        sorted_pm = posterior_means.sort_values(method_name, ascending=False)

        # cum_prop is [0, 1/n, 2/n, ..., (n-1)/n] where n is the number of observations
        cum_prop = np.arange(len(sorted_pm)) / len(sorted_pm)

        pcts = np.linspace(0.01, 1, 100)  # 1% to 100%

        # selectors is a boolean matrix where each row is a tract and each column is a
        # percentile. The value is True if the tract is in the top x%
        selectors = cum_prop[:, None] <= pcts[None, :]

        # take mean of the true value for each top percentile
        utils = (selectors * sorted_pm[true_col].values[:, None]).sum(0) / selectors.sum(0)
        output.append(pd.Series(utils, index=pcts, name=method_name))
    return pd.DataFrame(output).T


@click.command()
@click.option("--est_var", type=str, required=True)
@click.option("--starting-seed", default=94301, type=int)
@click.option("--nsim", default=100, type=int)
def main(est_var, starting_seed, nsim):

    results = 0
    for seed in tqdm(range(94301, 94301 + nsim), desc=f"{est_var}", total=nsim):
        posterior_means, rank_table, mse_table, metadata = compute_posterior_means_additive(
            est_var, seed
        )
        result = pd.DataFrame({"mean_top_third": rank_table.loc[0.33], "mse": mse_table})
        result["good_obs"] = metadata["good_obs"]
        results += result

    results = results / nsim
    results.reset_index().to_csv(f"results/covariate_additive_model/{est_var}.csv", index=False)


if __name__ == "__main__":
    main()
