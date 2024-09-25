import pandas as pd
import numpy as np
import os
import json
import multiprocessing
from scipy.special import softmax
import datetime
from tqdm.auto import tqdm
import click

from build_data import load_data_for_outcome, covariates
from build_data import est_vars as EST_VARS

from simulator.simulator import make_simulator
from conditional_means.kernel import local_linear_regression_conditional_moments
from conditional_means.parametric import parametric_conditional_moments

from empirical_bayes.ebmethods import close_npmle
from empirical_bayes.ebmethods import independent_npmle as indep_npmle
from empirical_bayes.ebmethods import close_gaussian as close_gauss
from empirical_bayes.ebmethods import independent_gaussian as indep_gauss

from residualize import residualize


def calibrated_simulation(
    est_var,
    simulator_name,
    methods,
    nsim,
    data_dir="data/simulated_posterior_means",
    starting_seed=94301,
):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = f"{data_dir}/{simulator_name}/{est_var}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = load_data_for_outcome(est_var)
    covariate_fn, estimates = residualize(df, est_var, covariates, within_cz=False)
    standard_errors = df[est_var + "_se"].values

    conditional_mean_sim, conditional_std_sim = local_linear_regression_conditional_moments(
        estimates, standard_errors
    )
    simulator = make_simulator(
        simulator_name,
        est_var,
        estimates,
        standard_errors,
        covariate_fn,
        conditional_mean_sim,
        conditional_std_sim,
    )

    is_coupled_bootstrap = simulator_name.startswith(
        "coupled_bootstrap"
    ) or simulator_name.startswith("proxy")

    meta = {
        "est_var": est_var,
        "simulator_name": simulator_name,
        "methods": methods,
        "nsim": nsim,
        "date": today,
        "is_coupled_bootstrap": is_coupled_bootstrap,
    }

    oracle = make_oracle(simulator, covariate_fn) if not is_coupled_bootstrap else None

    for seed in tqdm(range(starting_seed, starting_seed + nsim), desc=f"{est_var}", total=nsim):

        if os.path.exists(out_dir + f"/{seed}.feather"):
            continue

        sample = simulator(seed)

        # Make the sample dict into a df by joining it w/ covariates and standard errors
        sample_df = df.copy()
        for k, v in sample.items():
            sample_df[k] = v
        sample_df[est_var] = sample_df["sampled_data"]

        # Compute the posterior means for each method
        posterior_mean_df = get_posterior_means(
            sample_df, est_var, methods, True, is_coupled_bootstrap, oracle
        )
        posterior_mean_df_no_cov = get_posterior_means(
            sample_df, est_var, methods, False, is_coupled_bootstrap, None
        )

        # concatenate the columns without duplicates
        for c in posterior_mean_df_no_cov.columns:
            if c not in posterior_mean_df.columns:
                posterior_mean_df[c] = posterior_mean_df_no_cov[c]

        posterior_mean_df.to_feather(out_dir + f"/{seed}.feather")

    with open(out_dir + "/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def get_posterior_means(sample, est_var, methods, partial_out, is_coupled_bootstrap, oracle):
    if is_coupled_bootstrap:
        # break sample into commuting zones and apply
        # `compute_posterior_means_subsample` on each commuting zone
        posterior_mean_df = []
        for cz in sample["czname"].unique():
            idx = sample["czname"] == cz
            subsample = sample.loc[idx]
            posterior_mean_df_ = compute_posterior_means_subsample(
                subsample, est_var, is_coupled_bootstrap, methods, partial_out, oracle
            )
            posterior_mean_df_["czname"] = cz
            posterior_mean_df.append(posterior_mean_df_)
        posterior_mean_df = pd.concat(posterior_mean_df, axis=0).reset_index(drop=True)

    else:
        posterior_mean_df = compute_posterior_means_subsample(
            sample, est_var, is_coupled_bootstrap, methods, partial_out, oracle
        )

    return posterior_mean_df


# how to estimate conditional moment, method to plug in
method_features = {
    "indep_npmle": ("const", indep_npmle),
    "indep_gauss": ("const", indep_gauss),
    "close_npmle": ("np", close_npmle),
    "close_gauss": ("np", close_gauss),
    "close_gauss_parametric": ("parametric", close_gauss),
    "close_npmle_norm": ("np", lambda *args: close_npmle(*args, normalize_prior=True)),
}


def make_oracle(simulator, covariate_fn, npriorsim=2000, starting_seed=131385713):
    """
    Draw from the true DGP to create a set of prior samples, then use those to
    compute the oracle posterior means
    """
    prior_samples = np.array(
        [
            simulator(seed=seed)["_sampled_residualized_parameter"] + covariate_fn
            for seed in range(starting_seed, starting_seed + npriorsim)
        ]
    )  # n_prior_samples x n_data

    def oracle(estimates_data, standard_errors):
        log_posterior = -((prior_samples - estimates_data[None, :]) ** 2) / (
            2 * standard_errors[None, :] ** 2
        )
        posterior_mass = softmax(log_posterior, axis=0)
        posterior_means = (prior_samples * posterior_mass).sum(0)
        return posterior_means

    return oracle


def compute_posterior_means_subsample(
    subsample, est_var, is_coupled_bootstrap, methods, partial_out, oracle
):
    """On a (sub)sample, compute the posterior means for each method. For coupled
    bootstrap, a subsample is a CZ. For the Monte Carlo exercise, a subsample is just the
    whole sample."""

    if partial_out:
        covariate_fn, estimates = residualize(
            subsample, est_var, covariates, within_cz=is_coupled_bootstrap
        )
    else:
        estimates = subsample[est_var].values
        covariate_fn = np.zeros_like(estimates)
    standard_errors = subsample[est_var + "_se"].values

    estimated_conditional_moments = {
        "np": local_linear_regression_conditional_moments(
            estimates, standard_errors, fast=(not is_coupled_bootstrap)
        ),
        "parametric": parametric_conditional_moments(estimates, standard_errors),
    }

    posterior_mean_df = {"naive": estimates + covariate_fn}
    for met in methods:
        cm_name, eb_method = method_features[met]
        if cm_name != "const":
            conditional_mean, conditional_std = estimated_conditional_moments[cm_name]
            posterior_means, _ = eb_method(
                estimates, standard_errors, conditional_mean, conditional_std
            )
        else:
            posterior_means, _ = eb_method(estimates, standard_errors)
        met_name = met if partial_out else f"{met}_nocov"
        posterior_mean_df[met_name] = posterior_means + covariate_fn

    if not is_coupled_bootstrap and (oracle is not None):
        posterior_mean_df["oracle"] = oracle(estimates + covariate_fn, standard_errors)
        posterior_mean_df["truth"] = subsample["_sampled_residualized_parameter"] + covariate_fn
    if is_coupled_bootstrap:
        posterior_mean_df["validation"] = subsample["_validation_data"]
        posterior_mean_df["validation_se"] = subsample["_validation_standard_errors"]

    return pd.DataFrame(posterior_mean_df)


@click.command()
@click.option("--est_var", type=str, required=True)
@click.option("--simulator-name", type=str)
@click.option("--methods", default="all", type=str)
@click.option("--nsim", default=1000, type=int)
@click.option("--data-dir", default="data/simulated_posterior_means", type=str)
@click.option("--starting-seed", default=94301, type=int)
def main(est_var, simulator_name, methods, nsim, data_dir, starting_seed):
    methods_list = list(method_features.keys())
    methods_list.remove("close_npmle_norm")
    methods = methods.split(",") if methods != "all" else methods_list
    calibrated_simulation(
        est_var, simulator_name, methods, nsim, data_dir=data_dir, starting_seed=starting_seed
    )


if __name__ == "__main__":
    main()
