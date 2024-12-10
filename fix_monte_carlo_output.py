import pandas as pd
import numpy as np
import os
import json
import multiprocessing
from scipy.special import softmax
import datetime
from tqdm.auto import tqdm
import click

from build_data import est_vars as EST_VARS

from simulator.simulator import make_simulator
from conditional_means.kernel import local_linear_regression_conditional_moments
from conditional_means.parametric import parametric_conditional_moments

from empirical_bayes.ebmethods import close_npmle
from empirical_bayes.ebmethods import independent_npmle as indep_npmle
from empirical_bayes.ebmethods import close_gaussian as close_gauss
from empirical_bayes.ebmethods import independent_gaussian as indep_gauss
from build_data import load_data_for_outcome, covariates
from empirical_exercise import compute_posterior_means_subsample, make_oracle
from residualize import residualize


def _load_data(est_var):
    df = load_data_for_outcome(est_var)
    covariate_fn, estimates = residualize(df, est_var, covariates, within_cz=False)
    standard_errors = df[est_var + "_se"].values

    conditional_mean_sim, conditional_std_sim = local_linear_regression_conditional_moments(
        estimates, standard_errors
    )

    simulator = make_simulator(
        "npmle_by_bins",
        est_var,
        estimates,
        standard_errors,
        covariate_fn,
        conditional_mean_sim,
        conditional_std_sim,
    )

    return df, simulator, covariate_fn


for est_var in EST_VARS:
    df, simulator, true_cov_fn = _load_data(est_var)
    seed_to_check = np.random.RandomState(123145).randint(94301, 94301 + 1000)
    for seed in tqdm(range(94301, 94301 + 1000), desc=f"{est_var}"):
        fname = f"data/simulated_posterior_means/npmle_by_bins/{est_var}/{seed}.feather"
        pm_seed = pd.read_feather(fname)

        sim = simulator(seed)

        if seed == seed_to_check:
            # check this seed more thoroughly
            sample_df = df.copy()
            for k, v in sim.items():
                sample_df[k] = v
            sample_df[est_var] = sample_df["sampled_data"]

            oracle = make_oracle(simulator, true_cov_fn)
            posterior_mean_df = compute_posterior_means_subsample(
                sample_df,
                est_var,
                False,
                "indep_npmle indep_gauss close_npmle close_gauss close_gauss_parametric".split(),
                True,
                oracle,
            )
            tqdm.write(f"seed: {seed} | est_var: {est_var}")
            for c in posterior_mean_df.columns:
                if c in pm_seed:
                    tqdm.write(f"{c}: {(posterior_mean_df[c] - pm_seed[c]).abs().max()}")

        pm_seed["truth"] = sim["_sampled_residualized_parameter"] + true_cov_fn
        pm_seed["true_covariate_fn"] = true_cov_fn
        pm_seed["truth_residualized"] = sim["_sampled_residualized_parameter"]

        if seed == seed_to_check:
            tqdm.write(f"{pm_seed.columns}")
            tqdm.write(f"{posterior_mean_df.columns}")

            for c in posterior_mean_df.columns:
                if c in pm_seed:
                    tqdm.write(f"{c}: {(posterior_mean_df[c] - pm_seed[c]).abs().max()}")

        pm_seed.to_feather(f"data/simulated_posterior_means/npmle_by_bins/{est_var}/{seed}.feather")
