import click
import os
import pandas as pd
import numpy as np
from build_data import est_vars as EST_VARS
from postprocessing.score import mean_squared_error, rank_score_within_cz
from tqdm.auto import tqdm
import warnings
from simulator.simulator import make_simulator
from build_data import load_data_for_outcome, covariates
from residualize import residualize
from conditional_means.kernel import local_linear_regression_conditional_moments


def mse_and_rank_scores(simulator_name, nsim=1000):
    is_coupled_bootstrap = simulator_name.startswith(
        "coupled_bootstrap"
    ) or simulator_name.startswith("proxy")
    ranks_table = {}
    mse_table = {}
    for est_var in EST_VARS:
        ranks = {}
        mses = {}

        for seed in tqdm(range(94301, 94301 + nsim), desc=f"{est_var}"):
            fname = f"data/simulated_posterior_means/{simulator_name}/{est_var}/{seed}.feather"
            if not os.path.exists(fname):

                # Silence warnings for expectedly missing files in Weibull
                if simulator_name == "weibull" and (
                    seed >= 94301 + 100
                    or not os.path.exists(
                        f"data/simulated_posterior_means/{simulator_name}/{est_var}/"
                    )
                ):
                    continue

                warnings.warn(f"File {fname} does not exist")
                continue
            pm_seed = pd.read_feather(fname)
            if is_coupled_bootstrap:
                ranks[seed] = rank_score_within_cz(pm_seed).loc[0.33]
            mses[seed] = mean_squared_error(pm_seed)
        if is_coupled_bootstrap:
            ranks_table[est_var] = pd.DataFrame(ranks).mean(axis=1)
        mse_table[est_var] = pd.DataFrame(mses).mean(axis=1)

    out_dir = f"results/{simulator_name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if is_coupled_bootstrap:
        pd.DataFrame(ranks_table).to_csv(f"results/{simulator_name}/rank_scores.csv")
    pd.DataFrame(mse_table).to_csv(f"results/{simulator_name}/mse_scores.csv")


@click.command()
@click.option("--simulator-name", type=str, default="coupled_bootstrap-0.9")
@click.option("--nsim", type=int, default=1000)
def main(simulator_name, nsim):
    mse_and_rank_scores(simulator_name, nsim)


if __name__ == "__main__":
    main()
