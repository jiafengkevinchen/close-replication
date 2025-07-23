# Check some Monte Carlo outputs

from empirical_exercise import _main as simulation_main
import pandas as pd
import numpy as np
import rpy2.rinterface_lib.callbacks
import click

# Suppress R warnings in Python
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None


def check(est_var, seed, simulator_name):
    print("Checking Monte Carlo outputs...")
    simulation_main(
        est_var=est_var,
        methods="all",
        nsim=1,
        starting_seed=seed,
        data_dir="data/simulated_posterior_means_sample",
        simulator_name=simulator_name,
    )

    orig = pd.read_feather(
        f"data/simulated_posterior_means/{simulator_name}/{est_var}/{seed}.feather"
    )
    new = pd.read_feather(
        f"data/simulated_posterior_means_sample/{simulator_name}/{est_var}/{seed}.feather"
    )

    if "czname" in orig.columns:
        orig = orig.drop(columns=["czname"])
    if "czname" in new.columns:
        new = new.drop(columns=["czname"])

    print("-" * 50)
    print("Seed:", seed)
    print("Outcome variable:", est_var)
    print("Simulator name:", simulator_name)
    print(
        "Correlation between original and new Monte Carlo samples\n(some differences may exist due to hardware):"
    )
    print("-" * 50)
    corrs = orig.corrwith(new)
    intercepts = {}
    regression_coefs = {}
    for col in orig.columns:
        orig_col = orig[col]
        new_col = new[col]
        regression_coef, intercept = np.polyfit(orig_col, new_col, 1)
        intercepts[col] = intercept
        regression_coefs[col] = regression_coef

    print(
        pd.DataFrame(
            {"Correlation": corrs, "Intercept": intercepts, "Regression Coef": regression_coefs}
        )
    )

    print("-" * 50)


@click.command()
@click.option("--est_var", type=str, default="kfr_black_pooled_p25")
@click.option("--seed_number", type=int, default=94999)
@click.option("--simulator_name", type=str, default="npmle_by_bins")
def main(est_var, seed_number, simulator_name):
    seed = seed_number % 1000 + 94301 if simulator_name != "weibull" else seed_number % 100 + 94301
    check(est_var, seed, simulator_name)


if __name__ == "__main__":
    main()
