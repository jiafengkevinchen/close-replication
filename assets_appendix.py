"""Builds figures and tables for the appendix of the paper."""

import pandas as pd
import numpy as np
from build_data import est_vars as EST_VARS
from postprocessing.visualize_utils import explanation
from postprocessing.visualize_utils import CORAL, RUBY, ASHER, ACCENT
from simulator.simulator import make_simulator

import matplotlib.pyplot as plt
import seaborn as sns
from build_data import load_data_for_outcome, covariates
from build_data import est_vars as EST_VARS
from residualize import residualize

from conditional_means.kernel import local_linear_regression_conditional_moments
import matplotlib.gridspec as gridspec
from binsreg import binsreg

sns.set_style("white")
sns.set_context("paper")
sns.set_color_codes()


# Figure OA5.1: Negative variances
kfrs = [
    "kfr_pooled_pooled_p25",
    "kfr_black_pooled_p25",
    "kfr_white_pooled_p25",
    "kfr_white_male_p25",
    "kfr_black_male_p25",
]

kfr_top20s = [
    "kfr_top20_pooled_pooled_p25",
    "kfr_top20_black_pooled_p25",
    "kfr_top20_white_pooled_p25",
    "kfr_top20_black_male_p25",
    "kfr_top20_white_male_p25",
]

jails = [
    "jail_pooled_pooled_p25",
    "jail_black_pooled_p25",
    "jail_white_pooled_p25",
    "jail_black_male_p25",
    "jail_white_male_p25",
]


def binsreg_data(y, x, nbins=20):
    v = binsreg(y, x, nbins=nbins, cb=(0, 0), level=95, noplot=True)
    return pd.concat(
        [v.data_plot[0].dots, v.data_plot[0].cb.groupby("bin")[["cb_l", "cb_r"]].nth(0)], axis=1
    )


def generate_successive_diff(estimates, standard_errors, skip=False):
    xs = np.log10(standard_errors)
    idx = xs.argsort()
    sorted_y = estimates[idx]
    sorted_var = standard_errors[idx] ** 2
    diff_y = sorted_y[1:] - sorted_y[:-1]
    pre_smooth_cond_var = (diff_y**2 - (sorted_var[:-1] + sorted_var[1:])) / 2
    if skip:
        pre_smooth_cond_var[::2], xs[idx][::2], xs[idx][:-1][::2], xs
    return pre_smooth_cond_var, xs[idx], xs[idx][:-1], xs


val_sets = [kfrs, kfr_top20s, jails]

plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)
a0 = plt.subplot(gs[0, :2])
a1 = plt.subplot(gs[0, 2:])
a2 = plt.subplot(gs[1, 1:3])

# f, axs = plt.subplots(nrows=3, sharex=False, sharey=False, figsize=(5, 8))
axs = [a0, a1, a2]
for i, vs in enumerate(val_sets):
    for val in vs[:3]:
        df = load_data_for_outcome(val)
        estimates = df[val].values
        standard_errors = df[val + "_se"].values

        pre_smooth_cond_var, sorted_x, sorted_x_less_1, xs = generate_successive_diff(
            estimates, standard_errors, skip=True
        )
        br_plot_data = binsreg_data(pre_smooth_cond_var, sorted_x_less_1, nbins=10)

        axs[i].errorbar(
            x=br_plot_data["x"],
            y=br_plot_data["fit"],
            yerr=(br_plot_data[["cb_l", "cb_r"]] - br_plot_data["fit"].values[:, None]).abs().T,
            ls="",
            marker="o",
            capthick=2,
            capsize=2,
            elinewidth=2,
            label=f"{explanation[val]}",
        )
        sns.despine()
    axs[i].axhline(0, ls="--", color="black")
    axs[i].legend(frameon=False)
axs[2].set_xlabel("$\\log_{10}($Standard error $\\sigma_i)$")
plt.savefig("assets/variance_right_tail.pdf", bbox_inches="tight")


## Table OA5.1: Sample sizes
sample_sizes = pd.Series({est_var: len(load_data_for_outcome(est_var)) for est_var in EST_VARS})
### The CZs are
# 'Phoenix, San Francisco, Los Angeles, Bridgeport, Washington DC, Miami, Tampa, Atlanta, Chicago, Boston, Detroit, Minneapolis, Philadelphia, Newark, New York, Cleveland, Pittsburgh, Houston, Dallas, Seattle'

with open("assets/sample_sizes.tex", "w") as f:
    f.write(
        sample_sizes.rename(index=explanation).rename("Sample size").to_latex().replace("|", "$|$")
    )

## Figure OA5.2: Draw of simulated data

# Consider a single example for the simulation
est_var = "kfr_top20_black_pooled_p25"  # for example
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

sample = simulator(94301)
sample_df = df.copy()
for k, v in sample.items():
    sample_df[k] = v
sample_df[est_var] = sample_df["sampled_data"]

plt.figure(figsize=(5, 3))
plt.scatter(
    y=df[est_var],
    x=np.log10(standard_errors),
    s=0.3,
    color=ACCENT,
    alpha=0.5,
    label="Real estimates",
    rasterized=True,
)

plt.scatter(
    y=sample_df[est_var],
    x=np.log10(standard_errors),
    s=0.3,
    color=CORAL,
    alpha=0.5,
    label="Simulated estimates",
    rasterized=True,
)

xlab = "$\\log_{10}($Standard error $\\sigma_i)$"
title = "Opportunity Atlas estimates for \n P(Income ranks in top 20 | Black, Parent at 25th Percentile)\nAll tracts in the largest 20 Commuting Zones"
plt.xlabel(xlab)
plt.suptitle(title, y=1.05)
sns.despine()
plt.legend(loc="upper left", frameon=False)
plt.savefig("assets/compare_real_sim_data.pdf", bbox_inches="tight")


## Figure OA5.3: Weibull exercise results
pretty_method_names = {
    "indep_gauss_nocov": "Indepndent Gaussian\n[no residualization]",
    "indep_npmle_nocov": "Indepndent NPMLE\n[no residualization]",
    "close_gauss_parametric_nocov": "CLOSE-Gauss (parametric)\n[no residualization]",
    "close_gauss_nocov": "CLOSE-Gauss\n[no residualization]",
    "close_npmle_nocov": "CLOSE-NPMLE\n[no residualization]",
    "indep_gauss": "Indepndent-Gauss",
    "indep_npmle": "Indepndent-NPMLE",
    "close_gauss_parametric": "CLOSE-Gauss (parametric)",
    "close_gauss": "CLOSE-Gauss",
    "close_npmle": "CLOSE-NPMLE",
}

mse_results = pd.read_csv("results/weibull/mse_scores.csv", index_col=0).T
pct_of_naive_to_oracle = (
    (
        (mse_results["naive"].values[:, None] - mse_results).drop(["naive", "oracle"], axis=1)
        / (mse_results["naive"] - mse_results["oracle"]).values[:, None]
    )
    .loc[list(explanation.keys())][
        [
            "indep_gauss_nocov",
            "close_gauss_parametric_nocov",
            "close_gauss_nocov",
            "close_npmle_nocov",
            "indep_gauss",
            "close_gauss_parametric",
            "close_gauss",
            "close_npmle",
        ]
    ]
    .rename(pretty_method_names, axis=1)
    .T.assign(**{"Column median": lambda x: x.median(axis=1)})
    .T.rename(explanation, axis=0)
    .dropna(axis=0)
)
fgsize = (8, 2)

oracle_mse_plot_kwargs = dict(vmin=20, vmax=100, cbar=False, linewidth=0.01)

plt.figure(figsize=fgsize)
sns.heatmap(
    pct_of_naive_to_oracle * 100, annot=True, fmt=".0f", cmap="PiYG", **oracle_mse_plot_kwargs
)
plt.xticks(rotation=60, ha="right")
plt.axvline(4, color="k", lw=5)
plt.xticks()
plt.title("MSE performance measured by the % of Naive-to-Oracle MSE captured", weight="bold")
plt.savefig("assets/mse_table_weibull.pdf", bbox_inches="tight")


## Figure OA5.4: Flexible covariates
ranks = pd.DataFrame(
    {
        est_var: pd.read_csv(f"results/covariate_additive_model/{est_var}.csv", index_col=0)[
            "mean_top_third"
        ]
        for est_var in EST_VARS
        if "male" not in est_var and "white" not in est_var
    }
).T.loc[
    [
        "kfr_pooled_pooled_p25",
        "kfr_black_pooled_p25",
        "kfr_top20_pooled_pooled_p25",
        "kfr_top20_black_pooled_p25",
        "jail_pooled_pooled_p25",
        "jail_black_pooled_p25",
    ]
]

proportion_good_obs = {
    est_var: pd.read_csv(f"results/covariate_additive_model/{est_var}.csv", index_col=0)[
        "good_obs"
    ][0]
    for est_var in EST_VARS
    if "male" not in est_var and "white" not in est_var
}

method_names = {
    "indep_gauss_flexible": ("Independent Gaussian (flexible)", ASHER),
    "close_gauss_res": ("CLOSE-Gauss (x residualized)", RUBY),
    "close_npmle_res": ("CLOSE-NPMLE (x residualized)", CORAL),
    "close_gauss": ("CLOSE-Gauss (flexible)", RUBY),
    "close_npmle": ("CLOSE-NPMLE (flexible)", CORAL),
}

plt.figure(figsize=(5, 2))
for i, method in enumerate(method_names.keys()):
    method_name = method.replace("_nocov", "")
    nocov = "res" in method

    name, color = method_names[method_name]
    offset = (1 if nocov else -1) * 0.15
    plt.scatter(
        x=(ranks[method] - ranks["naive"]) * 100,
        y=np.arange(len(ranks))[::-1] + offset + np.random.uniform(-0.1, 0.1),
        marker="o" if not nocov else "x",
        color=color,
        label=name,
    )

# Annotate with the proportion of good observations
for i, est_var in enumerate(ranks.index[::-1]):
    plt.text(
        x=-1,
        y=i,
        s=f"{proportion_good_obs[est_var]*100:.0f}%",
        ha="left",
        va="center",
        color=CORAL,
        weight="bold",
    )

plt.yticks(np.arange(len(ranks))[::-1], ranks.rename(index=explanation).index)
for y in np.arange(len(ranks))[::-1]:
    # plt.axhline(y=y - offset, color="grey", linewidth=0.5, ls="--")
    plt.axhline(y=y, color="grey", linewidth=0.5, ls="--")
plt.axvline(0, color="r", alpha=0.3)
plt.xlim((-3, 4))
plt.xlabel(
    "Performance difference relative to screening on raw estimates (percentile rank or percentage point)"
)
sns.despine()
plt.legend(loc=(1.05, 0), frameon=False)

plt.savefig("assets/rank_table_covariate_additive.pdf", bbox_inches="tight")
