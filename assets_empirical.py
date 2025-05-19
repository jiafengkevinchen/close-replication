"""Builds figures for the empirical section of the paper."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from build_data import est_vars as EST_VARS
from postprocessing.visualize_utils import explanation
from postprocessing.visualize_utils import plot_league_table_value_basic_eb, CORAL
from build_data import load_data_for_outcome
from build_data import est_vars as EST_VARS


sns.set_style("white")
sns.set_context("paper")
sns.set_color_codes()


pretty_method_names = {
    "indep_gauss_nocov": "Independent-Gauss\n[no residualization]",
    "indep_npmle_nocov": "Independent-NPMLE\n[no residualization]",
    "close_gauss_parametric_nocov": "CLOSE-Gauss (parametric)\n[no residualization]",
    "close_gauss_nocov": "CLOSE-Gauss\n[no residualization]",
    "close_npmle_nocov": "CLOSE-NPMLE\n[no residualization]",
    "indep_gauss": "Independent-Gauss",
    "indep_npmle": "Independent-NPMLE",
    "close_gauss_parametric": "CLOSE-Gauss (parametric)",
    "close_gauss": "CLOSE-Gauss",
    "close_npmle": "CLOSE-NPMLE",
    "naive": "Naive",
}

## Figure 4: MSE performance
mse_results = pd.read_csv("results/npmle_by_bins/mse_scores.csv", index_col=0).T
col_order = [
    "indep_gauss_nocov",
    "indep_npmle_nocov",
    "close_gauss_parametric_nocov",
    "close_gauss_nocov",
    "close_npmle_nocov",
    "indep_gauss",
    "indep_npmle",
    "close_gauss_parametric",
    "close_gauss",
    "close_npmle",
]
pct_of_naive_to_oracle = (
    (
        (mse_results["naive"].values[:, None] - mse_results).drop(["naive", "oracle"], axis=1)
        / (mse_results["naive"] - mse_results["oracle"]).values[:, None]
    )
    .loc[list(explanation.keys())][col_order]
    .rename(pretty_method_names, axis=1)
    .T.assign(**{"Column median": lambda x: x.median(axis=1)})
    .T.rename(explanation, axis=0)
)
fgsize = (8, 8 / 1.618)
oracle_mse_plot_kwargs = dict(vmin=20, vmax=100, cbar=False, linewidth=0.01)
plt.figure(figsize=fgsize)
sns.heatmap(
    pct_of_naive_to_oracle * 100, annot=True, fmt=".0f", cmap="PiYG", **oracle_mse_plot_kwargs
)
plt.xticks(rotation=60, ha="right")
plt.axvline(5, color="k", lw=5)
plt.xticks()
plt.title("MSE performance measured by the % of Naive-to-Oracle MSE captured", weight="bold")
plt.savefig("assets/mse_table_calibrated.pdf", bbox_inches="tight")


## Figure not included in the paper (old version of Figure 5)
col_list_rank = [
    "indep_gauss",
    "close_gauss_parametric",
    "close_gauss",
    "close_npmle",
    "indep_gauss_nocov",
    "close_gauss_parametric_nocov",
    "close_gauss_nocov",
    "close_npmle_nocov",
]
ranks = pd.read_csv("results/coupled_bootstrap-0.9/rank_scores.csv", index_col=0).T.loc[
    list(explanation.keys())
]
plt.figure(figsize=(5, 4))
plot_league_table_value_basic_eb(ranks, methods=col_list_rank)
plt.title(
    "Selection performance measured by coupled bootstrap estimates of mean parameter among selected",
    weight="bold",
)
plt.savefig("assets/ranks.pdf", bbox_inches="tight")


## Voice over in the paper

# Footnote 30 in the paper
print("\n\n\nGap between close-npmle and indep-gauss")
print((ranks["close_npmle"] - ranks["indep_gauss"]) * 100)

# Median improvement calculation
median_improvement = (
    (ranks["close_npmle"] - ranks["indep_gauss"])
    / np.maximum(ranks["indep_gauss"] - ranks["naive"], 0.0001)
    * 100
).median()
print("\n\n\nMedian improvement")
print(median_improvement)

# Number of indep-gauss that underperforms naive
bad_count = (ranks["indep_gauss"] - ranks["naive"] < 0).sum()
print("\n\n\nBad indep-gauss count")
print(bad_count)


# Value of data
grand_means = pd.Series(
    {est_var: load_data_for_outcome(est_var)[est_var].mean() for est_var in EST_VARS}
)
value_data = (
    (ranks["close_npmle"] - ranks["indep_gauss"]) / (ranks["indep_gauss"] - grand_means) * 100
).loc[explanation.keys()]

print("\n\n\nValue of data")
print(value_data.median(), (value_data > 100).sum(), value_data)

# Raw estimated ranks
print("\n\n\nRaw estimated ranks")
print((ranks[["close_npmle", "indep_gauss"]] * 100))


## Figure 5

col_order = [
    "indep_gauss_nocov",
    "indep_npmle_nocov",
    "close_gauss_parametric_nocov",
    "close_gauss_nocov",
    "close_npmle_nocov",
    "naive",
    "indep_gauss",
    "indep_npmle",
    "close_gauss_parametric",
    "close_gauss",
    "close_npmle",
]
rank_diff = (ranks - ranks["naive"].values[:, None]) * 100
tab = rank_diff[col_order]
tab_n = tab / tab["close_npmle"].values[:, None]
fgsize = (8, 8 / 1.618)
plt.figure(figsize=fgsize)
sns.heatmap(
    tab_n.rename(explanation, axis=0).rename(pretty_method_names, axis=1),
    annot=ranks[tab_n.columns] * 100,
    fmt=".2f",
    cbar=False,
    linewidth=0.01,
    cmap="PiYG",
    vmin=-1.2,
    vmax=1.2,
)
plt.xticks(rotation=60, ha="right")
plt.axvline(5, color="k", lw=5)

# highlight largest entry in each row
for i in range(tab.shape[0]):
    plt.scatter(
        np.argmax(tab_n.values[i]) + 0.85,
        i + 0.5,
        s=10,
        marker="*",
        facecolors="none",
        edgecolors=CORAL,
        linewidth=2,
    )
plt.title(
    "Estimated average $\\vartheta$ among selected tracts",
    weight="bold",
)
plt.savefig("assets/rank_table_additional.pdf", bbox_inches="tight")
