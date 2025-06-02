"""Builds figures for the introductory example of the paper."""

import numpy as np
from postprocessing.visualize_utils import CORAL, RUBY, ASHER, ACCENT
import matplotlib.pyplot as plt
import seaborn as sns
from build_data import load_data_for_outcome

from empirical_bayes.ebmethods import close_npmle
from empirical_bayes.ebmethods import independent_npmle as indep_npmle
from empirical_bayes.ebmethods import independent_gaussian as indep_gauss
from conditional_means.kernel import local_linear_regression_conditional_moments, ucb_fast
import matplotlib.gridspec as gridspec

import warnings
import matplotlib

# Filter out MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


sns.set_style("white")
sns.set_context("paper")
sns.set_color_codes()


est_var = "kfr_top20_black_pooled_p25"
df = load_data_for_outcome(est_var)
covariate_fn = np.zeros_like(df[est_var].values)
estimates = df[est_var].values
standard_errors = df[est_var + "_se"].values


posterior_means, indep_gauss_meta = indep_gauss(estimates, standard_errors)
cm, cstd = local_linear_regression_conditional_moments(estimates, standard_errors, fast=True)
posterior_means_close, close_npmle_meta = close_npmle(estimates, standard_errors, cm, cstd)
posterior_means_npmle, npmle_meta = indep_npmle(estimates, standard_errors)
xs = np.log10(standard_errors)

signal_std = ((estimates).std() ** 2 - (standard_errors**2).mean()) ** 0.5
print("Signal SD (footnote 6); expected about 0.037", signal_std)

np_df, point, cov, max_t = ucb_fast(
    estimates, xs, frac=0.1, kernel="epa", bwselect="imse-dpi", truncate=0.99, ngrid=40
)


## Figure 1: Raw data w/ estimated conditional mean
# Raw estimates and conditional means
title = "Opportunity Atlas estimates for \n P(Income ranks in top 20 | Black, Parent at 25th Percentile)\nAll tracts in the largest 20 Commuting Zones"
xlab = "$\\log_{10}($Standard error $\\sigma_i)$"

scatterkws = dict(
    s=0.5,
    label="Estimates $Y_i \\mid \\theta_i, \\sigma_i \\sim N(\\theta_i, \\sigma_i^2)$",
    color=ACCENT,
    alpha=0.5,
)
cond_mean_kws = dict(color=CORAL, label="Estimated $E[\\theta \mid \\sigma] = E[Y \mid \\sigma]$")
title = "Opportunity Atlas estimates for \n P(Income ranks in top 20 | Black, Parent at 25th Percentile)\nAll tracts in the largest 20 Commuting Zones"
xlab = "$\\log_{10}($Standard error $\\sigma_i)$"

plt.figure(figsize=(6, 4))
plt.scatter(x=xs, y=estimates, rasterized=True, **scatterkws)

plt.plot(np_df["x"], point, **cond_mean_kws)
plt.fill_between(
    np_df["x"],
    point - max_t * np_df["se_rb"],
    point + max_t * np_df["se_rb"],
    alpha=0.3,
    color=CORAL,
    label="95% uniform confidence band for $E[\\theta \\mid \\sigma]$",
)

sns.despine()
plt.legend(loc=(0.05, 0.65), frameon=False)
plt.xlabel("$\\log_{10}(\\sigma_i)$")
plt.suptitle(title)
plt.ylabel("Estimates $Y_i$")
plt.savefig("assets/example_raw.pdf", bbox_inches="tight", dpi=500)


## Figure 2: Posterior means
# The posterior means from a battery of EB methods
idx = xs.argsort()
scatterkws = dict(
    s=0.5,
    label="Estimates $Y_i \\mid \\theta_i, \\sigma_i \\sim N(\\theta_i, \\sigma_i^2)$",
    color=ACCENT,
    alpha=0.5,
    rasterized=True,
)

cond_mean_kws = dict(color=CORAL, label="Estimated $E[\\theta \mid \\sigma] = E[Y \mid \\sigma]$")

plt.figure(figsize=(9, 6))
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)
a0 = plt.subplot(gs[0, :2])
a1 = plt.subplot(gs[0, 2:])
a2 = plt.subplot(gs[1, 1:3])

a0.scatter(x=xs, y=estimates, **scatterkws)
a0.scatter(
    x=xs, y=posterior_means, s=0.5, color=ASHER, label="EB posterior means (Independent Gaussian)"
)
a0.axhline(indep_gauss_meta["estimated_grand_mean"], color="k", label="Estimated $E[\\theta]$  ")
a0.legend(frameon=False)

a1.scatter(x=xs, y=estimates, **scatterkws)
a1.scatter(
    x=xs, y=posterior_means_npmle, s=0.5, color=RUBY, label="EB posterior means (Independent NPMLE)"
)
a1.axhline(indep_gauss_meta["estimated_grand_mean"], color="k", label="Estimated $E[\\theta]$  ")
# a1.axhline(indep_gauss_meta["estimated_grand_mean"], color="k", label="Estimated $E[\\theta]$  ")


a1.legend(frameon=False)

a2.scatter(x=xs, y=estimates, **scatterkws)
a2.scatter(
    x=xs, y=posterior_means_close, s=0.5, color=CORAL, label="EB posterior means (CLOSE-NPMLE)"
)
a2.plot(xs[idx], cm[idx], **cond_mean_kws)
a2.legend(frameon=False)

sns.despine()
a2.set_xlabel("$\\log_{10}(\\sigma_i)$")
plt.suptitle(title)

plt.savefig("assets/example_eb_posterior_means.pdf", bbox_inches="tight")


## Figure 3: selection example
selected_ig = posterior_means > np.quantile(posterior_means, 0.67)
selected_close = posterior_means_close > np.quantile(posterior_means_close, 0.67)
idx = [
    (((selected_ig) & (~selected_close) & (xs < -1.1)) * (estimates)).argmax(),
    (((~selected_ig) & (selected_close) & (xs < -1.1)) * (estimates)).argmax(),
]
print("idx", idx, "Expected (5461, 500)")
print(df.loc[idx, ["czname", "tract"]])
print("Expected: Newark, NJ CZ, 34003015200; San Francisco, CA CZ, 06013370000")
print()
#  idx = [5461, 500]
# Englewood, NJ (Newark, NJ CZ, 34003015200) 77% nonwhite
# East Richmond, CA (San Francisco, CA CZ, 06013370000) 57% nonwhite

xlim = (-1.6, -1)
ylim = (0, 0.2)

plt.figure(figsize=(6, 4))
plt.scatter(x=xs, y=estimates, **(scatterkws | {"alpha": 0.1, "label": None}))

plt.axhline(indep_gauss_meta["estimated_grand_mean"], color="k")
plt.scatter(x=xs[idx], y=estimates[idx], s=30, color=ACCENT, marker="X", label="Raw data")
plt.text(
    x=xs[idx[0]] + 0.02,
    y=estimates[idx[0]] + 0.005,
    s="Tract A\nEnglewood, NJ\nNewark-Trenton, NJ CZ\n(FIPS 34003015200)",
    color=ACCENT,
    ha="center",
)

plt.text(
    x=xs[idx[1]],
    y=estimates[idx[1]] + 0.005,
    s="Tract B\nEast Richmond, CA\nSan Francisco-Oakland, CA CZ\n(FIPS 06013370000)",
    color=ACCENT,
    ha="center",
)

idxx = xs.argsort()
plt.scatter(
    x=xs[idx],
    y=posterior_means[idx],
    s=30,
    color=ASHER,
    marker="o",
    label="EB posterior mean (Independent Gaussian)",
)

for r in range(2):
    # Illustrate shrinkage with arrows
    plt.arrow(
        x=xs[idx[r]] + 0.01,
        y=estimates[idx[r]],
        dx=0,
        dy=posterior_means[idx[r]] - estimates[idx[r]],
        color=ASHER,
        width=0.003,
        # head_width=0.15,
        head_length=0.01,
        length_includes_head=True,
        alpha=0.5,
    )

plt.plot(np_df["x"], point, **(cond_mean_kws | dict(label=None)))

plt.scatter(
    x=xs[idx],
    y=posterior_means_close[idx],
    color=CORAL,
    s=50,
    marker="*",
    label="EB posterior mean (CLOSE-NPMLE)",
)

idxx = xs.argsort()

for r in range(2):
    # Illustrate shrinkage with arrows
    plt.arrow(
        x=xs[idx[r]] - 0.01,
        y=estimates[idx[r]],
        dx=0,
        dy=posterior_means_close[idx[r]] - estimates[idx[r]],
        color=CORAL,
        width=0.003,
        # head_width=0.15,
        head_length=0.01,
        length_includes_head=True,
        alpha=0.5,
    )

plt.xlim(xlim)
plt.ylim(ylim)

sns.despine()
plt.xlabel("$\\log_{10}(\\sigma_i)$")
plt.suptitle(
    "Opportunity Atlas estimates for \n P(Income ranks in top 20 | Black, Parent at 25th Percentile)"
)
plt.legend(loc=(0.05, 0.8), frameon=False)
plt.savefig(
    "assets/example_shrink_ranking.pdf",
    bbox_inches="tight",
    dpi=500,
)

# differences = estimates[idx][1] - estimates[idx][0], cm[idx][1] - cm[idx][0]
# (0.050343551, 0.05350922144597065)
