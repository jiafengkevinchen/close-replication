import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

CORAL = "#F26D21"
ALICE = "#107895"
RUBY = "#9a2515"
ASHER = "#555F61"
ACCENT = "#268bd2"

sns.set_palette(sns.color_palette([ACCENT, CORAL, RUBY, ASHER, ALICE]))

explanation = {
    "kfr_pooled_pooled_p25": "Mean income rank",
    "kfr_white_pooled_p25": "Mean income rank [white]",
    "kfr_black_pooled_p25": "Mean income rank [Black]",
    "kfr_white_male_p25": "Mean income rank [white male]",
    "kfr_black_male_p25": "Mean income rank [Black male]",
    "kfr_top20_pooled_pooled_p25": "P(Income ranks in top 20)",
    "kfr_top20_white_pooled_p25": "P(Income ranks in top 20 | white)",
    "kfr_top20_black_pooled_p25": "P(Income ranks in top 20 | Black)",
    "kfr_top20_white_male_p25": "P(Income ranks in top 20 | white male)",
    "kfr_top20_black_male_p25": "P(Income ranks in top 20 | Black male)",
    "jail_pooled_pooled_p25": "Incarceration",
    "jail_white_pooled_p25": "Incarceration [white]",
    "jail_black_pooled_p25": "Incarceration [Black]",
    "jail_white_male_p25": "Incarceration [white male]",
    "jail_black_male_p25": "Incarceration [Black male]",
}

method_names = {
    "close_npmle": ("CLOSE-NPMLE", CORAL),
    "indep_gauss": ("Independent Gaussian", ASHER),
    "close_gauss": ("CLOSE-Gauss", RUBY),
    "close_gauss_parametric": ("CLOSE-Gauss (parametric)", ALICE),
    "naive": ("Naive", ACCENT),
}


def plot_league_table_value_basic_eb(league_table, methods, legend_x=1.05):
    for i, method in enumerate(methods):
        method_name = method.replace("_nocov", "")
        nocov = "_nocov" in method

        name, color = method_names[method_name]
        offset = (1 if nocov else -1) * 0.15
        plt.scatter(
            x=(league_table[method] - league_table["naive"]) * 100,
            y=np.arange(len(league_table))[::-1] + offset + np.random.uniform(-0.1, 0.1),
            marker="o" if not nocov else "x",
            color=color,
            label=name + (" [no residualization]" if nocov else ""),
        )

    plt.yticks(np.arange(len(league_table))[::-1], league_table.rename(index=explanation).index)
    for y in np.arange(len(league_table))[::-1]:
        # plt.axhline(y=y - offset, color="grey", linewidth=0.5, ls="--")
        plt.axhline(y=y, color="grey", linewidth=0.5, ls="--")
    plt.axvline(0, color="r", alpha=0.3)
    plt.xlim((-3, 4))
    plt.xlabel(
        "Performance difference relative to screening on raw estimates (percentile rank or percentage point)"
    )
    sns.despine()
    plt.legend(loc=(legend_x, 0), frameon=False)


def plot_mse_league_table(league_table, methods, legend_x=1.05):
    for i, method in enumerate(methods):
        method_name = method.replace("_nocov", "")
        nocov = "_nocov" in method

        name, color = method_names[method_name]
        offset = (1 if nocov else -1) * 0.1
        plt.scatter(
            x=league_table["naive"] - league_table[method],
            y=np.arange(len(league_table))[::-1] + offset + np.random.uniform(-0.05, 0.05),
            marker="o" if not nocov else "x",
            color=color,
            label=name + (" [no covariates]" if nocov else ""),
        )

    plt.yticks(np.arange(len(league_table))[::-1], league_table.rename(index=explanation).index)
    for y in np.arange(len(league_table))[::-1]:
        # plt.axhline(y=y - offset, color="grey", linewidth=0.5, ls="--")
        plt.axhline(y=y, color="grey", linewidth=0.5, ls="--")
    plt.axvline(0, color="r", alpha=0.3)
    # plt.xlim((-3, 4))
    plt.xlabel("MSE improvement over Naive (higher is better)")
    sns.despine()
    plt.legend(loc=(legend_x, 0), frameon=False)
