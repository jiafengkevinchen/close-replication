import pandas as pd
import numpy as np


def mean_squared_error(posterior_means):
    is_coupled_bootstrap = "validation" in posterior_means.columns
    true_col = "truth" if not is_coupled_bootstrap else "validation"
    quants = posterior_means.drop(
        columns=(
            [true_col, "czname", "validation_se"]
            if is_coupled_bootstrap
            else [true_col, "truth_residualized", "true_covariate_fn"]
        ),
        axis=1,
    )
    return ((quants - posterior_means[true_col].values[:, None]) ** 2).mean()


def rank_score_within_cz(posterior_means):
    """
    For each method, compute the mean of the validation variable within the top x% of
    each CZ.
    """
    assert "validation" in posterior_means.columns
    true_col = "validation"
    methods = posterior_means.drop(columns=[true_col, "czname", "validation_se"], axis=1).columns
    output = []
    for method_name in methods:
        # Sort by the posterior mean, largest first
        sorted_pm = posterior_means.sort_values(method_name, ascending=False)

        # cum_count is [0, 1, 2, ..., n-1] where n is the number of observations in the CZ
        cum_count = sorted_pm.groupby("czname")[true_col].transform("cumcount")
        counts = sorted_pm.groupby("czname")[true_col].transform("count")

        # cum_prop is [0, 1/n, 2/n, ..., 1] where n is the number of observations in the CZ
        cum_prop = cum_count / counts

        pcts = np.linspace(0.01, 1, 100)  # 1% to 100%

        # selectors is a boolean matrix where each row is a tract and each column is a
        # percentile. The value is True if the tract is in the top x% of the CZ
        selectors = cum_prop.values[:, None] <= pcts[None, :]

        # take mean of the true value for each top percentile
        utils = (selectors * sorted_pm[true_col].values[:, None]).sum(0) / selectors.sum(0)
        output.append(pd.Series(utils, index=pcts, name=method_name))
    return pd.DataFrame(output).T
