"""Build a processed dataset and offer a function to load it for a given outcome
variable"""

import pandas as pd
import numpy as np

est_vars = [
    "kfr_pooled_pooled_p25",
    "kfr_white_male_p25",
    "kfr_black_male_p25",
    "kfr_black_pooled_p25",
    "kfr_white_pooled_p25",
    "jail_black_male_p25",
    "jail_white_male_p25",
    "jail_black_pooled_p25",
    "jail_white_pooled_p25",
    "jail_pooled_pooled_p25",
    "kfr_top20_black_male_p25",
    "kfr_top20_white_male_p25",
    "kfr_top20_black_pooled_p25",
    "kfr_top20_white_pooled_p25",
    "kfr_top20_pooled_pooled_p25",
]

se_vars = [est_var + "_se" for est_var in est_vars]

identifiers = [
    "state",
    "county",
    "tract",
    "cz",
    "czname",
]

covs = [
    "par_rank_pooled_pooled_mean",
    "par_rank_black_pooled_mean",
    "kid_black_pooled_blw_p50_n",
    "kid_pooled_pooled_blw_p50_n",
]

cov_df_covariates = [
    "poor_share2010",
    "share_black2010",
    "hhinc_mean2000",
    "ln_wage_growth_hs_grad",
    "frac_coll_plus2010",
]

covariates = (
    covs + cov_df_covariates + ["log_kid_black_pooled_blw_p50_n", "log_kid_pooled_pooled_blw_p50_n"]
)

TOP = 20


def build_data(output_dir="data/processed/oa_data_used.feather"):
    """
    Load and merge data from the Opportunity Atlas and the tract-level covariates
    Filter to the TOP commuting zones by number of tracts (TOP=20 by default)
    """
    # Load data and merge in covariates
    df = pd.read_csv(
        "data/raw/tract_outcomes_early.csv",
        usecols=identifiers + est_vars + se_vars + covs,
        low_memory=False,
    )
    cov_df = pd.read_stata("data/raw/tract_covariates.dta")[identifiers + cov_df_covariates]
    df = df.merge(cov_df, on=identifiers, how="left").reset_index(drop=True)

    # Take log for number of kids
    df["log_kid_black_pooled_blw_p50_n"] = np.where(
        df["kid_black_pooled_blw_p50_n"] > 0, np.log(df["kid_black_pooled_blw_p50_n"]), np.nan
    )
    df["log_kid_pooled_pooled_blw_p50_n"] = np.where(
        df["kid_pooled_pooled_blw_p50_n"] > 0, np.log(df["kid_pooled_pooled_blw_p50_n"]), np.nan
    )

    # Filter to the largest [TOP] commuting zones by number of tracts
    cz_by_num_tracts = (
        df.groupby("czname")["tract"].count().sort_values(ascending=False)[:TOP].index
    )
    bools = df["czname"].isin(cz_by_num_tracts)

    to_save = df.loc[bools].reset_index(drop=True)
    to_save.to_feather(output_dir)

    return to_save


def load_data_for_outcome(est_var, input_dir="data/processed/oa_data_used.feather"):
    """
    Load the processed data for a given outcome variable
    Filter out missing values and values where the standard error is too large
    """
    df = pd.read_feather(input_dir)
    se_var = est_var + "_se"
    subset = df[[est_var, se_var, "czname", "state", "county", "tract"] + covariates].dropna()
    thresh = subset[se_var].quantile(0.995)
    subset = subset.loc[subset[se_var] <= thresh].reset_index(drop=True)

    return subset


if __name__ == "__main__":
    build_data()
