# Replication files for Chen (2025) "Empirical Bayes When Estimation Precision Predicts Parameters"

Current as of https://arxiv.org/abs/2212.14444v5

Jiafeng Chen

December 10, 2024

## Reproducing results

```bash
# Run the Monte Carlo
# The following generates results/[simulator-name]
# where [simulator-name] is one of "coupled_bootstrap-0.9",
# "covariate_additive_model", "npmle_by_bins", "weibull"

# Calibrated simulation exercise
sh monte_carlo.sh # see note at the end

# Validation exercise using coupled bootstrap
sh coupled_bootstrap.sh

# Weibull exercise in OA5.3
sh weibull_model.sh

# Additive model exercise in OA5.4
sh additive_model.sh

# Clean up the generated raw Monte Carlo results
python generate_scores.py --simulator-name coupled_bootstrap-0.9 --nsim 1000
python generate_scores.py --simulator-name npmle_by_bins --nsim 1000
python generate_scores.py --simulator-name weibull --nsim 100
# The additive model results are processed directly

# Generate figures and tables in assets/
# Fig 1, 2, 3 and footnote 6
python assets_introduction.py

# Fig 4, 5
python assets_empirical.py

# Table OA5.1, Fig OA5.1, OA5.2, OA5.3, OA5.4
python assets_appendix.py
```


### Running a small test
```bash
python empirical_exercise.py --nsim 1 --starting-seed 94999 --data-dir "data/simulated_posterior_means_sample" --simulator-name npmle_by_bins --methods all --est_var kfr_black_pooled_p25

python empirical_exercise.py --nsim 1 --starting-seed 94999 --data-dir "data/simulated_posterior_means_sample" --simulator-name coupled_bootstrap-0.9 --methods all --est_var kfr_black_pooled_p25

python empirical_exercise.py --nsim 1 --starting-seed 94399  --data-dir "data/simulated_posterior_means_sample" --simulator-name weibull --methods indep_gauss,close_npmle,close_gauss,close_gauss_parametric  --est_var kfr_black_pooled_p25
```

```python
import pandas as pd

# Reproduces as of 2024-12-10
orig = pd.read_feather("data/simulated_posterior_means/coupled_bootstrap-0.9/kfr_black_pooled_p25/94999.feather")
new = pd.read_feather("data/simulated_posterior_means_sample/coupled_bootstrap-0.9/kfr_black_pooled_p25/94999.feather")
print((orig.drop("czname", axis=1) - new.drop("czname", axis=1)).values.std())

orig = pd.read_feather("data/simulated_posterior_means/npmle_by_bins/kfr_black_pooled_p25/94999.feather")
new = pd.read_feather("data/simulated_posterior_means_sample/npmle_by_bins/kfr_black_pooled_p25/94999.feather")
print((orig - new).values.std())

orig = pd.read_feather("data/simulated_posterior_means/weibull/kfr_black_pooled_p25/94399.feather")
new = pd.read_feather("data/simulated_posterior_means/weibull/kfr_black_pooled_p25/94399.feather")
print((orig - new).values.std())

# 0.0
# 0.0
# 0.0
```

## Data sources
 - The Opportunity Atlas (https://opportunityinsights.org/paper/the-opportunity-atlas/)
    + `data/raw/tract_covariates.dta` ("Neighborhood Characteristics by Census Tract"):
      https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes_dta.zip
      (Accessed 2024-09-16)
    + `data/raw/tract_outcomes_early.csv` ("All Outcomes by Census Tract, Race, Gender and Parental Income Percentile"):
    https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes.zip
    (Accessed 2024-09-16)

## Versions and packages

See environment.yml for list of python package.

[How to install `rmosek`](https://docs.mosek.com/latest/rmosek/install-interface.html).

```R
> R.version
platform       aarch64-apple-darwin20
arch           aarch64
os             darwin20
system         aarch64, darwin20
status
major          4
minor          2.1
year           2022
month          06
day            23
svn rev        82513
language       R
version.string R version 4.2.1 (2022-06-23)
nickname       Funny-Looking Kid


> Rmosek::mosek_version()
[1] "MOSEK 10.2.5"

> library(REBayes) # Load required packages
> library(nprobust)
> sessionInfo()
R version 4.2.1 (2022-06-23)
Platform: aarch64-apple-darwin20 (64-bit)
Running under: macOS 15.1.1

Matrix products: default
LAPACK: /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/lib/libRlapack.dylib

locale:
[1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
[1] REBayes_2.51   Matrix_1.6-4   nprobust_0.4.0

loaded via a namespace (and not attached):
 [1] Rcpp_1.0.11      rstudioapi_0.14  magrittr_2.0.3   splines_4.2.1    tidyselect_1.2.0 munsell_0.5.0    colorspace_2.1-0 lattice_0.20-45  R6_2.5.1
[10] rlang_1.1.2.9000 fansi_1.0.5      dplyr_1.1.3      tools_4.2.1      grid_4.2.1       gtable_0.3.4     utf8_1.2.4       cli_3.6.1        tibble_3.2.1
[19] lifecycle_1.0.4  ggplot2_3.4.4    vctrs_0.6.4      glue_1.6.2       compiler_4.2.1   pillar_1.9.0     generics_0.1.3   scales_1.2.1     Rmosek_10.2.0
[28] pkgconfig_2.0.3
```

Rpy2 configuration
```bash
$ python -m rpy2.situation
rpy2 version:
3.5.16
Python version:
3.9.13 | packaged by conda-forge | (main, May 27 2022, 17:01:00)
[Clang 13.0.1 ]
Looking for R's HOME:
    Environment variable R_HOME: None
    Calling `R RHOME`: /Library/Frameworks/R.framework/Resources
    Environment variable R_LIBS_USER: None
R's value for LD_LIBRARY_PATH:

R version:
    In the PATH: R version 4.2.1 (2022-06-23) -- "Funny-Looking Kid"
    Loading R library from rpy2: OK
Additional directories to load R packages from:
None
C extension compilation:
  include:
  ['/Library/Frameworks/R.framework/Resources/include']
  libraries:
  ['pcre2-8', 'lzma', 'bz2', 'z', 'icucore', 'dl', 'm', 'iconv']
  library_dirs:
  ['/opt/R/arm64/lib', '/opt/R/arm64/lib']
  extra_compile_args:
  ['-std=c99']
  extra_link_args:
  ['-F/Library/Frameworks/R.framework/..', '-framework', 'R']
Directory for the R shared library:
lib
CFFI extension type
  Environment variable: RPY2_CFFI_MODE
  Value: CFFI_MODE.ANY
  ABI: PRESENT
  API: PRESENT
```

### Note
For some upstream reason having to do with MOSEK or REBayes, running `monte_carlo.sh` for
many iterations might silently fail, due to memory leak. When it has failed, the code
would appear to run but resource consumption is low and no new output is generated.
Interrupting the code prints `Segmentation fault`. I find it quite difficult to reproduce
the issue, as there's no fixed data seed causing a problem. When this happens,
interrupting and restarting resolves the issue. This has only happened when the I
repeatedly apply NPMLE to sample new data and to estimate various
methods.

## References

O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
