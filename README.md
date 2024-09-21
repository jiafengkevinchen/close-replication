## Data sources
 - The Opportunity Atlas (https://opportunityinsights.org/paper/the-opportunity-atlas/)
    + `data/raw/tract_covariates.dta` ("Neighborhood Characteristics by Census Tract"):
      https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes_dta.zip
      (Accessed 2024-09-16)
    + `data/raw/tract_outcomes_early.csv` ("All Outcomes by Census Tract, Race, Gender and Parental Income Percentile"):
    https://opportunityinsights.org/wp-content/uploads/2018/10/tract_outcomes.zip
    (Accessed 2024-09-16)

## R environment specs

```
platform       aarch64-apple-darwin20
arch           aarch64
os             darwin20
system         aarch64, darwin20
status
major          4
minor          4.0
year           2024
month          04
day            24
svn rev        86474
language       R
version.string R version 4.4.0 (2024-04-24)
nickname       Puppy Cup


> Rmosek::mosek_version()
[1] "MOSEK 10.2.5"

# TODO
```

Rpy2 configuration
```
❯ python -m rpy2.situation
rpy2 version:
3.5.16
Python version:
3.10.14 (main, May  6 2024, 14:42:37) [Clang 14.0.6 ]
Looking for R's HOME:
    Environment variable R_HOME: None
    Calling `R RHOME`: /Library/Frameworks/R.framework/Resources
    Environment variable R_LIBS_USER: None
R's value for LD_LIBRARY_PATH:

R version:
    In the PATH: R version 4.4.0 (2024-04-24) -- "Puppy Cup"
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
