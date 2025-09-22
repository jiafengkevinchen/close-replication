# Source Installation Guide

While Docker is the recommended setup for this replication package, you can install the
stack directly on your host machine by following the steps below. This path requires
managing Python, R, MOSEK, and `rpy2` dependencies manually.

## Prerequisites

- Python 3.10 with `pip`
- R 4.4.0 ("Puppy Cup") with [`renv`](https://rstudio.github.io/renv/articles/renv.html) 1.1.4
- MOSEK 10.2.5 and a valid license
- GNU Parallel for running the bash scripts (`brew install parallel` on macOS, or use your
  package manager on Linux)

> **Note**: Rmosek is not supported on aarch64 Linux. See the
> [official platform list](https://docs.mosek.com/10.2/rmosek/install-interface.html) for
> details.

## Step-by-step setup

1. **Python environment**

   Create a clean conda environment and install the Python requirements with `pip`:

   ```bash
   # Install conda first if you do not already have it
   conda create -n eb-replication python=3.10
   conda activate eb-replication

   # Install Python dependencies
   pip install --no-cache-dir -r requirements.txt
   ```

2. **MOSEK**

   Download MOSEK 10.2.5 from <https://www.mosek.com/downloads/10.2.5/> and request a
   license from <https://www.mosek.com/license/request/?i=acp>. Follow Sections 4.2–4.3 in
   the [MOSEK installation guide](https://docs.mosek.com/10.2/install/installation.html)
   to install the solver and place your `mosek.lic` file in a location accessible to both
   Python (via `MOSEKLM_LICENSE_FILE`) and R (`Rmosek`).

3. **R packages**

   With R 4.4.0 installed, use `renv` to restore the required packages:

   ```r
   renv::restore()  # Installs packages listed in ./renv.lock
   ```

4. **Rmosek builder**

   Load `Rmosek` once in R to trigger the builder instructions, then follow the manual
   build steps. You should see the prompt:

   ```r
   > library(Rmosek)

      The Rmosek meta-package is ready. Please call

         mosek_attachbuilder(what_mosek_bindir)

      to complete the installation. See also '?mosek_attachbuilder'.
   ```

   Next, run the builder script provided by MOSEK (replace `<RMOSEKDIR>` with the path to
   your MOSEK installation, e.g. `~/mosek/10.2/tools/platform/osxaarch64/rmosek`):

   ```r
   source("<RMOSEKDIR>/builder.R")
   attachbuilder()
   install.rmosek()
   # Restart the R session when prompted.
   ```

   After installation the `renv` status command will report that Rmosek is
   "out-of-sync"; this warning is expected and can be ignored.

## Verifying the setup

Use the following checks to confirm the Python–R bridge and MOSEK integration are
working. The command outputs shown below match the reference environment.

### rpy2 configuration

```bash
❯ python -m rpy2.situation
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

### Python integration smoke test

```bash
❯ python test.py
- Project '/app' loaded. [renv 1.1.4]
Call: lprobust

Sample size (n)                              =    1000
Polynomial order for point estimation (p)    =    1
Order of derivative estimated (deriv)        =    0
Polynomial order for confidence interval (q) =    2
Kernel function                              =    Epanechnikov
Bandwidth method                             =    imse-dpi



Call:
	NULL

Data:  ( obs.);	Bandwidth 'bw' =

       x                 y
 Min.   :-3.0196   Min.   :0.000000
 1st Qu.:-1.2831   1st Qu.:0.000000
 Median : 0.4534   Median :0.000000
 Mean   : 0.4534   Mean   :0.003333
 3rd Qu.: 2.1898   3rd Qu.:0.000000
 Max.   : 3.9263   Max.   :0.998168
```

Running `test.py` also generates `test.png` in the project root for a quick plotting
sanity check.

### R-only check

```bash
❯ Rscript test.R
Loading required package: Matrix
Warning message:
package ‘nprobust’ was built under R version 4.4.1
Call: lprobust

Sample size (n)                              =    100
Polynomial order for point estimation (p)    =    1
Order of derivative estimated (deriv)        =    0
Polynomial order for confidence interval (q) =    2
Kernel function                              =    Epanechnikov
Bandwidth method                             =    imse-dpi


Call:
	NULL

Data:  ( obs.);	Bandwidth 'bw' =

       x                 y
 Min.   :-2.9932   Min.   :0.000000
 1st Qu.:-1.6732   1st Qu.:0.000000
 Median :-0.3532   Median :0.000000
 Mean   :-0.3532   Mean   :0.003333
 3rd Qu.: 0.9668   3rd Qu.:0.000000
 Max.   : 2.2867   Max.   :0.951492
```

To isolate workflow issues, you can also run

```bash
python check_monte_carlo.py --simulator_name npmle_by_bins --est_var kfr_top20_black_pooled_p25 --seed_number 381
```

which validates one of the Monte Carlo runs against the archived results.

If any of these steps fail, re-run the builder scripts or revisit the MOSEK license
configuration.
