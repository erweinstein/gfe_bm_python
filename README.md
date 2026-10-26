# Grouped fixed-effects (GFE) estimation in Python

### A Python translation of the replication code for Bonhomme and Manresa (2015)

> Bonhomme, Stéphane and Elena Manresa (2015), "Grouped Patterns of Heterogeneity
> in Panel Data," *Econometrica* 83(3), 1147–1184. doi:10.3982/ECTA11319

**Python translation by**  
Eliot R. Weinstein  
eliotw@uchicago.edu  
[erweinstein.com](https://erweinstein.com)

**Credit.** The grouped fixed-effects estimator, its asymptotic theory, the
computational algorithms (Algorithm 1, variable neighborhood search, the exact
branch-and-bound extension), the fixed-T variance, the Monte Carlo design, the
empirical application, and the original MATLAB, Fortran, and Stata code are the
work of **Stéphane Bonhomme and Elena Manresa**. This repository translates
their replication package (`Bonhomme_Manresa_codes`, distributed with the paper
by the Econometric Society) into Python, by way of my R translation,
[gfe_bm_R](https://github.com/erweinstein/gfe_bm_R), which follows the
original files line by line. Every source file names the original file it
follows. Please cite their paper when you use this code (see `CITATION.md`).
The Python translation is by Eliot R. Weinstein; it was not reviewed by the
original authors, and errors introduced in translation are his.

The code has two goals:

1. **Preserve the method.** The estimator, the algorithms, and the inference
   procedures follow the original code step by step, and the scripts check the
   results against the numbers published in the paper and its Supplementary
   Appendix (see "Validation" below).
2. **Serve as a model of the research workflow.** The scripts follow the
   structure of the paper: the estimator as an optimization problem, the
   algorithms and a check of their accuracy, the main estimates, inference
   with a calibrated Monte Carlo, robustness organized around the main
   finding, and an attempt to explain it. `GFE_walkthrough.qmd` reads the
   paper alongside the code, and `EXAMPLE.md` is a short first exercise.

---

## Quick start

Python 3.9 or later. From the project folder:

```bash
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib (+ optional numba, pytest)
python scripts/02_application_income_democracy.py
```

or interactively (Jupyter, VS Code, Spyder, or RStudio with reticulate):

```python
from gfe import load_bm_panel, gfe
p = load_bm_panel()                       # 90 countries x 7 periods (1970-2000)
fit = gfe(p, G=4, algorithm='vns', n_starts=10, neighmax=10, itermax=10, seed=1)
print(fit)                                # objective 14.31867, theta = (0.302, 0.082)
```

**Dependencies.** numpy, scipy, pandas, and matplotlib. Optional: `numba`
compiles the exact branch-and-bound algorithm (without it, the exact algorithm
runs in plain Python, which is fine for small problems but far too slow for
N = 30); `pytest` runs the tests.

**Parallel computing.** `gfe_bootstrap(..., cores=4)` and
`scripts/04_montecarlo.py` (`CORES = 4`) use worker processes
(`concurrent.futures`). Each replication gets its own child seed, so results
do not depend on the number of cores. On Windows, run them as scripts (each
script has the required `if __name__ == "__main__":` guard).

---

## What is here

```
LICENSE                     GNU GPL (>= 2) for the translation, with scope note
CITATION.md                 how to cite the paper, the data, and this translation
EXAMPLE.md                  a first exercise: change G and the starting values
GFE_walkthrough.qmd         a reading of the paper with the code (Quarto, Python)
requirements.txt            dependencies
gfe/__init__.py             the package: `from gfe import ...`
gfe/core.py                 the estimator: update step, assignment step,
                            Algorithm 1, local search, VNS (Algorithm 2)
gfe/exact.py                exact solution: Brusco's repetitive branch and bound
                            with covariates (numba-compiled when available);
                            brute force for tiny problems
gfe/inference.py            large-T clustered variance, fixed-T (Pollard)
                            variance and bandwidth rule, unit-level bootstrap
gfe/select.py               BIC for the number of groups
gfe/extensions.py           deviations from unit means, first-differenced IV,
                            heterogeneous coefficients, two layers of groups
gfe/utils.py                Hungarian matching of group labels, group means,
                            plotting in the paper's style, pinv, cluster-robust IV
gfe/data.py                 load_bm_panel(), load_bm_published()
scripts/00_build_data.py    builds the panel from the AJRY (2008) data
scripts/01_validate_against_published.py   Tables S.I, S.III, S.XI, S.XIV, S.XV
scripts/02_application_income_democracy.py Table S.XI, Figures 1, 2, S.4
scripts/03_exact_algorithm.py              Table S.I (exact vs heuristics)
scripts/04_montecarlo.py                   Tables S.III-S.VIII, Figure S.1
scripts/05_extensions.py                   Tables S.XIV-S.XVI, Figures S.5-S.7
scripts/06_explaining_the_waves.py         Tables S.XVII-S.XVIII
tests/test_gfe.py           unit tests (pytest) against brute force, direct
                            least squares, direct formulas, published numbers
data/                       see below
output/                     tables (CSV) and figures (PDF) written by the scripts
```

### Data

| file | content | source |
|---|---|---|
| `bm_final_data.csv` | estimation sample: `dem`, `lag_dem`, `lag_income`, N = 90, T = 7 | the authors' `final_data.mat` |
| `ajry_5yearpanel_extract.csv` | raw variables used to build the sample (90 sample countries, 1965–2000) | the authors' copy of AJRY's `5yearpanel.dta` |
| `explaining_the_waves_G4.csv` | covariates for Tables S.XVII–S.XVIII | the authors' `explaining_the_waves_G4.dta` |
| `bm_published_groupings.csv` | optimal groupings G = 2..15, continents, deviations G = 2..5, two-layer (5,2,2), **1-based labels** | hard-coded in `GFE_estimates.m`, `GFE_deviations_estimates.m`, `GFE_twolayers_estimates.m` |
| `bm_published_bootstrap_se.csv` | bootstrap standard errors, G = 2..15 | `std_cluster_b` in `GFE_estimates.m` |
| `rcode_hetero_groupings.csv` | groupings for the heterogeneous-coefficient models, **1-based labels** | found by the R translation (not published by the authors) |

The underlying data are from Acemoglu, Johnson, Robinson, and Yared (2008),
"Income and Democracy," *AER* 98(3), 808–842,
[jstor.org/stable/29730096](https://www.jstor.org/stable/29730096). They remain
subject to their original terms (see "License" below).

---

## Translating MATLAB (and R) to Python: conventions used here

- **Groups are 0-based inside the package.** MATLAB, Fortran, R, and the paper
  label groups 1..G; Python indexes from 0, so `fit.groups` takes values
  0..G-1. `load_bm_published()` subtracts 1 from the published groupings when
  it reads them, and the scripts add 1 back wherever a label is shown to a
  reader. The one exception is the inner loop of the exact algorithm, which
  keeps Fortran's 1-based labels on purpose so it can be compared with
  `Brusco_cov.f90` line by line (see `gfe/exact.py`).
- **Data are stacked unit by unit**, as in the original: element `i*T + t`
  holds unit `i`, period `t` (MATLAB's `(i-1)*T + t`).
- **MATLAB operators.** `A'` is `A.T`; `A*B` (matrix product) is `A @ B`;
  `A.*B` (elementwise) is `A * B`; `A\b` is `np.linalg.solve(A, b)` (or
  `lstsq` when A is not square); `kron(gi, eye(T))` design matrices are built
  by direct indexing.
- **Random numbers.** Every function that draws random numbers takes `seed`
  (an int, `None`, or a `numpy.random.Generator`, so one stream can be shared
  across calls, like `set.seed()` in R). Python's and R's random streams
  differ, so a random search here takes different paths than in R, but it
  reaches the same optima.
- **Quantiles.** MATLAB's `quantile` (used for the IQR in the bandwidth rule)
  is `np.percentile(..., method='hazen')`.
- **Input checks** raise `ValueError` with a message saying what is wrong,
  rather than using `assert` (Python skips asserts when run with `-O`).

---

## Validation

The numbers below were produced by the scripts in this folder; they agree with
the R translation to all printed digits.

- **Main estimates (Table S.XI).** Given the authors' groupings, 160 of 166
  published entries (objective, BIC, coefficients, cumulative effect, large-T
  and fixed-T standard errors, G = 1..15) agree to the third decimal, and the
  other 6 differ by at most 0.001. The BIC selects G = 10, as in the paper.
- **Table S.III** "true" values: 9 of 9. **Table S.XIV**: 20 of 20.
  **Table S.XV**: 20 of 20.
- **Exact algorithm.** Branch and bound agrees with brute-force enumeration
  on small simulated problems (tests).
- **Table S.XVI.** Both heterogeneous-coefficient versions reproduce the
  published estimates and standard errors (using the groupings found by the R
  search).
- **Tables S.XVII–S.XVIII.** All multinomial-logit coefficients and standard
  errors match, with N = 68 in column (10); the descriptive statistics match
  except two entries for log GDP per capita in 1500 (Group 1 s.d. 0.290 vs
  0.300; Group 3 mean 6.478 vs 6.49).
- **Unit tests:** `python -m pytest tests` (18 tests).

Not yet run in Python (expensive; the R versions were validated): the
random-start searches of `scripts/01` Part B, the N = 30 exact solution of
`scripts/03`, and a full-size Monte Carlo of `scripts/04`. See the R
repository's README for those results.

---

## Differences from the original code, and things we noticed

These are documented in the code where they occur.

- **Fixed-T bandwidth.** `GFE_estimates.m`, as distributed, snaps the
  rule-of-thumb bandwidth to the grid `linspace(0.05, 0.15, 10)`. The fixed-T
  standard errors published in Table S.XI are reproduced by the unsnapped rule,
  which is our default (`gfe_bandwidth(fit, snap_to_grid=True)` gives the
  snapped version).
- **Bootstrap standard errors for G = 7.** In `GFE_estimates.m` the G = 7
  entry of the plotted bootstrap bands reads `std_cluster2_b` instead of
  `std_cluster_b`, so Figure 1 reuses the G = 8 values at G = 7 (flagged in
  `data/bm_published_bootstrap_se.csv`).
- **G = 1, large-T standard error of theta1.** The MATLAB formula gives 0.0483;
  the table reports 0.049, which is what Stata's small-sample factors give.
- **First-differenced IV.** Stata's `ivregress` applies no small-sample factor
  to clustered standard errors; `ols_iv_cluster(small='none')` matches Table
  S.XV. The standard errors of the cumulative effect in Table S.XV are not
  reproduced: the code that computed them is not in the replication files.
- **Exact algorithm tolerance.** `Brusco_cov.f90` tests `ABS(DISC) >
  1*10**(-6)`; with integer arithmetic, `10**(-6)` evaluates to 0 in Fortran.
  We use a small relative tolerance. The published objective 6.1591 (N = 30)
  is for data rounded to 4 decimals, as in the authors' input files.
- **Monte Carlo errors, alternative DGP.** The text of Section S.6.2 describes
  resampling residual vectors; `DGP_final.m` uses a wild bootstrap. Both are
  available (`DGP = 'resample'`, `DGP = 'wild'`).
- **BIC in the Monte Carlo.** `BIC_G3_MC.m` averages sigma2_hat across
  simulated samples; the script does the same.
- **Misclassification.** `match_groups()` uses the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`), which finds the exact minimum over
  all G! relabelings.
- **Heterogeneous coefficients.** The paper uses 1,000,000 random starts.
  `gfe_hetero(..., polish_top=k)` adds a single-unit local search to the best
  k solutions.

---

## License

The Python translation is Copyright (C) 2026 Eliot R. Weinstein and is licensed
under the GNU General Public License, version 2 or (at your option) any later
version; see `LICENSE`.

The license covers only the translation: the Python code, the scripts, the
tests, and the documentation. It grants no rights in the grouped fixed-effects
method and the original MATLAB, Fortran, and Stata code, which are Stéphane
Bonhomme and Elena Manresa's, or in the data, which come from Acemoglu,
Johnson, Robinson, and Yared (2008) via Bonhomme and Manresa's replication
files. Both remain subject to their own terms. If you use this code, please
cite Bonhomme and Manresa (2015); see `CITATION.md`.
