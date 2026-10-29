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
LICENSE                     BSD 3-Clause License for the translation
NOTICE.md                   what the license covers, and attribution
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
scripts/04b_bic_tableS8.py                 Table S.VIII under both readings of the BIC penalty
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

- **Main estimates (Table S.XI).** Given the authors' groupings, 162 of 168
  comparisons (objective, BIC, coefficients, cumulative effect, large-T and
  fixed-T standard errors, G = 1..15, plus the objectives found by search)
  agree to the third decimal, and the other 6 differ by at most 0.001. The BIC
  selects G = 10, as in the paper.
- **Searching from scratch.** VNS with (10; 10; 10) recovers the published
  groupings for G = 2, 3, 4, 5 with no country classified differently (10, 6,
  4, and 3 of 10 starts reach the optimum; 4 to 11 seconds per G), and
  Algorithm 1 with 1,000 starts reaches the same optima for G = 2 and 3.
- **Exact algorithm (Table S.I).** For the first 30 countries and G = 2, the
  numba-compiled branch and bound (161 million nodes, 33 seconds) gives
  objective 6.159134 and theta = (0.6070, 0.0689), as in the authors'
  RESULTS2.DAT and outputtheta.txt; Algorithm 1 and VNS reach the same
  solution. It also agrees with brute-force enumeration on small problems.
- **Table S.III** "true" values: 9 of 9. **Table S.XIV**: 20 of 20.
  **Table S.XV**: 20 of 20.
- **Table S.XVI.** Both heterogeneous-coefficient versions reproduce the
  published estimates and standard errors (using the groupings found by the R
  search).
- **Tables S.XVII–S.XVIII.** All multinomial-logit coefficients and standard
  errors match, with N = 68 in column (10); the descriptive statistics match
  except two entries for log GDP per capita in 1500 (Group 1 s.d. 0.290 vs
  0.300; Group 3 mean 6.478 vs 6.49).
- **Unit tests:** `python -m pytest tests` (18 tests), run automatically on
  Linux and Windows by GitHub Actions on every push.

- **Monte Carlo (Tables S.III–S.IV).** `scripts/04_montecarlo.py` with
  i.i.d. errors (one core per run; 1,000 replications for G0 = 3, 600 for
  G0 = 5 and 10; outputs in `output/montecarlo_G*_iid.csv`):

  | G0 | GFE mean (theta1, theta2, cumulative) | Published | Misclassified (published) | Coverage, large-T / fixed-T, theta1 (published) |
  |---|---|---|---|---|
  | 3 | 0.391, 0.099, 0.164 | 0.391, 0.099, 0.163 | 9.7% (9.5%) | 0.86 / 0.96 (0.85 / 0.97) |
  | 5 | 0.259, 0.087, 0.118 | 0.262, 0.086, 0.117 | 7.8% (9.7%) | 0.83 / 0.98 (0.85 / 0.96) |
  | 10 | 0.282, 0.080, 0.111 | 0.286, 0.078, 0.110 | 16.6% (44.7%) | 0.80 / 0.93 (0.80 / 0.90) |

  Means, standard errors, and coverage agree with the published tables to
  within Monte Carlo error. The misclassification frequency for G0 = 10 is much
  lower than published. We have not traced why; two candidates are how groups
  are matched to the truth before counting (we use the exact Hungarian
  matching) and VNS reaching better optima. Bootstrap standard errors
  (`BOOTSTRAP_B > 0`) were not run at full size.
- **Monte Carlo with wild-bootstrap errors (Section S.6.2).** G0 = 3, 600
  replications (`output/montecarlo_G3_wild.csv`): GFE means 0.376, 0.094,
  0.152; misclassification 9.2%; fixed-T coverage 0.83, 0.90, 0.93.

- **BIC choice of G (Table S.VIII): reproduced, once the original code's
  variance estimate is used.** `scripts/04b_bic_tableS8.py`, 100 samples,
  groupings by Algorithm 1 with 100 starts (`output/tableS8_bic_G3.csv`,
  `output/tableS8_bic_G10.csv`):

  | G0 = 3: source of sigma2_hat | sigma2_hat | G = 3 | G = 4 | G = 5 | G = 6 | G = 7 |
  |---|---|---|---|---|---|---|
  | each sample's own G = 15 fit, as (S20) and footnote 12 describe | 0.023 | 0 | 2 | 43 | 51 | 4 |
  | G = 15 groupings from the G0 = 10 samples, as `BIC_G3_MC.m` does | 0.044 | **98** | **2** | 0 | 0 | 0 |
  | true error variance | 0.032 | 15 | 76 | 9 | 0 | 0 |
  | published | | 98 | 2 | 0 | 0 | 0 |

  The difference is one line of `MonteCarlo/BIC_G3_MC.m`:
  `BigG = load('assignmentG010G15.txt')`. It computes sigma2_hat for the
  G0 = 3 samples from the G = 15 groupings estimated on the G0 = 10 samples
  (the same file `BIC_G10_MC.m` loads), so the groupings do not belong to the
  data they are applied to. Q(15) is then much larger than the samples' own
  G = 15 fit, sigma2_hat roughly doubles, and so does the penalty. With the
  variance estimate the text describes, the BIC overfits (G = 5 or 6); with the
  true variance it still picks G = 4 most often. So the published 98% depends on
  the inflated penalty. The appendix's footnote 27 (a steeper penalty chose
  G = 1 in every simulation) is consistent with the choice being sensitive to
  the penalty's scale. `04_montecarlo.py` offers both readings
  (`BIC_SIGMA2 = 'own'` or `'g0_10'`) and now compares G = 1..7, as the
  original does.

  For G0 = 10, `BIC_G10_MC.m` uses each sample's own G = 15 fit, and sigma2_hat
  is close to the true variance (0.0165 against 0.0166). Over G = 3..15 the BIC
  chose G = 8, 9, 10, 11 in 27, 46, 19, 2 samples (and G = 6-7 in 6), against
  the published 10, 42, 42, 6: the same centre, shifted down by about half a
  group. The shift comes from the search, not the penalty: in 30 samples
  (G = 6..12, true variance), keeping the better of Algorithm 1 and VNS lowered
  Q(G) more the larger G was (by 0.05 at G = 6, 0.23 at G = 12), and moved the
  choice up, from G = 7-11 in 3, 7, 9, 9, 2 samples to G = 7-10 in 1, 6, 13, 10,
  closer to the published distribution. The authors' groupings came from their
  Fortran program (MC1.exe), which presumably searched harder.

  Our earlier runs (VNS, as in `04_montecarlo.py`: G = 6 in 90 of 100, G = 5 in
  10; the R translation: G = 6 in 29 of 30) used the own-fit sigma2_hat over
  G = 1..6 and agree with the first row.

Not yet run at full size in Python: the N = 90 exact solution.

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
- **BIC in the Monte Carlo.** `BIC_G3_MC.m` and `BIC_G10_MC.m` average
  sigma2_hat across simulated samples and use the average in every sample; the
  scripts do the same. `BIC_G3_MC.m` takes the G = 15 groupings for sigma2_hat
  from the G0 = 10 runs (`assignmentG010G15.txt`), which doubles the penalty and
  is what produces the published 98% (see Validation); `BIC_SIGMA2 = 'g0_10'`
  emulates it, and the default `'own'` follows the text.
- **Misclassification.** `match_groups()` uses the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`), which finds the exact minimum over
  all G! relabelings.
- **Heterogeneous coefficients.** The paper uses 1,000,000 random starts.
  `gfe_hetero(..., polish_top=k)` adds a single-unit local search to the best
  k solutions.

---

## License

The Python translation is Copyright (c) 2026 Eliot R. Weinstein and is licensed
under the BSD 3-Clause License (`LICENSE`), the license of NumPy, SciPy,
pandas, scikit-learn, and statsmodels. (The companion R translation,
gfe_bm_R, uses the GNU GPL, following the R community's convention.)

The license covers only the translation: the Python code, the scripts, the
tests, and the documentation. It grants no rights in the grouped fixed-effects
method and the original MATLAB, Fortran, and Stata code, which are Stéphane
Bonhomme and Elena Manresa's, or in the data, which come from Acemoglu,
Johnson, Robinson, and Yared (2008) via Bonhomme and Manresa's replication
files. Both remain subject to their own terms; see `NOTICE.md`. If you use this
code, please cite Bonhomme and Manresa (2015); see `CITATION.md`.
