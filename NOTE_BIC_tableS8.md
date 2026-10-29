# Note: the BIC choice of the number of groups (Table S.VIII)

Eliot R. Weinstein, 29 October 2026

This note explains one discrepancy between the Supplementary Appendix of
Bonhomme and Manresa (2015) and their replication code. It affects only the
Monte Carlo of the BIC choice of the number of groups (Table S.VIII). It does
not affect the estimator, the empirical application, or the other Monte Carlo
tables.

## What the appendix reports

The appendix selects the number of groups G by minimizing

    BIC(G) = Q(G)/NT + sigma2_hat (GT + N + K)/NT ln(NT),          (S20)

where Q(G) is the GFE objective with G groups, and (footnote 12)

    sigma2_hat = Q(Gmax)/(NT - Gmax T - N - K),   Gmax = 15.

The design has N = 90, T = 7, K = 2. Table S.VIII reports, over 100 simulated
samples:

- true G0 = 3: G = 3 chosen in 98% of samples, G = 4 in 2%;
- true G0 = 10: G = 8, 9, 10, 11 chosen in 10%, 42%, 42%, 6%.

## What we first found

Following (S20) and footnote 12, with sigma2_hat from each simulated sample's own
G = 15 fit, both translations chose too many groups for G0 = 3. The Python
translation chose G = 6 in 90 of 100 samples and G = 5 in 10; the R translation
chose G = 6 in 29 of 30 and G = 5 in 1. Neither chose G = 3.

We checked the obvious explanations before looking at the original code:

1. The penalty formula matches (S20) and footnote 12.
2. sigma2_hat averaged 0.021 to 0.024 across samples, against a true error
   variance of 0.032. So the penalty was too small, which favours larger G.
3. A stronger optimizer did not help. It lowered Q(G) further and made the
   choice worse (with variable neighborhood search, G = 6 in 19 of 20 samples).
4. Plugging in the true error variance still chose G = 3 in only 15 of 100
   samples (G = 4 in 76).

So the published 98% could not be explained by the optimizer or by the variance
estimate alone.

## What the original code does

The replication files (`Bonhomme_Manresa_codes`, from Stéphane Bonhomme's
website: https://sites.google.com/site/stephanebonhommeresearch/) include
`MonteCarlo/BIC_G3_MC.m` and `MonteCarlo/BIC_G10_MC.m`. Both compute sigma2_hat
for each sample from a G = 15 grouping, average it over the 100 samples
(`sigma22 = mean(sigma2)`), and use the average in every sample, as the
translations do.

The difference is in where `BIC_G3_MC.m` gets the G = 15 grouping. It loads the
G0 = 3 data and then

    G = 15;
    BigG = load('assignmentG010G15.txt');

`assignmentG010G15.txt` holds the G = 15 groupings estimated on the **G0 = 10**
samples. It is the same file `BIC_G10_MC.m` loads for the G0 = 10 samples, where
it belongs. In `BIC_G3_MC.m` it is applied to the G0 = 3 samples, whose units it
was never fitted to. With those groupings, Q(15) is much larger than each G0 = 3
sample's own G = 15 fit, so sigma2_hat roughly doubles (0.044 instead of 0.023),
and so does the penalty. The rest of `BIC_G3_MC.m` loads the G0 = 3 groupings
(`assignmentG03G2.txt` to `assignmentG03G7.txt`) and compares G = 1 to 7.

## Reproducing Table S.VIII

`scripts/04b_bic_tableS8.py` runs both readings on 100 simulated samples, with
groupings estimated by Algorithm 1 with 100 random starts.

True G0 = 3 (`output/tableS8_bic_G3.csv`):

| Source of sigma2_hat | sigma2_hat | G = 3 | G = 4 | G = 5 | G = 6 | G = 7 |
|---|---|---|---|---|---|---|
| Each sample's own G = 15 fit, as (S20) and footnote 12 describe | 0.023 | 0 | 2 | 43 | 51 | 4 |
| G = 15 groupings from G0 = 10 samples, as `BIC_G3_MC.m` does | 0.044 | **98** | **2** | 0 | 0 | 0 |
| True error variance | 0.032 | 15 | 76 | 9 | 0 | 0 |
| Published (Table S.VIII) | | 98 | 2 | 0 | 0 | 0 |

Emulating `BIC_G3_MC.m` (a G = 15 grouping estimated on an independent G0 = 10
sample, applied to the G0 = 3 sample) reproduces the published result exactly.

True G0 = 10 (`output/tableS8_bic_G10.csv`). Here `BIC_G10_MC.m` uses each
sample's own G = 15 fit, and sigma2_hat is close to the true variance (0.0165
against 0.0166):

| | G = 6-7 | G = 8 | G = 9 | G = 10 | G = 11 |
|---|---|---|---|---|---|
| Each sample's own G = 15 fit (100 samples) | 6 | 27 | 46 | 19 | 2 |
| Published (%) | 0 | 10 | 42 | 42 | 6 |

The centre matches; ours is shifted down by about half a group. That shift comes
from the search, not the penalty. In 30 samples (G = 6 to 12, true variance),
keeping the better of Algorithm 1 and variable neighborhood search lowered Q(G)
more the larger G was (by 0.05 at G = 6 and 0.23 at G = 12) and moved the choice
up, from G = 7 to 11 in 3, 7, 9, 9, 2 samples to G = 7 to 10 in 1, 6, 13, 10. The
authors' groupings came from their Fortran program (`MC1.exe`), which may have
searched harder.

## What this means

- The published G0 = 3 result reflects a penalty about twice as large as the one
  (S20) and footnote 12 describe. With the penalty as described, the BIC chooses
  G = 5 or 6 in this design; with the true error variance, it chooses G = 4 most
  often.
- So the claim that the BIC recovers G0 = 3 in 98% of samples depends on the
  inflated penalty. The appendix's footnote 27, which reports that a steeper
  penalty chose G = 1 in every simulation, is consistent with the choice being
  sensitive to the penalty's scale.
- This is not a translation error. Both translations reproduce the published
  table when they do what `BIC_G3_MC.m` does.
- We cannot tell from the code whether loading the G0 = 10 file was intended. The
  file names and the structure of the two scripts suggest that `BIC_G3_MC.m` was
  adapted from `BIC_G10_MC.m` and kept its G = 15 file, but that is an inference.

## Two smaller things in `BIC_G3_MC.m`

Neither changes the results above.

- In the loop over G = 7 down to 1, `gisum` (the group sizes) is not recomputed;
  it keeps its value from the G = 15 step. It is used only to treat one-member
  groups differently, so it matters only if a group has a single unit.
- The script compares G = 1 to 7. Our first runs compared G = 1 to 6; the
  scripts now use G = 1 to 7 for G0 = 3 and G = 3 to 15 for G0 = 10, as the
  original scripts do.

## How to run either version

- Python: `python scripts/04b_bic_tableS8.py 3` (or `10`) gives both readings in a
  few minutes. In `scripts/04_montecarlo.py`, set `RUN_BIC = True` and
  `BIC_SIGMA2 = 'own'` (the default, following the text) or `'g0_10'` (following
  `BIC_G3_MC.m`).
- R: in `scripts/04_montecarlo.R`, set `run_bic <- TRUE` and
  `bic_sigma2 <- "own"` (default) or `"g0_10"`.

## Reference

Bonhomme, S., and E. Manresa (2015). Grouped patterns of heterogeneity in panel
data. *Econometrica*, 83(3), 1147-1184. Supplementary Appendix and replication
files: https://sites.google.com/site/stephanebonhommeresearch/
