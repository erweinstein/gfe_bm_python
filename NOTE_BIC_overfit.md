# Note: why the BIC overstates the number of groups in the Table S.VIII design

Eliot R. Weinstein, 3 November 2026 (follows [NOTE_BIC_tableS8.md](NOTE_BIC_tableS8.md))

## The question

In the G0 = 3 design of the Supplementary Appendix (N = 90, T = 7), the BIC (S20)
chose too many groups even when it was given the **true** error variance: G = 4 in
76 of 100 samples, G = 3 in only 15 (NOTE_BIC_tableS8.md). So the overstatement is
not only a matter of estimating sigma2. Why?

## A mechanism

The penalty in (S20) charges sigma2 ln(NT) for each parameter, so an extra group
costs sigma2 T ln(NT) (the T new group-time effects). But an extra group also
brings a new *assignment*: the units can be re-sorted so that the new group takes
the members of one true group whose errors happen to point the same way. That gain
is not a fixed number of parameters, and it grows with the size of the group.

A classical approximation makes this concrete. Splitting a normal sample of n
points into two clusters by least squares reduces the within-cluster sum of
squares by about (2/pi) times n times the variance along the direction of the
split: in one dimension this is the half-normal mean (Hartigan 1978), and for a
p-dimensional normal with a largest variance sigma_1^2, Hartigan (1978, section 7)
conjectures that the optimal split is along that direction, with the sum of
squares reduced by about (2/pi) n sigma_1^2. Our groups are spherical (equal
error variance in every period), the case Hartigan leaves open; there the sample
picks the direction in which its own variance is largest, so we replace sigma_1^2
by the largest sample eigenvalue, about sigma2 (1 + sqrt(T/n))^2 (the upper edge of
the Marchenko–Pastur law). This step is a heuristic, checked by simulation below.
So splitting a true group of n units
lowers the objective by about

    gain(n, T) ~ (2/pi) n sigma2 (1 + sqrt(T/n))^2,

against the BIC's charge of sigma2 T ln(NT). In the design, the largest true group
has n = 38 units and T = 7: the gain is ~49.4 sigma2 and the charge ~45.1 sigma2, so
the BIC adds a group in most samples. More generally, with T fixed the gain grows
linearly in N and the penalty only like ln N, so the overstatement gets worse as N
grows; with N fixed the penalty grows linearly in T and the gain only like
(1 + sqrt(T/n))^2, so it disappears as T grows.

## Checks (`scripts/04c_bic_overfit.py`, 100 samples per cell)

**The design itself** (democracy–income model, true sigma2 = 0.0321, Algorithm 1
with 100 starts): (Q(3) − Q(4))/sigma2 had mean 50.1 and median 49.5, against the
predicted 49.4 and the penalty 45.1; it exceeded the penalty in 82 of 100 samples.
The next split, (Q(4) − Q(5))/sigma2, averaged 39.3 and exceeded the penalty in 5,
which is why the BIC mostly stops at 4. (The estimated G = 3 grouping also fits
22 sigma2 better, on average, than the true grouping.)

**The mechanism without covariates**: y_i = alpha_{g_i} + e_i in R^T, with the
design's alpha (tiled in time for T = 14 and 28), group shares (24/28/38 of 90) and
sigma2; k-means (the GFE objective without covariates); G chosen from 1 to 8 by the
BIC with the true sigma2.

| T | N | Penalty T ln(NT) | Simulated (Q(3) − Q(4))/sigma2 | Predicted gain | BIC chose G = 3 | Most common choice |
|---|---|---|---|---|---|---|
| 7 | 90 | 45.1 | 49.0 | 49.4 | 26 | 4 (60) |
| 7 | 180 | 50.0 | 82.4 | 82.2 | 0 | 6 (51) |
| 7 | 360 | 54.8 | 141.8 | 142.8 | 0 | 8 (96)* |
| 7 | 720 | 59.7 | 253.8 | 256.7 | 0 | 8 (100)* |
| 14 | 90 | 99.9 | 63.5 | 62.5 | 100 | 3 |
| 14 | 180 | 109.6 | 99.6 | 98.8 | 89 | 3 |
| 14 | 360 | 119.4 | 165.6 | 164.4 | 0 | 5 (50) |
| 14 | 720 | 129.1 | 286.8 | 285.5 | 0 | 8 (93)* |
| 28 | 90 | 219.3 | 85.4 | 83.5 | 100 | 3 |
| 28 | 180 | 238.7 | 128.3 | 124.9 | 100 | 3 |
| 28 | 360 | 258.1 | 202.3 | 197.7 | 100 | 3 |
| 28 | 720 | 277.5 | 336.7 | 328.8 | 0 | 4 (88) |

\* the largest G considered.

The approximation predicts the simulated gain to within ~3% in every cell. When
the predicted gain for the largest group is below the penalty, the BIC chooses
G0 = 3 in 89–100 of 100 samples; when it is above, in 26 or fewer (none once the
gain is well above).

## What this means

- In this design the BIC's penalty for an extra group is about the same size as
  the fit gained by splitting the largest true group, so the BIC is close to
  indifferent between 3 and 4 groups even with the true sigma2. The inflated sigma2
  in `BIC_G3_MC.m` (about twice the true value) moved the penalty well above the
  gain, which is why it reproduced the published 98%.
- The comparison that matters is between N/G0 (group size) and T ln(NT). The
  criterion works when groups are small relative to T ln(NT), and overstates G when
  they are large, as in many applications with short panels and many units. This
  agrees with Nishi (2025), who reports that the Bonhomme–Manresa BIC overestimates
  the number of groups in finite samples when T is small relative to N and proposes
  modified criteria whose penalty depends on whether N ≤ T.
- A penalty that charges for the assignment as well as for the T parameters
  (something of order N/G sigma2 per extra group, rather than T ln(NT) sigma2)
  would be the natural repair; how to calibrate it is an open question.

## References

- Bonhomme S, Manresa E (2015). Grouped patterns of heterogeneity in panel data.
  *Econometrica* 83(3), 1147–1184 (and Supplementary Appendix).
- Hartigan JA (1978). Asymptotic distributions for clustering criteria. *Annals of
  Statistics* 6(1), 117–131.
- Liu R, Shang Z, Zhang Y, Zhou Q (2020). Identification and estimation in panel
  models with overspecified number of groups. *Journal of Econometrics* 215(2),
  574–590. https://doi.org/10.1016/j.jeconom.2019.09.008
- Nishi M (2025). K-means panel data clustering in the presence of small groups.
  arXiv:2508.15408. https://arxiv.org/abs/2508.15408
