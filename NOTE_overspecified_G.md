# What does choosing too many groups cost? (An extension, not part of the replication)

Eliot R. Weinstein · eliotw@uchicago.edu · erweinstein.com
7 November 2026

`NOTE_BIC_overfit.md` explains why the BIC picks G = 4 rather than 3 in the
Table S.VIII design (G0 = 3), even with the true error variance: an extra group
can split a true group along the direction its errors happen to point, and the
penalty does not count that gain. A natural question is whether over-fitting
matters for the common coefficients θ, which are usually the target.

Liu, Shang, Zhang and Zhou (2020, *Journal of Econometrics* 215(2), 574–590)
suggest it may not, for large N and T. In panels where the *coefficients* differ
by group, they show that with more groups than the truth the estimates stay
uniformly consistent, because true groups may be split but units from different
groups are not mixed. Their model is not the GFE model (group-specific time
effects), and their result is asymptotic, so we checked the Table S.VIII design
directly.

**Design** (`scripts/07_overspecified_G.py`). The G0 = 3 Monte Carlo DGP
(published G = 3 grouping and estimates, iid errors with the GFE residual
variance), N = 90, T = 7, 100 samples. In each sample, GFE was fitted with
G = 3, 4, 5 and 6 (Algorithm 1, 100 starts). "Purity" is the share of units
whose estimated group's majority true group is their own (1 = no mixing). True
θ = (0.406, 0.089).

| G | bias θ₁ | SD θ₁ | coverage θ₁ (large-T SE) | bias θ₂ | SD θ₂ | coverage θ₂ | mean purity |
|---|---|---|---|---|---|---|---|
| 3 (true) | −0.006 | 0.046 | 87% | 0.008 | 0.011 | 85% | 0.91 |
| 4 | −0.004 | 0.055 | 76% | 0.010 | 0.016 | 72% | 0.89 |
| 5 | 0.010 | 0.079 | 76% | 0.010 | 0.019 | 69% | 0.87 |
| 6 | 0.006 | 0.086 | 58% | 0.009 | 0.022 | 71% | 0.87 |

(Large-T standard errors averaged 0.034-0.035 for θ₁ and 0.009-0.010 for θ₂ at
every G.)

**What it shows.**

1. **Over-fitting does not bias θ here** (all biases under a quarter of an SD).
2. **It does make θ noisier.** The SD of θ̂₁ is 1.2, 1.7 and 1.9 times the
   true-G value at G = 4, 5, 6; for θ̂₂ it is 1.4-2.0 times. Each extra group adds
   T = 7 group-time effects fitted to noise.
3. **The reported standard errors do not see it.** They stay at the true-G
   size, so 95% intervals cover 58-76% (θ₁) and 69-72% (θ₂). Even at the true
   G they cover only 85-87%, the familiar shortfall of the large-T variance with
   T = 7 (the supplement's fixed-T variance, `gfe_vcov_fixedT()` here, is meant
   for short panels; it was not used in this check).
4. **Groups are mixed a little, not only split.** Purity falls from 0.91 at the
   true G (some units are misclassified even then, with T = 7) to 0.87-0.89.

So in this short panel, choosing G = 4 because of the BIC's bias is not
harmless for θ: at G = 4 it costs a fifth (θ₁) to a half (θ₂) more in SD and
11-13 points of coverage. It is a reason to report θ for neighbouring
values of G rather than for the BIC's choice alone. Liu et al.'s reassurance
is asymptotic and for a different model; it is not contradicted here, but at
T = 7 it does not yet apply.

**Caveats.** One design, one N and T; groupings from Algorithm 1 with 100
starts, which may not reach the global optimum for G = 5-6 (VNS or more starts
could lower the objective and change the SDs somewhat); iid errors only.
