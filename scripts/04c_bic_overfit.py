'''04c_bic_overfit.py -- Why the BIC overstates G in the Table S.VIII design even with
the true error variance (a question raised by Stephane Bonhomme, November 2026).

With G0 = 3 (N = 90, T = 7) and the true sigma2, the BIC (S20) chose G = 4 in 76 of 100
samples (NOTE_BIC_tableS8.md). The BIC charges sigma2 ln(NT) T for each extra group:
the T new group-time effects. But an extra group also brings a new *assignment*: the
units can be re-sorted so that the new group picks up a subset of one true group whose
errors happen to point the same way. That gain is not a fixed number of parameters.

Heuristic. Splitting a true group of n units (pure noise around its mean path, in T
dimensions) into two by k-means reduces the sum of squares by about
(2/pi) n lambda_1, where lambda_1 ~ sigma2 (1 + sqrt(T/n))^2 is the largest eigenvalue
of the group's error covariance (the half-normal mean of a projection onto the leading
direction, times the Marchenko-Pastur edge). That is O(n sigma2), against the BIC's
sigma2 T ln(NT): with T fixed, the gain grows linearly in N and the penalty only
logarithmically, so the BIC overstates G more often as N grows; with N fixed, the
penalty grows linearly in T and the gain only like (1 + sqrt(T/n))^2, so it overstates
G less often as T grows. For the design, n = 38 (the largest true group), T = 7:
(2/pi) 38 (1 + sqrt(7/38))^2 ~ 49 against T ln(NT) ~ 45.

Part A: the design itself (democracy-income model, true sigma2): distribution of
  (Q(3) - Q(4)) / sigma2 against the penalty T ln(NT).
Part B: the mechanism without covariates: y_i = alpha_{g_i} + e_i in R^T, with the
  design's alpha (3 x 7, tiled in time for T = 14, 28), its group shares (24/28/38 of 90)
  and sigma2, for N = 90, 180, 360, 720 and T = 7, 14, 28. Groups are fit by k-means
  (scikit-learn; with no covariates the GFE objective is exactly the k-means objective),
  and G is chosen from 1..8 by the BIC with the true sigma2.

Usage: python scripts/04c_bic_overfit.py [R]
Output: output/bic_overfit_design.csv, output/bic_overfit_grid.csv

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util  # noqa: E402
from gfe import ROOT, load_bm_panel, load_bm_published, gfe  # noqa: E402

_spec = importlib.util.spec_from_file_location('mc', Path(__file__).resolve().parent / '04_montecarlo.py')
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
p = load_bm_panel()
pub = load_bm_published()['groupings']
N0, T0, K = p.N, p.T, p.K
truth = gfe(p, 3, groups=pub['G3'])
s2 = truth.objective / (N0 * T0 - 3 * T0 - N0 - K)
shares = np.bincount(truth.groups) / N0
alpha = truth.alpha                                   # T0 x 3


def split_gain(n, T):
    return 2 / np.pi * n * (1 + np.sqrt(T / n)) ** 2


# ---- Part A: the design --------------------------------------------------------------------
rng = np.random.default_rng(20261103)
rows = []
t0 = time.time()
for r in range(R):
    ps = mc.simulate_panel(truth, 'iid', s2, rng)
    q_true = gfe(ps, 3, groups=truth.groups).objective
    q = {G: gfe(ps, G, algorithm='alg1', n_starts=100, seed=rng).objective for G in (3, 4, 5)}
    rows.append({'r': r, 'Q3_true_groups': q_true, 'Q3': q[3], 'Q4': q[4], 'Q5': q[5]})
    if (r + 1) % 25 == 0:
        print(f'design {r + 1}/{R} {time.time() - t0:.0f}s', flush=True)
A = pd.DataFrame(rows)
A['gain_3to4'] = (A.Q3 - A.Q4) / s2
A['gain_4to5'] = (A.Q4 - A.Q5) / s2
A['overfit_at_G3'] = (A.Q3_true_groups - A.Q3) / s2
pen = T0 * np.log(N0 * T0)
A.to_csv(ROOT / 'output' / 'bic_overfit_design.csv', index=False)
print(f'\nDesign (N = {N0}, T = {T0}, true sigma2 = {s2:.4f}); penalty per extra group '
      f'T ln(NT) = {pen:.1f} (in units of sigma2)')
print(f'  (Q(3) - Q(4)) / sigma2: mean {A.gain_3to4.mean():.1f}, median {A.gain_3to4.median():.1f}, '
      f'share above the penalty {np.mean(A.gain_3to4 > pen):.2f}')
print(f'  (Q(4) - Q(5)) / sigma2: mean {A.gain_4to5.mean():.1f}; share above {np.mean(A.gain_4to5 > pen):.2f}')
print(f'  (Q at the true groups - Q(3)) / sigma2: mean {A.overfit_at_G3.mean():.1f}')
print(f'  heuristic split gain for the largest group (n = {int(shares.max() * N0)}): '
      f'{split_gain(shares.max() * N0, T0):.1f}')

# ---- Part B: the mechanism, varying N and T ------------------------------------------------
rows = []
for T in (7, 14, 28):
    a = np.tile(alpha, (T // T0, 1))                     # T x 3
    for N in (90, 180, 360, 720):
        n_g = np.round(shares * N).astype(int)
        n_g[-1] = N - n_g[:-1].sum()
        g = np.repeat(np.arange(3), n_g)
        pen = T * np.log(N * T)
        choice, gain = [], []
        for r in range(R):
            Y = a[:, g].T + rng.normal(scale=np.sqrt(s2), size=(N, T))
            Q = np.array([KMeans(G, n_init=20, random_state=int(rng.integers(1 << 31))).fit(Y).inertia_
                          for G in range(1, 9)])
            bic = Q / (N * T) + s2 * np.log(N * T) * np.arange(1, 9) * T / (N * T)
            choice.append(int(bic.argmin()) + 1)
            gain.append((Q[2] - Q[3]) / s2)
        choice = np.array(choice)
        rows.append({'T': T, 'N': N, 'penalty': round(pen, 1), 'mean_gain_3to4': round(np.mean(gain), 1),
                     'heuristic_gain': round(split_gain(n_g.max(), T), 1),
                     **{f'G{k}': int(np.sum(choice == k)) for k in range(1, 9)}})
        print(rows[-1], flush=True)
B = pd.DataFrame(rows)
B.to_csv(ROOT / 'output' / 'bic_overfit_grid.csv', index=False)
print(B.to_string(index=False))
