'''07_overspecified_G.py -- What does choosing too many groups cost? (An extension, not
part of the replication.)

Liu, Shang, Zhang and Zhou (2020, Journal of Econometrics 215(2), 574-590) show, for
panels whose coefficients differ by group, that with more groups than the truth the
estimates stay uniformly consistent: true groups may be split, but units from
different groups are not mixed. NOTE_BIC_overfit.md found that the BIC prefers G = 4
when G0 = 3 even with the true error variance, because an extra group can split a true
group. If over-fitting only splits groups, the common coefficients theta should be
little affected, at some cost in precision. This script checks that in the Table
S.VIII design.

Design: the G0 = 3 DGP of the Monte Carlo (published G = 3 grouping and estimates,
iid errors with the GFE residual variance), N = 90, T = 7. For each of 100 samples,
GFE with G = 3, 4, 5, 6 (Algorithm 1, 100 starts). Reported for each G: bias, SD and
RMSE of theta_1 (lagged democracy) and theta_2 (lagged income); coverage of 95%
intervals with the large-T clustered variance; and "purity", the share of units whose
estimated group's majority true group is their own (1 = no mixing of true groups).

Usage: python scripts/07_overspecified_G.py [R] [N_STARTS]
Output: output/overspecified_G.csv

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util  # noqa: E402
from gfe import ROOT, load_bm_panel, load_bm_published, gfe  # noqa: E402
from gfe.inference import gfe_vcov_largeT, gfe_se_theta  # noqa: E402

_spec = importlib.util.spec_from_file_location('mc', Path(__file__).resolve().parent / '04_montecarlo.py')
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
GS = (3, 4, 5, 6)

p = load_bm_panel()
pub = load_bm_published()['groupings']
N, T, K = p.N, p.T, p.K
truth = gfe(p, 3, groups=pub['G3'])
s2_true = truth.objective / (N * T - 3 * T - N - K)
rng = np.random.default_rng(20261107)
rows = []
t0 = time.time()
for r in range(R):
    ps = mc.simulate_panel(truth, 'iid', s2_true, rng)
    for G in GS:
        fit = gfe(ps, G, algorithm='alg1', n_starts=N_STARTS, seed=rng)
        se = gfe_se_theta(fit, gfe_vcov_largeT(fit))
        # purity: for each estimated group, its majority true group; share of units matching it
        maj = {g: np.bincount(truth.groups[fit.groups == g], minlength=3).argmax()
               for g in np.unique(fit.groups)}
        purity = np.mean([truth.groups[i] == maj[fit.groups[i]] for i in range(N)])
        rows.append({'rep': r, 'G': G, 'theta1': fit.theta[0], 'theta2': fit.theta[1],
                     'se1': se[0], 'se2': se[1], 'purity': purity, 'objective': fit.objective})
    if (r + 1) % 10 == 0:
        print(f'{r + 1}/{R}  {time.time() - t0:.0f}s', flush=True)
d = pd.DataFrame(rows)
out = []
for G, g in d.groupby('G'):
    row = {'G': G}
    for k in (1, 2):
        th = truth.theta[k - 1]
        e = g[f'theta{k}'] - th
        row.update({f'bias{k}': e.mean(), f'sd{k}': g[f'theta{k}'].std(), f'rmse{k}': np.sqrt((e ** 2).mean()),
                    f'cover{k}': np.mean(np.abs(e) <= 1.96 * g[f'se{k}']), f'mean_se{k}': g[f'se{k}'].mean()})
    row.update({'purity_mean': g['purity'].mean(), 'purity_all_pure': np.mean(g['purity'] == 1.0)})
    out.append(row)
tab = pd.DataFrame(out)
print(f'\nG0 = 3, true theta = ({truth.theta[0]:.3f}, {truth.theta[1]:.3f}), {R} samples, '
      f'Algorithm 1 with {N_STARTS} starts:')
print(tab.round(4).to_string(index=False))
tab.to_csv(ROOT / 'output' / 'overspecified_G.csv', index=False)
d.to_csv(ROOT / 'output' / 'overspecified_G_draws.csv', index=False)
