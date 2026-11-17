'''04d_authors_groupings.py -- Table S.VIII with G0 = 10, on the authors' own simulated samples and groupings.

Our BIC choices for G0 = 10 centre about half a group below the published ones (NOTE_BIC_tableS8.md). The note
put this down to the authors' Fortran search (MC1.exe) finding lower objective values than our Algorithm 1 and
VNS runs, which was an inference. This script settles it with the replication files BIC_G10_MC.m reads:
inputMCXG10_BIC.txt and inputMCYG10_BIC.txt (the simulated samples; 1,000 of N = 90, T = 7, K = 2, of which the
script uses the first 100) and assignmentG010G3.txt ... assignmentG010G15.txt (the authors' groupings of every
sample for G = 3..15). For each of the first R samples and each G:
  Q_authors   the GFE objective of the authors' grouping (theta and alpha concentrated out, as BIC_G10_MC.m does)
  Q_alg1      our Algorithm 1 with N_STARTS random starts
  Q_local     our local search (single-unit moves to convergence) started from the authors' grouping
and the BIC choice of G (S20, sigma2-hat from G = 15 averaged over the samples, as BIC_G10_MC.m does) from each.

Usage: python scripts/04d_authors_groupings.py <dir with the replication files> [R] [N_STARTS]
Output: output/tableS8_authors_groupings.csv (one row per sample and G), output/tableS8_authors_choices.csv

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import ROOT  # noqa: E402
from gfe.core import gfe_panel, gfe_objective, gfe_alg1, gfe_local_search  # noqa: E402

SRC = Path(sys.argv[1])
R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_STARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
N, T, K, GS = 90, 7, 2, range(3, 16)


def find(name):
    hits = sorted(SRC.glob(f'*{name}'))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


X = np.loadtxt(find('inputMCXG10_BIC.txt'))
Y = np.loadtxt(find('inputMCYG10_BIC.txt'))
Y = Y[:, 0] if Y.ndim == 2 else Y
A = {G: np.loadtxt(find(f'assignmentG010G{G}.txt')).astype(int) for G in GS}
rows = []
t0 = time.time()
for r in range(R):
    sl = slice(r * N * T, (r + 1) * N * T)
    panel = gfe_panel(Y[sl], X[sl], N=N, T=T)
    for G in GS:
        g_auth = A[G][r] - 1
        q_auth = gfe_objective(panel, g_auth, G)
        loc = gfe_local_search(panel, g_auth.copy(), G)
        q_loc = loc['objective'] if isinstance(loc, dict) else float(loc.objective)
        fit = gfe_alg1(panel, G, n_starts=N_STARTS, seed=1_000_000 + 100 * r + G)
        rows.append(dict(sample=r + 1, G=G, groups_used_authors=len(np.unique(g_auth)), Q_authors=q_auth,
                         Q_local_from_authors=q_loc, Q_alg1=float(fit.objective)))
    if (r + 1) % 10 == 0:
        print(f'{r + 1} samples, {time.time() - t0:.0f}s', flush=True)
D = pd.DataFrame(rows)
D['Q_best'] = D[['Q_authors', 'Q_local_from_authors', 'Q_alg1']].min(axis=1)
out = ROOT / 'output'
D.to_csv(out / 'tableS8_authors_groupings.csv', index=False)


def bic_choice(col):
    s2 = (D.loc[D.G == 15, col] / (N * T - 15 * T - N - K)).mean()
    b = D[col] / (N * T) + s2 * np.log(N * T) * (D.G * T + N + K) / (N * T)
    ch = D.assign(b=b).loc[lambda d: d.groupby('sample')['b'].idxmin(), ['sample', 'G']]
    return ch.G.value_counts().sort_index(), s2


summ = []
for col in ['Q_authors', 'Q_alg1', 'Q_local_from_authors', 'Q_best']:
    vc, s2 = bic_choice(col)
    summ.append(dict(objective=col, sigma2_hat=s2, **{f'G{g}': int(vc.get(g, 0)) for g in GS}))
S = pd.DataFrame(summ)
S.to_csv(out / 'tableS8_authors_choices.csv', index=False)
D['alg1_minus_authors'] = D.Q_alg1 - D.Q_authors
print(D.groupby('G')[['Q_authors', 'Q_alg1', 'Q_local_from_authors', 'alg1_minus_authors']].mean().round(4))
print('share of samples where our Algorithm 1 is lower than the authors (by G):')
print(D.assign(lower=D.alg1_minus_authors < -1e-8, higher=D.alg1_minus_authors > 1e-8).groupby('G')[['lower', 'higher']].mean().round(2))
print(S.to_string(index=False))
print('published: G = 8, 9, 10, 11 in 10, 42, 42, 6 of 100')
