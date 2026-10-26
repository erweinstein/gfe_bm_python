'''scripts/03_exact_algorithm.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184, Supplementary Appendix S.1.2 and Table S.I.

Purpose. The GFE objective is minimized over all G^N groupings, so the
heuristics (Algorithms 1 and 2) come with no guarantee of reaching the global
minimum. Bonhomme and Manresa check them against an EXACT solution computed by
their covariate extension of Brusco's (2006) repetitive branch and bound
(Exact algorithm/Brusco_cov.f90). This script repeats that check:

  1. A small simulated problem, where the exact solution can also be found by
     brute-force enumeration: brute force = branch and bound.
  2. The first N = 30 countries of the application, G = 2, with the data
     rounded to 4 decimals as in the authors' input files inputY.txt and
     inputX.txt. Target (Exact algorithm/RESULTS2.DAT and outputtheta.txt):
     objective 6.1591..., theta = (0.6070, 0.0689). Needs numba to finish in
     reasonable time (about 160 million nodes; under a minute compiled, hours
     in plain Python).
  3. Optionally (RUN_N90 = True) the full sample, N = 90, G = 2: target
     objective 19.846 (Table S.I). This takes a long time.

In each case we also run Algorithm 1 and VNS, and report whether they reach
the exact optimum: that is the comparison made in Table S.I.

Python translation Copyright (C) 2026 Eliot R. Weinstein. Licensed under the
GNU GPL, version 2 or (at your option) any later version; see LICENSE. The
license covers the translation only, not Bonhomme and Manresa's method or
original code.
'''

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import (ROOT, DATA, HAVE_NUMBA, gfe_panel, load_bm_panel, gfe,  # noqa: E402
                 gfe_exact_bb, gfe_bruteforce, match_groups)

RUN_N30 = True       # set False without numba (plain Python would take hours)
RUN_N90 = False      # True: N = 90, G = 2 (hours, even compiled)
SEED = 2015


def compare(p, ex, rng, label):
    '''Exact solution vs Algorithm 1 and VNS on the same panel.'''
    a1 = gfe(p, ex.G, algorithm='alg1', n_starts=1000, seed=rng)
    vn = gfe(p, ex.G, algorithm='vns', n_starts=10, neighmax=10, itermax=10, seed=rng)
    rows = []
    for name, f in (('exact (branch & bound)', ex), ('Algorithm 1 (1,000 starts)', a1),
                    ('Algorithm 2, VNS (10; 10; 10)', vn)):
        rows.append({'sample': label, 'G': ex.G, 'method': name, 'objective': f.objective,
                     'theta1': f.theta[0], 'theta2': f.theta[1],
                     'misclassified_vs_exact':
                         round(match_groups(f.groups, ex.groups, ex.G)['misclassification'] * p.N)})
    return rows


def main():
    rng = np.random.default_rng(SEED)
    print('numba available:', HAVE_NUMBA)
    results = []

    # -------------------------------------------------------------------------
    # 1. A brute-force check on a toy problem (N = 10, T = 4, K = 2, G = 3)
    # -------------------------------------------------------------------------
    N, T, G = 10, 4, 3
    g0 = rng.integers(0, G, size=N)
    a0 = rng.normal(size=(T, G))
    X = rng.normal(size=(N * T, 2))
    y = X @ np.array([0.5, -0.3]) + a0[np.tile(np.arange(T), N), np.repeat(g0, T)] \
        + rng.normal(scale=0.7, size=N * T)
    toy = gfe_panel(y, X, N, T)
    bf = gfe_bruteforce(toy, G)                      # G^(N-1) = 19,683 groupings
    ex = gfe_exact_bb(toy, G, seed=rng)
    print(f'\nToy problem (N = 10, G = 3):\n  brute force     {bf["objective"]:.10f}\n'
          f'  branch & bound  {ex.objective:.10f}   nodes = {ex.nodes}')

    # -------------------------------------------------------------------------
    # 2. N = 30, G = 2 (Table S.I, first row)
    # -------------------------------------------------------------------------
    # The authors' input files hold the data with 4 decimals. With unrounded
    # data the optimum is 6.15958; with the rounded data it is 6.15913, the
    # published value. Either way the optimal grouping is the same.
    if RUN_N30:
        dat = pd.read_csv(DATA / 'bm_final_data.csv').sort_values(['unit', 'year'])
        n30 = 30
        sub = dat.iloc[:n30 * 7]
        p30 = gfe_panel(sub['dem'].round(4).to_numpy(),
                        sub[['lag_dem', 'lag_income']].round(4).to_numpy(),
                        N=n30, T=7, x_names=['lag_dem', 'lag_income'])
        t0 = time.time()
        ex30 = gfe_exact_bb(p30, 2, seed=rng)
        print(f'\nN = 30, G = 2: exact solution in {time.time() - t0:.1f}s, nodes = {ex30.nodes}')
        results += compare(p30, ex30, rng, 'first 30 countries (4-decimal data)')
        print('Published (RESULTS2.DAT / outputtheta.txt): objective 6.1591, theta = (0.6070, 0.0689)')

    # -------------------------------------------------------------------------
    # 3. N = 90, G = 2 (Table S.I): optional
    # -------------------------------------------------------------------------
    if RUN_N90:
        p90 = load_bm_panel()
        ex90 = gfe_exact_bb(p90, 2, seed=rng, verbose=True)
        results += compare(p90, ex90, rng, 'all 90 countries')
        print('Published (Table S.I): 19.846')

    if results:
        res = pd.DataFrame(results)
        print(res.to_string(index=False))
        (ROOT / 'output').mkdir(exist_ok=True)
        res.to_csv(ROOT / 'output' / 'tableS1_exact_vs_heuristics.csv', index=False)


if __name__ == '__main__':
    main()
