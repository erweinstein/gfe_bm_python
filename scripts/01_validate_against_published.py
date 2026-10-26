'''scripts/01_validate_against_published.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184.

Checks this Python translation against the numbers published in the paper's
Supplementary Appendix:

  Part A. Given the authors' optimal groupings (hard-coded in their MATLAB
          files), reproduce Table S.XI: objective, BIC, coefficients, large-T
          standard errors, and fixed-T standard errors, G = 1..15; the
          fixed-effects row; and the "True" values of Table S.III.
  Part B. Starting from random values, check that Algorithm 1 and Algorithm 2
          (VNS) find the same optima (Tables S.I and S.II), and recover the
          authors' groupings. (A few minutes; set RUN_SEARCH = False to skip.)
  Part C. Table S.XIV (GFE in deviations from country means) and Table S.XV
          (first-differenced IV with estimated groups).

Output: printed comparisons and output/validation_report.csv. Published values
are reported to 3 decimals. They appear to have been rounded from 4-decimal
MATLAB output (e.g. 0.406464 -> 0.4065 -> 0.407), so "match" means agreement
after rounding to 3 decimals, directly or via 4.

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
from gfe import (ROOT, DATA, load_bm_panel, load_bm_published, gfe, gfe_vns,  # noqa: E402
                 gfe_alg1, gfe_bic, gfe_vcov_largeT, gfe_se_theta, gfe_delta,
                 gfe_vcov_fixedT, gfe_bandwidth, gfe_within, gfe_fd_iv, match_groups)

RUN_SEARCH = True        # Part B: random-start searches (a few minutes)
SEARCH_G = [2, 3, 4, 5]  # add 10 to reproduce Table S.II (slower)
SEED = 20150501

report = []


def round_hu(x, d):
    '''Round half away from zero on the decimal representation (as when
    copying MATLAB's 4-decimal display into a table by hand). Python's
    built-in round() does banker's rounding, which is not what we want here.'''
    return np.sign(x) * np.floor(np.abs(x) * 10 ** d + 0.5 + 1e-9) / 10 ** d


def add(table, item, published, ours, digits=3):
    match = (abs(round_hu(ours, digits) - published) < 1e-9 or
             abs(round_hu(round_hu(ours, digits + 1), digits) - published) < 1e-9)
    report.append({'table': table, 'item': item, 'published': published,
                   'this_code': round(float(ours), 6), 'match': bool(match)})


def cum(th):
    return th[1] / (1 - th[0])


def cum_grad(th):
    return np.array([th[1] / (1 - th[0]) ** 2, 1 / (1 - th[0])])


# Published Table S.XI (G = 1..15).
S11 = pd.DataFrame({
    'G': range(1, 16),
    'obj': [24.301, 19.847, 16.599, 14.319, 12.593, 11.132, 10.059, 9.251, 8.426, 7.749, 7.218, 6.809, 6.391, 5.996, 5.664],
    'bic': [0.052, 0.046, 0.042, 0.039, 0.037, 0.036, 0.035, 0.035, 0.034, 0.034, 0.034, 0.034, 0.035, 0.035, 0.035],
    'th1': [0.665, 0.601, 0.407, 0.302, 0.255, 0.465, 0.403, 0.333, 0.312, 0.277, 0.293, 0.304, 0.236, 0.237, 0.244],
    'th2': [0.083, 0.061, 0.089, 0.082, 0.079, 0.064, 0.065, 0.070, 0.069, 0.075, 0.073, 0.074, 0.072, 0.071, 0.071],
    'cum': [0.247, 0.152, 0.151, 0.118, 0.107, 0.119, 0.108, 0.104, 0.101, 0.104, 0.104, 0.107, 0.094, 0.094, 0.094],
    'lt1': [0.049, 0.041, 0.052, 0.054, 0.050, 0.043, 0.043, 0.044, 0.045, 0.049, 0.042, 0.044, 0.040, 0.042, 0.043],
    'lt2': [0.014, 0.011, 0.011, 0.009, 0.010, 0.007, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.009, 0.009, 0.009],
    'ltc': [0.018, 0.021, 0.013, 0.011, 0.009, 0.011, 0.011, 0.010, 0.010, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009],
    'ft1': [np.nan, 0.061, 0.083, 0.108, 0.088, 0.054, 0.074, 0.085, 0.072, 0.062, 0.062, 0.054, 0.046, 0.047, 0.046],
    'ft2': [np.nan, 0.013, 0.015, 0.012, 0.012, 0.008, 0.013, 0.012, 0.010, 0.010, 0.012, 0.009, 0.010, 0.010, 0.010],
    'ftc': [np.nan, 0.030, 0.022, 0.021, 0.014, 0.014, 0.019, 0.014, 0.011, 0.011, 0.013, 0.010, 0.010, 0.010, 0.010],
}).set_index('G')


def main():
    (ROOT / 'output').mkdir(exist_ok=True)
    p = load_bm_panel()
    pub = load_bm_published()['groupings']

    # -------------------------------------------------------------------------
    # Part A. Table S.XI given the published groupings
    # -------------------------------------------------------------------------
    fits = {G: gfe(p, G, groups=None if G == 1 else pub[f'G{G}']) for G in range(1, 16)}
    bic = gfe_bic(list(fits.values())).set_index('G')
    for G, f in fits.items():
        V = gfe_vcov_largeT(f)
        se = gfe_se_theta(f, V)
        row = S11.loc[G]
        add('S.XI', f'G={G:2d} objective', row.obj, f.objective)
        add('S.XI', f'G={G:2d} BIC', row.bic, bic.loc[G, 'BIC'])
        add('S.XI', f'G={G:2d} theta1 (lag dem)', row.th1, f.theta[0])
        add('S.XI', f'G={G:2d} theta2 (income)', row.th2, f.theta[1])
        add('S.XI', f'G={G:2d} cumulative income', row.cum, cum(f.theta))
        add('S.XI', f'G={G:2d} large-T se theta1', row.lt1, se[0])
        add('S.XI', f'G={G:2d} large-T se theta2', row.lt2, se[1])
        add('S.XI', f'G={G:2d} large-T se cumulative', row.ltc,
            np.sqrt(gfe_delta(f, V, cum_grad(f.theta))))
        if G >= 2:
            Vf = gfe_vcov_fixedT(f, gfe_bandwidth(f, snap_to_grid=False))
            sf = gfe_se_theta(f, Vf)
            add('S.XI', f'G={G:2d} fixed-T se theta1', row.ft1, sf[0])
            add('S.XI', f'G={G:2d} fixed-T se theta2', row.ft2, sf[1])
            add('S.XI', f'G={G:2d} fixed-T se cumulative', row.ftc,
                np.sqrt(gfe_delta(f, Vf, cum_grad(f.theta))))
    print('BIC selects G =', bic.attrs['G_BIC'], '(the paper: G = 10)')

    # Fixed-effects row: two-way fixed effects = GFE with G = 1 in deviations
    # from country means.
    ffe = gfe(gfe_within(p), 1)
    add('S.XI', 'FE objective', 17.517, ffe.objective)
    add('S.XI', 'FE theta1', 0.284, ffe.theta[0])
    add('S.XI', 'FE theta2', -0.031, ffe.theta[1])
    add('S.XI', 'FE cumulative income', -0.044, cum(ffe.theta))

    # Table S.III, "True" columns: GFE estimates used as the Monte Carlo DGP.
    true_vals = {3: (0.407, 0.089, 0.151), 5: (0.255, 0.079, 0.107), 10: (0.277, 0.075, 0.104)}
    for G, tru in true_vals.items():
        f = fits[G]
        add('S.III', f'G={G:2d} true theta1', tru[0], f.theta[0])
        add('S.III', f'G={G:2d} true theta2', tru[1], f.theta[1])
        add('S.III', f'G={G:2d} true cumulative', tru[2], cum(f.theta))

    # -------------------------------------------------------------------------
    # Part B. Do our searches find the published optima?
    # -------------------------------------------------------------------------
    if RUN_SEARCH:
        rng = np.random.default_rng(SEED)
        for G in SEARCH_G:
            t0 = time.time()
            v = gfe_vns(p, G, n_starts=10, neighmax=10, itermax=10, seed=rng)
            m = match_groups(v.groups, pub[f'G{G}'], G)
            n_reach = int(np.sum(np.abs(v.objectives_by_start - v.objective) < 1e-8))
            print(f'G={G:2d} VNS(10;10;10): objective {v.objective:.6f} (published grouping: '
                  f'{fits[G].objective:.6f}), {n_reach}/10 starts reach it, '
                  f'{round(m["misclassification"] * p.N)} units classified differently, '
                  f'{time.time() - t0:.0f}s')
            tab = 'S.II' if G == 10 else ('S.I' if G <= 3 else 'S.XI')
            add(tab, f'G={G:2d} objective from VNS search', S11.loc[G, 'obj'], v.objective)
            if G <= 3:
                a1 = gfe_alg1(p, G, n_starts=1000, seed=rng)
                add('S.I', f'G={G:2d} objective from Algorithm 1 (1,000 starts)',
                    S11.loc[G, 'obj'], a1.objective)

    # -------------------------------------------------------------------------
    # Part C. Extensions: Tables S.XIV and S.XV
    # -------------------------------------------------------------------------
    S14 = pd.DataFrame({'G': [2, 3, 4, 5], 'obj': [12.859, 10.400, 9.221, 8.174],
                        'th1': [0.061, -0.033, -0.072, -0.093], 'th2': [-0.038, -0.035, 0.045, -0.013],
                        'se1': [0.049, 0.043, 0.046, 0.042], 'se2': [0.027, 0.027, 0.027, 0.026]})
    pw = gfe_within(p)
    for _, row in S14.iterrows():
        G = int(row.G)
        f = gfe(pw, G, groups=pub[f'dev_G{G}'])
        se = gfe_se_theta(f, gfe_vcov_largeT(f))
        add('S.XIV', f'G={G} objective', row.obj, f.objective)
        add('S.XIV', f'G={G} theta1', row.th1, f.theta[0])
        add('S.XIV', f'G={G} theta2', row.th2, f.theta[1])
        add('S.XIV', f'G={G} large-T se theta1', row.se1, se[0])
        add('S.XIV', f'G={G} large-T se theta2', row.se2, se[1])

    S15 = pd.DataFrame({'G': [1, 2, 3, 4, 5], 'th1': [0.472, 0.338, 0.202, 0.065, 0.085],
                        'th2': [-0.075, -0.063, -0.064, -0.037, -0.078],
                        'se1': [0.131, 0.120, 0.094, 0.089, 0.089], 'se2': [0.068, 0.062, 0.055, 0.049, 0.049]})
    raw = pd.read_csv(DATA / 'ajry_5yearpanel_extract.csv')
    bal = p.data
    for _, row in S15.iterrows():
        G = int(row.G)
        grp = np.zeros(p.N, dtype=int) if G == 1 else pub[f'dev_G{G}'].to_numpy()
        df = pd.DataFrame({'code': bal['code'], 'year': bal['year'],
                           'group': grp[bal['unit'].to_numpy() - 1]})   # 'unit' is 1-based in the CSV
        df = df.merge(raw[['code', 'code_numeric', 'year', 'fhpolrigaug', 'lrgdpch']],
                      on=['code', 'year'])
        r = gfe_fd_iv(df, id='code_numeric', year='year', y='fhpolrigaug', x='lrgdpch', group='group')
        add('S.XV', f'G={G} IV theta1', row.th1, r['coef'][0])
        add('S.XV', f'G={G} IV theta2', row.th2, r['coef'][1])
        add('S.XV', f'G={G} IV se theta1', row.se1, r['se'][0])
        add('S.XV', f'G={G} IV se theta2', row.se2, r['se'][1])

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    rep = pd.DataFrame(report)
    rep.to_csv(ROOT / 'output' / 'validation_report.csv', index=False)
    print('\nAgreement with the published values, by table:')
    print(rep.groupby('table')['match'].agg(lambda x: f'{x.sum()} of {len(x)}'))
    print('\nItems that differ in the third decimal:')
    print(rep[~rep['match']].to_string(index=False))
    print('\nNotes:\n'
          ' - Fixed-T standard errors use the rule-of-thumb bandwidth without snapping it to the\n'
          '   grid linspace(0.05, 0.15, 10) used in GFE_estimates.m; the unsnapped rule reproduces\n'
          '   Table S.XI to within 0.001 (try gfe_bandwidth(fit, snap_to_grid=True)).\n'
          ' - G = 1: the published large-T s.e. of theta1 is 0.049; the formula in\n'
          '   GFE_estimates.m gives 0.0483 (Stata small-sample factors would give 0.0489).')


if __name__ == '__main__':
    main()
