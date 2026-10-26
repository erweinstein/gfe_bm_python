'''scripts/06_explaining_the_waves.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184, Supplementary Appendix S.7.4 ("Explaining the estimated grouped
patterns"): Table S.XVII (descriptive statistics by group) and Table S.XVIII
(multinomial logits of group membership).

Follows Explaining_the_waves.do (original code by Stephane Bonhomme and Elena
Manresa), which runs Stata's mlogit on explaining_the_waves_G4.dta, shipped
here as data/explaining_the_waves_G4.csv. The G = 4 groups are those of
Figure 2: 1 high democracy, 2 low democracy (the base category), 3 early
transition, 4 late transition. (These are the paper's 1-based labels, kept as
is: here they are category codes of the dependent variable, not indices.)

Treating estimated groups as data. Section S.5.3 of the Supplementary Appendix
justifies using the GFE group estimates as a dependent variable, and computing
standard errors as if they were known, when N and T are both large:
misclassification then vanishes at an exponential rate.

Variables (Acemoglu, Johnson, Robinson, and Yared, 2008):
    GDP_1500_cons  log GDP per capita in 1500 (footnote 34 of the Appendix)
    indyear        year of independence (displayed per 100 years)
    consfirstaug   constraints on the executive at independence
    lag_dem1970    democracy in 1965 (the lag of the 1970 observation)
    lag_income1970 log GDP per capita in 1965
    education1970  average years of schooling, 1970
    rel_catho80, rel_protmg80  shares of Catholic and Protestant, 1980

Python translation Copyright (C) 2026 Eliot R. Weinstein. Licensed under the
GNU GPL, version 2 or (at your option) any later version; see LICENSE. The
license covers the translation only, not Bonhomme and Manresa's method or
original code.
'''

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import ROOT, DATA  # noqa: E402

GROUP_NAMES = ['1 (High Dem.)', '2 (Low Dem.)', '3 (Early Trans.)', '4 (Late Trans.)']
VARS = {'log GDP p.c. (1500)': 'GDP_1500_cons', 'Independence Year': 'indyear',
        'Constraints': 'consfirstaug', 'Democracy (1965)': 'lag_dem1970',
        'log GDP p.c. (1965)': 'lag_income1970', 'Education (1970)': 'education1970',
        'Share Catholic (1980)': 'rel_catho80', 'Share Protestant (1980)': 'rel_protmg80'}
SPECS = {
    '(1)': ['GDP_1500_cons'],
    '(2)': ['GDP_1500_cons', 'indyear100', 'consfirstaug'],
    '(3)': ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'rel_catho80', 'rel_protmg80'],
    '(4)': ['lag_dem1970', 'lag_income1970'],
    '(5)': ['lag_dem1970', 'education1970'],
    '(6)': ['lag_dem1970', 'lag_income1970', 'education1970'],
    '(7)': ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'education1970'],
    '(8)': ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'lag_dem1970', 'education1970'],
    '(9)': ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'lag_dem1970', 'lag_income1970',
            'education1970'],
    '(10)': ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'lag_dem1970', 'lag_income1970',
             'education1970', 'rel_catho80', 'rel_protmg80'],
}


def mlogit(y, Z, base, tol=1e-12, max_iter=100):
    '''Multinomial logit by Newton-Raphson.

        P(g_i = j | z_i) = exp(z_i' b_j) / sum_k exp(z_i' b_k),  b_base = 0.

    Written out instead of calling statsmodels' MNLogit because the
    log-likelihood is globally concave, so Newton's method converges in a few
    steps from zero, and the score and Hessian give both the usual
    (observed-information, Stata's default) and the robust ("clustered by
    country", one observation per country) variance directly. It also keeps
    the package's dependencies to numpy, scipy, and pandas.

    Returns dict with coef, se, se_robust (k x J DataFrames, columns g1, g3, ...).
    '''
    alt = [j for j in np.unique(y) if j != base]
    Z_df, Z = Z, np.asarray(Z, dtype=float)
    n, k = Z.shape
    J = len(alt)
    Y = np.column_stack([(y == j).astype(float) for j in alt])          # n x J
    B = np.zeros((k, J))
    for _ in range(max_iter):
        E = np.exp(Z @ B)
        P = E / (1 + E.sum(axis=1, keepdims=True))                      # n x J
        score_i = np.hstack([Z * (Y[:, [j]] - P[:, [j]]) for j in range(J)])   # n x kJ
        H = np.zeros((k * J, k * J))                                    # minus the Hessian
        for a in range(J):
            for b in range(J):
                w = P[:, a] * ((a == b) - P[:, b])
                H[a * k:(a + 1) * k, b * k:(b + 1) * k] = (Z * w[:, None]).T @ Z
        step = np.linalg.solve(H, score_i.sum(axis=0))
        # B is stored column by column (one column per alternative), so the
        # stacked step vector reshapes with order='F' (MATLAB/R's column-major order).
        B += step.reshape((k, J), order='F')
        if np.max(np.abs(step)) < tol:
            break
    E = np.exp(Z @ B)
    P = E / (1 + E.sum(axis=1, keepdims=True))
    score_i = np.hstack([Z * (Y[:, [j]] - P[:, [j]]) for j in range(J)])
    V_oim = np.linalg.inv(H)
    V_rob = V_oim @ (score_i.T @ score_i) @ V_oim * n / (n - 1)        # Stata's vce(cluster) factor
    cols = [f'g{j}' for j in alt]
    Z_names = list(Z_df.columns) if hasattr(Z_df, 'columns') else [f'z{i}' for i in range(k)]

    def as_frame(v):
        return pd.DataFrame(v.reshape((k, J), order='F'), index=Z_names, columns=cols)
    return {'coef': pd.DataFrame(B, index=Z_names, columns=cols),
            'se': as_frame(np.sqrt(np.diag(V_oim))),
            'se_robust': as_frame(np.sqrt(np.diag(V_rob))), 'n': n}


def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    w = pd.read_csv(DATA / 'explaining_the_waves_G4.csv')
    w = w[w['assignment'].notna()].copy()                 # the 90 countries of the panel
    w['indyear100'] = w['indyear'] / 100                  # "Independence Year/100"

    # -------------------------------------------------------------------------
    # 1. Table S.XVII: means and standard deviations by group
    # -------------------------------------------------------------------------
    rows = []
    for lab, var in VARS.items():
        grp = w.groupby('assignment')[var]
        rows.append([lab, 'mean'] + list(grp.mean()))
        rows.append([lab, 'sd'] + list(grp.std(ddof=1)))
    rows.append(['Number of observations', 'n'] + list(w['assignment'].value_counts().sort_index()))
    desc = pd.DataFrame(rows, columns=['variable', 'statistic'] + GROUP_NAMES)
    print('Table S.XVII: descriptive statistics by group')
    print(desc.round(3).to_string(index=False))
    desc.to_csv(out / 'tableS17_descriptives.csv', index=False)
    # Every entry agrees with Table S.XVII to the published precision, except
    # two for log GDP p.c. in 1500: the Group 1 s.d. (0.290 here, 0.300
    # published) and the Group 3 mean (6.478 here, 6.49 published).

    # -------------------------------------------------------------------------
    # 2. Table S.XVIII: columns (1)-(10), base category = Group 2
    # -------------------------------------------------------------------------
    rows_order = ['GDP_1500_cons', 'indyear100', 'consfirstaug', 'lag_dem1970',
                  'lag_income1970', 'education1970', 'rel_catho80', 'rel_protmg80']
    row_labels = ['log GDP p.c. (1500)', 'Independence Year/100', 'Constraints',
                  'Democracy (1965)', 'log GDP p.c. (1965)', 'Education (1970)',
                  'Share Catholic (1980)', 'Share Protestant (1980)']
    long = []
    fits = {}
    for col, v in SPECS.items():
        d = w.dropna(subset=v)                            # listwise deletion, as in Stata
        Z = d[v].copy()
        Z.insert(0, '(Intercept)', 1.0)
        fits[col] = f = mlogit(d['assignment'].to_numpy().astype(int), Z, base=2)
        for g in (1, 3, 4):
            for var, lab in zip(rows_order, row_labels):
                has = var in f['coef'].index
                long.append({'column': col, 'group': g, 'variable': lab,
                             'coef': f['coef'].loc[var, f'g{g}'] if has else np.nan,
                             'se': f['se'].loc[var, f'g{g}'] if has else np.nan,
                             'se_robust': f['se_robust'].loc[var, f'g{g}'] if has else np.nan,
                             'n': f['n']})
    long = pd.DataFrame(long)
    long.to_csv(out / 'tableS18_mlogit.csv', index=False)

    # Display in the layout of Table S.XVIII: coefficient (standard error).
    for g in (1, 3, 4):
        sub = long[long['group'] == g]
        cell = np.where(sub['coef'].isna(), '-',
                        sub['coef'].map('{:.3f}'.format) + ' (' + sub['se'].map('{:.3f}'.format) + ')')
        table = pd.DataFrame(np.array(cell).reshape(len(SPECS), len(row_labels)).T,
                             index=row_labels, columns=list(SPECS))
        print(f'\nTable S.XVIII: Group {g} vs. Group 2 (low democracy)')
        print(table.to_string())
    print(f'\nSample size in column (10): {fits["(10)"]["n"]} (paper: N = 68)')
    print('Published, column (1), Group 1: 1.39 (0.971); Group 3: 0.959 (1.19); Group 4: -1.06 (1.14).')
    print('Explaining_the_waves.do calls mlogit without vce(cluster ...), i.e. Stata\'s default\n'
          '(observed information) standard errors, reported in column `se`; `se_robust` in the\n'
          'CSV file gives the clustered alternative mentioned in the notes to the table.')


if __name__ == '__main__':
    main()
