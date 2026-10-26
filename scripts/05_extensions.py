'''scripts/05_extensions.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184, Supplementary Appendix S.7.3 (robustness of the grouped patterns).

Follows (original code by Stephane Bonhomme and Elena Manresa):
    GFE_estimates.m (continent groups)       Figure S.5
    GFE_deviations_estimates.m               Table S.XIV, Figure S.6 (top)
    GFE_replication_code_IV.do               Table S.XV
    GFE_twolayers_computation.m / _estimates.m   Figure S.6 (bottom)
    Heterogeneous_coefficientsA.m / B.m      Table S.XVI, Figure S.7

Each exercise asks the same question in a different way: is the division of
countries into "stable", "early transition", and "late transition" paths a
robust feature of the data, or an artifact of the baseline specification?

By default the script evaluates the published groupings (and, for the
heterogeneous-coefficient models, the groupings found by the R translation's
search; see data/rcode_hetero_groupings.csv), so it runs in seconds. The flags
turn on the searches.

Python translation Copyright (C) 2026 Eliot R. Weinstein. Licensed under the
GNU GPL, version 2 or (at your option) any later version; see LICENSE. The
license covers the translation only, not Bonhomme and Manresa's method or
original code.
'''

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import (ROOT, DATA, load_bm_panel, load_bm_published, gfe,  # noqa: E402
                 gfe_vcov_largeT, gfe_se_theta, gfe_delta, gfe_within, gfe_fd_iv,
                 gfe_hetero, gfe_twolayer, group_means, plot_profiles, match_groups)

SEARCH_DEVIATIONS = False   # True: re-estimate the deviation groupings with VNS
SEARCH_TWOLAYER = False     # True: search the (5, 2, 2) two-layer grouping (slow)
SEARCH_HETERO = False       # True: search the heterogeneous-coefficient groupings (slow)
HETERO_STARTS = 20000       # the paper used 1,000,000 random starts
SEED = 5

BASE_NAMES = ['high democracy', 'low democracy', 'early transition', 'late transition']


def cum(th):
    return th[1] / (1 - th[0])


def cum_grad(th):
    return np.array([th[1] / (1 - th[0]) ** 2, 1 / (1 - th[0])])


def three_panels(p, alpha, groups, file, legend, colors, linestyles,
                 alpha_label='time effects', alpha_ylim=None):
    '''Time effects, democracy, and income by group (as in Figure S.4).'''
    G = alpha.shape[1]
    f = SimpleNamespace(panel=p, groups=np.asarray(groups), G=G)
    years = p.time_labels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    plot_profiles(axes[0], alpha, years, alpha_label, colors=colors, linestyles=linestyles,
                  labels=legend, ylim=alpha_ylim)
    plot_profiles(axes[1], group_means(f, p.y), years, 'democracy', ylim=(0, 1),
                  colors=colors, linestyles=linestyles)
    plot_profiles(axes[2], group_means(f, p.X[:, 1]), years, 'income', ylim=(7, 10),
                  colors=colors, linestyles=linestyles)
    fig.legend(loc='lower center', ncol=len(legend), frameon=False,
               fontsize=8 if G > 5 else 10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(file)
    plt.close(fig)


def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    p = load_bm_panel()
    pub = load_bm_published()['groupings']
    N = p.N

    # Baseline G = 4 grouping with the paper's labels (see script 02), 0-based:
    # 0 = high democracy, 1 = low democracy, 2 = early transition, 3 = late transition.
    base4 = np.array([1, 2, 3, 0])[pub['G4'].to_numpy()]

    # =========================================================================
    # 1. Continent-specific time effects (Figure S.5)
    # =========================================================================
    # Groups fixed a priori to continents, instead of estimated: a natural
    # benchmark for the GFE classification. (Continents are coded 1 Africa,
    # 2 Asia incl. Australia and New Zealand, 3 Europe, 4 South America,
    # 5 North America incl. Mexico, as in GFE_estimates.m; 0-based here.)
    continents = ['Africa', 'Asia', 'Europe', 'South America', 'North America']
    fc = gfe(p, 5, groups=pub['continent5'])
    print(f'Continent-specific time effects: theta = {np.round(fc.theta, 3)}; objective = '
          f'{fc.objective:.3f} (GFE with G = 5: {gfe(p, 5, groups=pub["G5"]).objective:.3f})')
    three_panels(p, fc.alpha, fc.groups, out / 'figureS5_continents.pdf', continents,
                 colors=['0', '0.3', '0.5', '0.65', '0.8'], linestyles=['-', '--', '-.', '-', '--'])

    # =========================================================================
    # 2. Country fixed effects plus grouped patterns (Table S.XIV, Figure S.6 top)
    # =========================================================================
    # Model (5): y_it = x_it' theta + alpha_{g_i t} + eta_i + v_it. GFE applied
    # to deviations from country means (S.21) removes eta_i. G = 1 is two-way
    # fixed effects.
    pw = gfe_within(p)
    dev_fits = {}
    for G in range(1, 6):
        if G == 1:
            dev_fits[G] = gfe(pw, 1)
        elif SEARCH_DEVIATIONS:
            dev_fits[G] = gfe(pw, G, algorithm='vns', n_starts=10, neighmax=10, itermax=10, seed=rng)
        else:
            dev_fits[G] = gfe(pw, G, groups=pub[f'dev_G{G}'])
    rows = []
    for G, f in dev_fits.items():
        V = gfe_vcov_largeT(f)
        se = gfe_se_theta(f, V)
        rows.append({'G': G, 'objective': f.objective, 'theta1': f.theta[0], 'se1': se[0],
                     'theta2': f.theta[1], 'se2': se[1], 'cumulative': cum(f.theta),
                     'se_cum': np.sqrt(gfe_delta(f, V, cum_grad(f.theta)))})
    tabS14 = pd.DataFrame(rows)
    print('\nTable S.XIV: GFE in deviations from country means')
    print(tabS14.round(3).to_string(index=False))
    tabS14.to_csv(out / 'tableS14_deviations.csv', index=False)
    # Rows G = 2..5 match Table S.XIV. For G = 1 (two-way fixed effects) the
    # published standard errors (0.058, 0.049, 0.069) are larger than the
    # formula of GFE_deviations_estimates.m gives here (0.053, 0.045, 0.064),
    # plausibly because that row comes from Stata with country dummies counted
    # in the degrees of freedom.

    # The paper: with G = 3, "the match with the classification without
    # country fixed-effects and G = 4 is perfect for 80 out of the 90
    # countries", the stable group mostly comprising high- and low-democracy
    # countries.
    dev3 = dev_fits[3]
    xt = pd.crosstab(pd.Series(dev3.groups, name='deviations_G3'),
                     pd.Series([BASE_NAMES[g] for g in base4], name='baseline_G4'))
    print(xt)
    stable = (xt['high democracy'] + xt['low democracy']).idxmax()
    early = xt['early transition'].idxmax()
    late = xt['late transition'].idxmax()
    alike = (xt.loc[stable, 'high democracy'] + xt.loc[stable, 'low democracy']
             + xt.loc[early, 'early transition'] + xt.loc[late, 'late transition'])
    print(f'Countries classified alike: {alike} of {N} (paper: 80)')
    dev_names = [''] * 3
    for g, name in zip((stable, early, late), ('stable', 'early transition', 'late transition')):
        dev_names[g] = name
    shade = {stable: '0', early: '0.4', late: '0.7'}
    three_panels(p, dev3.alpha, dev3.groups, out / 'figureS6_top_deviations_G3.pdf', dev_names,
                 colors=[shade[g] for g in range(3)], linestyles=['-'] * 3,
                 alpha_label='time effects (deviations)')

    # =========================================================================
    # 3. First-differenced IV with estimated groups (Table S.XV)
    # =========================================================================
    # Delta y_it = theta1 Delta y_{i,t-1} + theta2 Delta x_{i,t-1} + mu_{g_i t} + Delta v_it,
    # instrumenting Delta y_{i,t-1} by y_{i,t-2}; groups from GFE in
    # deviations. IV corrects the small-T (Nickell) bias of the within
    # estimator of theta1.
    raw = pd.read_csv(DATA / 'ajry_5yearpanel_extract.csv')
    bal = p.data
    rows = []
    for G, f in dev_fits.items():
        df = pd.DataFrame({'code': bal['code'], 'year': bal['year'],
                           'group': f.groups[bal['unit'].to_numpy() - 1]})   # 'unit' is 1-based
        df = df.merge(raw[['code', 'code_numeric', 'year', 'fhpolrigaug', 'lrgdpch']],
                      on=['code', 'year'])
        r = gfe_fd_iv(df, id='code_numeric', year='year', y='fhpolrigaug', x='lrgdpch', group='group')
        rows.append({'G': G, 'theta1': r['coef'][0], 'se1': r['se'][0], 'theta2': r['coef'][1],
                     'se2': r['se'][1], 'cumulative': cum(r['coef']), 'n_obs': r['n']})
    tabS15 = pd.DataFrame(rows)
    print('\nTable S.XV: first-differenced IV with estimated groups')
    print(tabS15.round(3).to_string(index=False))
    print('(The published standard errors of the cumulative effect are not reproduced here: the\n'
          ' code that computed them is not part of the replication files.)')
    tabS15.to_csv(out / 'tableS15_iv.csv', index=False)

    # =========================================================================
    # 4. Two layers of groups (Figure S.6, bottom)
    # =========================================================================
    # Model (S.40): three "layer-1" groups with their own time paths, split
    # into (5, 2, 2) subgroups that differ from the layer-1 path by a constant:
    #   alpha_{(g,h) t} = a_{g t} + b_{g h}.
    H = [5, 2, 2]
    tl = (gfe_twolayer(p, H, n_starts=1000, seed=rng) if SEARCH_TWOLAYER
          else gfe_twolayer(p, H, groups=pub['twolayer_522']))
    print(f'\nTwo-layer model (5, 2, 2): objective {tl["objective"]:.4f}; theta = '
          f'{np.round(tl["theta"], 3)}; cumulative income effect = {cum(tl["theta"]):.3f}')

    # Check: the constrained least squares of GFE_twolayers_computation.m
    # equals the closed form of GFE_twolayers_estimates.m (two-way demeaning
    # within each layer-1 group: y - ybar_{g t} - ybar_{(g,h)} + ybar_g).
    g1 = tl['layer1'][p.unit]
    sg = tl['groups'][p.unit]

    def cell_mean(v, *keys):
        return pd.Series(v).groupby(list(keys)).transform('mean').to_numpy()

    def dmean(v):
        return v - cell_mean(v, g1, p.time) - cell_mean(v, sg) + cell_mean(v, g1)

    yd = dmean(p.y)
    Xd = np.column_stack([dmean(p.X[:, k]) for k in range(p.K)])
    theta_closed = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)
    assert np.max(np.abs(theta_closed - tl['theta'])) < 1e-8

    # The paper: the two-layer and deviation classifications agree "except in
    # five cases".
    print('Two-layer (layer 1) vs deviations (G = 3):')
    print(pd.crosstab(pd.Series(tl['layer1'], name='two_layer'),
                      pd.Series(dev3.groups, name='deviations')))
    S = sum(H)
    first = [np.flatnonzero(tl['groups'] == s)[0] for s in range(S)]
    tl_names = [f'({tl["layer1"][i] + 1},{tl["layer2"][i] + 1})' for i in first]
    # Line styles of GFE_twolayers_estimates.m
    three_panels(p, tl['c'], tl['groups'], out / 'figureS6_bottom_twolayer.pdf', tl_names,
                 colors=['0', '0.35', '0.75', '0.65', '0.2', '0.35', '0.35', '0.65', '0.65'],
                 linestyles=['-', '-', '-', '-', '-', '-.', '--', '-.', '--'], alpha_ylim=(-0.6, 0.5))

    # =========================================================================
    # 5. Heterogeneous coefficients (Table S.XVI, Figure S.7)
    # =========================================================================
    # Model (7): y_it = x_it' theta_{g_i} + alpha_{g_i t} + v_it, G = 4.
    #   Version B: income coefficient group-specific, lagged democracy common.
    #   Version A: both coefficients group-specific.
    # The authors did not publish these groupings; the ones in
    # data/rcode_hetero_groupings.csv were found by the R translation (seed
    # 100, 20,000 starts, polish_top = 30) and reproduce Table S.XVI.
    found = pd.read_csv(DATA / 'rcode_hetero_groupings.csv', comment='#')
    spec = {'B': [1], 'A': [0, 1]}
    het = {}
    for v, hetero in spec.items():
        if SEARCH_HETERO:
            het[v] = gfe_hetero(p, 4, hetero=hetero, n_starts=HETERO_STARTS, polish_top=30, seed=rng)
        else:
            start = found[f'hetero_{v}'].to_numpy() - 1                 # 1-based in the CSV
            het[v] = gfe_hetero(p, 4, hetero=hetero, start_groups=[start])

    # Relabel the groups to match the baseline G = 4 classification, so that
    # group 1 is "high democracy", etc., as in Table S.XVI.
    rows = []
    for v, h in het.items():
        m = match_groups(h['groups'], base4, 4)
        order = np.argsort(m['map'])                 # estimated label for paper group 0..3
        beta, se_beta = h['beta'].to_numpy(), h['se_beta'].to_numpy()
        for pg in range(4):
            eg = order[pg]
            if v == 'B':
                th1, s1 = h['theta_common'].iloc[0], h['se_theta_common'].iloc[0]
                th2, s2 = beta[eg, 0], se_beta[eg, 0]
            else:
                th1, s1, th2, s2 = beta[eg, 0], se_beta[eg, 0], beta[eg, 1], se_beta[eg, 1]
            rows.append({'version': v, 'group': pg + 1, 'group_name': BASE_NAMES[pg],
                         'objective': h['objective'], 'theta1': th1, 'se1': s1,
                         'theta2': th2, 'se2': s2, 'cumulative': th2 / (1 - th1),
                         'n': int(np.sum(m['groups'] == pg)),
                         'same_as_baseline': int(np.sum((m['groups'] == pg) & (base4 == pg)))})
    tabS16 = pd.DataFrame(rows)
    print('\nTable S.XVI: heterogeneous coefficients (G = 4)')
    print(tabS16.round(3).to_string(index=False))
    print('Published, version B: theta1 = 0.288 (0.054); theta2 = 0.103 (0.019), 0.047 (0.014), '
          '0.087 (0.018), 0.082 (0.013).')
    print('Published, version A: theta1 = 0.644, 0.319, 0.016, 0.248; theta2 = 0.070, 0.041, 0.122, 0.090.')
    tabS16.to_csv(out / 'tableS16_heterogeneous.csv', index=False)

    # Figure S.7: democracy and income by group in both versions.
    cols, ltys = ['0', '0.6', '0.4', '0.75'], ['-', '-.', '--', '-']
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, v in enumerate(('B', 'A')):
        m = match_groups(het[v]['groups'], base4, 4)
        f = SimpleNamespace(panel=p, groups=m['groups'], G=4)
        plot_profiles(axes[row, 0], group_means(f, p.y), p.time_labels, 'democracy', ylim=(0, 1),
                      colors=cols, linestyles=ltys)
        axes[row, 0].set_title('heterogeneous income coefficient' if v == 'B'
                               else 'heterogeneous coefficients', fontsize=10)
        plot_profiles(axes[row, 1], group_means(f, p.X[:, 1]), p.time_labels, 'income',
                      ylim=(7, 10), colors=cols, linestyles=ltys)
    fig.tight_layout()
    fig.savefig(out / 'figureS7_heterogeneous.pdf')
    plt.close(fig)
    print('\nFigures and tables written to output/.')


if __name__ == '__main__':
    main()
