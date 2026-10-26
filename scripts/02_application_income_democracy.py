'''scripts/02_application_income_democracy.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184, Section 5 (income and democracy) and Supplementary Appendix S.7.

Follows GFE_estimates.m (original code by Stephane Bonhomme and Elena Manresa)
and produces:

    output/table_S11_gfe_estimates.csv  Table S.XI: objective, BIC, estimates,
                                        and three standard errors (large-T,
                                        fixed-T, bootstrap), G = 1..15
    output/figure1_estimates_by_G.pdf   Figure 1: estimates by G with 95%
                                        bootstrap intervals
    output/figure2_grouped_patterns.pdf Figure 2: G = 4 group-specific time
                                        effects, democracy, and income by group
    output/figure2_map.pdf              Figure 2 map (needs the authors'
                                        world-c.dta / world-d.dta; see BM_DIR)
    output/figureS4_profiles_G2356.pdf  Figure S.4: G = 2, 3, 5, 6
    output/group_membership_G4.csv      group membership by country (G = 4)

Model (equation (13) in the paper):
    dem_it = theta1 dem_{i,t-1} + theta2 income_{i,t-1} + alpha_{g_i t} + v_it,
with the cumulative ("long-run") income effect theta2 / (1 - theta1).

By default the script uses the authors' optimal groupings, so it runs in
seconds. Set REESTIMATE = True to recompute them with VNS (Algorithm 2), and
BOOTSTRAP_B > 0 to recompute bootstrap standard errors (the default uses the
authors' bootstrap s.e., std_cluster_b in GFE_estimates.m, computed with 100
replications and VNS (5; 10; 5)).

Python translation Copyright (C) 2026 Eliot R. Weinstein. Licensed under the
GNU GPL, version 2 or (at your option) any later version; see LICENSE. The
license covers the translation only, not Bonhomme and Manresa's method or
original code.
'''

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')            # write files; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import (ROOT, load_bm_panel, load_bm_published, gfe, gfe_bic,  # noqa: E402
                 gfe_vcov_largeT, gfe_se_theta, gfe_delta, gfe_vcov_fixedT,
                 gfe_bandwidth, gfe_bootstrap, group_means, plot_profiles, plot_by_G)

REESTIMATE = False   # True: recompute the groupings with VNS (10; 10; 10)
BOOTSTRAP_B = 0      # e.g. 100 to recompute bootstrap s.e. (slow: hours for all G)
CORES = 1            # worker processes for the bootstrap
BM_DIR = None        # folder with the authors' world-c.dta and world-d.dta (for the map)
SEED = 1

GROUP_NAMES = ['1: high democracy', '2: low democracy', '3: early transition', '4: late transition']
COLS = ['0', '0.6', '0.4', '0.75']          # matplotlib grays, as in the paper
LTYS = ['-', '-.', '--', '-']


def cum(th):
    '''Cumulative income effect theta2 / (1 - theta1). Defined at the top
    level (not as a lambda) so the bootstrap can send it to worker processes.'''
    return th[1] / (1 - th[0])


def cum_grad(th):
    return np.array([th[1] / (1 - th[0]) ** 2, 1 / (1 - th[0])])


def three_panels(p, years, alpha, groups, G, file, legend):
    '''Time effects, democracy, and income by group, side by side (Figure 2).'''
    # group_means() only needs an object with a .panel attribute plus the grouping.
    fit_like = SimpleNamespace(panel=p, groups=groups, G=G)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    plot_profiles(axes[0], alpha, years, 'time effects', colors=COLS, linestyles=LTYS, labels=legend)
    plot_profiles(axes[1], group_means(fit_like, p.y), years, 'democracy', ylim=(0, 1),
                  colors=COLS, linestyles=LTYS)
    plot_profiles(axes[2], group_means(fit_like, p.X[:, 1]), years, 'income', ylim=(7, 9.5),
                  colors=COLS, linestyles=LTYS)
    fig.legend(loc='lower center', ncol=len(legend), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(file)
    plt.close(fig)


def draw_map(membership, out_file):
    '''Map of the country classification (spmap in GFE_replication_code.do),
    drawn from the authors' world-c.dta (coordinates) / world-d.dta (ids).'''
    from matplotlib.patches import Polygon
    wc = pd.read_stata(Path(BM_DIR) / 'world-c.dta')
    wd = pd.read_stata(Path(BM_DIR) / 'world-d.dta')
    code_to_group = dict(zip(membership['code'], membership['group']))
    wd['group'] = wd['GMI_CNTRY'].map(code_to_group)
    wd.loc[wd['id'] == 67, 'group'] = np.nan     # Gaza strip shares Israel's code (as in the .do file)
    id_to_group = dict(zip(wd['id'], wd['group']))
    fill = ['0.05', '0.4', '0.62', '0.82']
    fig, ax = plt.subplots(figsize=(11, 6))
    for pid, pts in wc.groupby('_ID'):
        g = id_to_group.get(pid, np.nan)
        # Polygons are separated by rows with missing coordinates, as in Stata's spmap.
        xy = pts[['_X', '_Y']].to_numpy()
        breaks = np.flatnonzero(np.isnan(xy).any(axis=1))
        for seg in np.split(xy, breaks):
            seg = seg[~np.isnan(seg).any(axis=1)]
            if len(seg) < 3:
                continue
            if np.isnan(g):
                ax.add_patch(Polygon(seg, facecolor='white', edgecolor='0.6', lw=0.3, hatch='///'))
            else:
                ax.add_patch(Polygon(seg, facecolor=fill[int(g) - 1], edgecolor='white', lw=0.3))
    ax.set_xlim(np.nanmin(wc['_X']), np.nanmax(wc['_X']))
    ax.set_ylim(-58, np.nanmax(wc['_Y']))                 # no Antarctica
    ax.set_aspect('equal')
    ax.axis('off')
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=fill[k], label=GROUP_NAMES[k]) for k in range(4)]
    handles.append(Patch(facecolor='white', edgecolor='0.4', hatch='///', label='not in sample'))
    ax.legend(handles=handles, loc='lower left', frameon=False)
    fig.savefig(out_file)
    plt.close(fig)


def main():
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    p = load_bm_panel()
    pub = load_bm_published()
    years = p.time_labels
    Gs = range(1, 16)
    rng = np.random.default_rng(SEED)

    # -------------------------------------------------------------------------
    # 1. GFE estimates for G = 1..15
    # -------------------------------------------------------------------------
    fits = {}
    for G in Gs:
        if G == 1:
            fits[G] = gfe(p, 1)
        elif REESTIMATE:
            fits[G] = gfe(p, G, algorithm='vns', n_starts=10, neighmax=10, itermax=10, seed=rng)
        else:
            fits[G] = gfe(p, G, groups=pub['groupings'][f'G{G}'])
    bic = gfe_bic(list(fits.values())).set_index('G')

    rows = []
    for G, f in fits.items():
        V = gfe_vcov_largeT(f)
        se = gfe_se_theta(f, V)
        se_c = np.sqrt(gfe_delta(f, V, cum_grad(f.theta)))
        if G >= 2:
            Vf = gfe_vcov_fixedT(f, gfe_bandwidth(f))
            sf = gfe_se_theta(f, Vf)
            sf_c = np.sqrt(gfe_delta(f, Vf, cum_grad(f.theta)))
        else:
            sf, sf_c = [np.nan, np.nan], np.nan
        b = pub['bootstrap_se'].set_index('G')
        boot = (b.loc[G, ['se_lag_dem', 'se_lag_income', 'se_cumulative_income']].to_numpy()
                if G in b.index else [np.nan] * 3)
        if BOOTSTRAP_B > 0 and G >= 2:
            bs = gfe_bootstrap(f, B=BOOTSTRAP_B, algorithm='vns', fun=cum, seed=rng.integers(10**9),
                               cores=CORES, n_starts=5, neighmax=10, itermax=5)
            boot = bs['se']
        rows.append({'G': G, 'objective': f.objective, 'BIC': bic.loc[G, 'BIC'],
                     'theta1': f.theta[0], 'theta2': f.theta[1], 'cumulative': cum(f.theta),
                     'se1_largeT': se[0], 'se1_fixedT': sf[0], 'se1_boot': boot[0],
                     'se2_largeT': se[1], 'se2_fixedT': sf[1], 'se2_boot': boot[1],
                     'sec_largeT': se_c, 'sec_fixedT': sf_c, 'sec_boot': boot[2]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'table_S11_gfe_estimates.csv', index=False)
    print(tab.iloc[:, :9].round(3).to_string(index=False))
    print('BIC selects G =', bic.attrs['G_BIC'])

    # -------------------------------------------------------------------------
    # 2. Figure 1: estimates by G with 95% bootstrap intervals
    #    (G = 1: common time effects only, large-T clustered s.e., as in the paper)
    # -------------------------------------------------------------------------
    def se_boot(col_boot, col_lt):
        return tab[col_boot].fillna(tab[col_lt])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_by_G(axes[0], list(Gs), tab.theta1, se_boot('se1_boot', 'se1_largeT'),
              'lagged democracy coefficient', ylim=(0, 1))
    plot_by_G(axes[1], list(Gs), tab.theta2, se_boot('se2_boot', 'se2_largeT'),
              'income coefficient', ylim=(0, 0.15))
    plot_by_G(axes[2], list(Gs), tab.cumulative, se_boot('sec_boot', 'sec_largeT'),
              'implied cumulative income effect', ylim=(0, 0.4))
    fig.tight_layout()
    fig.savefig(out / 'figure1_estimates_by_G.pdf')
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Figure 2: grouped patterns of democracy (G = 4)
    # -------------------------------------------------------------------------
    # Relabel the published groups as in the paper: 1 = high democracy,
    # 2 = low democracy, 3 = early transition, 4 = late transition. The labels
    # in GFE_estimates.m are a permutation of these; the mapping is checked
    # against the "assignment" variable of explaining_the_waves_G4.dta.
    # In 1-based terms the map is published label k -> paper label
    # c(2, 3, 4, 1)[k]; in 0-based terms it is [1, 2, 3, 0].
    paper_labels = np.array([1, 2, 3, 0])
    f4 = gfe(p, 4, groups=paper_labels[pub['groupings']['G4'].to_numpy()])
    three_panels(p, years, f4.alpha, f4.groups, 4, out / 'figure2_grouped_patterns.pdf', GROUP_NAMES)

    membership = pd.DataFrame({'code': p.unit_labels,
                               'group': f4.groups + 1,          # back to the paper's 1..4 labels
                               'group_name': [GROUP_NAMES[g] for g in f4.groups]})
    membership.to_csv(out / 'group_membership_G4.csv', index=False)
    print(membership['group_name'].value_counts().sort_index())
    if BM_DIR is not None:
        draw_map(membership, out / 'figure2_map.pdf')

    # -------------------------------------------------------------------------
    # 4. Figure S.4: profiles for G = 2, 3, 5, 6
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    for row, G in enumerate((2, 3, 5, 6)):
        f = fits[G]
        plot_profiles(axes[row, 0], f.alpha, years, f'time effects (G = {G})')
        plot_profiles(axes[row, 1], group_means(f, p.y), years, 'democracy', ylim=(0, 1))
        plot_profiles(axes[row, 2], group_means(f, p.X[:, 1]), years, 'income', ylim=(7, 9.5))
    fig.tight_layout()
    fig.savefig(out / 'figureS4_profiles_G2356.pdf')
    plt.close(fig)
    print('Figures and tables written to output/.')


if __name__ == '__main__':
    main()
