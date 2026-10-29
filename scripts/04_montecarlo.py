'''scripts/04_montecarlo.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184, Supplementary Appendix S.6 (simulation exercises).

Follows the authors' Monte Carlo code (original code by Stephane Bonhomme and
Elena Manresa):
    MonteCarlo/DGP_final.m                    data generating process
    MonteCarlo/Executable Files/MC1.exe       GFE estimation in each sample
    MonteCarlo/Executable Files/MC_bootstrap.exe  bootstrap in each sample
    MonteCarlo/Processing_bootstrap_G10_dgp1.m    bootstrap coverage
    MonteCarlo/BIC_G3_MC.m                    BIC choice of G (Table S.VIII)

Design (Section S.6.1): N = 90, T = 7, and

    y_it = theta1 y_{i,t-1} + theta2 x_it + alpha_{g_i t} + v_it ,

with x_it = lagged log income, taken from the data, and (theta, alpha, g_i)
fixed at their GFE estimates on the data with G0 groups. The outcome is
generated recursively from each country's observed 1970 lagged democracy.
Errors:
    DGP = 'iid'      : i.i.d. normal with variance Q / (NT - G T - N - K)
                       (sigma2_naive in DGP_final.m), Tables S.III-S.IV
    DGP = 'wild'     : unit-level wild bootstrap of the GFE residuals,
                       v*_it = eta_i * v_hat_it, eta_i ~ N(0, 1), as coded in
                       DGP_final.m (DGP = 2), Tables S.VI-S.VII
    DGP = 'resample' : unit-specific residual sequences drawn with
                       replacement, as DESCRIBED in the text of Section S.6.2.
                       The code and the text differ; we offer both.

For each simulated sample we compute the GFE estimates (VNS with (5; 10; 5),
as in the paper), the misclassification frequency after the best relabeling
(Hungarian algorithm: the exact minimum over all G! relabelings), large-T and
fixed-T standard errors with the coverage of 95% normal intervals, and
optionally (BOOTSTRAP_B > 0) bootstrap standard errors and the coverage of 95%
percentile intervals.

Cost. The paper uses 1,000 replications and 100 bootstrap replications per
sample. The defaults below are much smaller so that the script runs in
minutes; results then carry simulation noise of order 1/sqrt(N_REP). Each
replication gets its own child seed, so results do not depend on CORES.

Python translation Copyright (c) 2026 Eliot R. Weinstein. Licensed under the
BSD 3-Clause License; see LICENSE and NOTICE.md (the license covers the
translation only, not Bonhomme and Manresa's method or original code).
'''

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy.stats import norm     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import (ROOT, gfe_panel, load_bm_panel, load_bm_published, gfe,  # noqa: E402
                 gfe_vcov_largeT, gfe_vcov_fixedT, gfe_bandwidth, gfe_se_theta,
                 gfe_delta, gfe_bootstrap, match_groups)

G0 = 3                 # true number of groups: 3, 5, or 10 in the paper
DGP = 'iid'            # 'iid', 'wild', or 'resample' (see above)
N_REP = 50             # 1,000 in the paper
VNS_TUNING = {'n_starts': 5, 'neighmax': 10, 'itermax': 5}     # (5; 10; 5)
BOOTSTRAP_B = 0        # 100 in the paper (with Algorithm 1, 1,000 starts)
BOOT_TUNING = {'algorithm': 'alg1', 'n_starts': 100}          # paper: 1,000 starts
RUN_BIC = False        # Table S.VIII: BIC choice of G across simulated samples
BIC_G = {3: list(range(1, 8)), 10: list(range(3, 16))}.get(G0, list(range(1, 8)))  # as in BIC_G3_MC.m, BIC_G10_MC.m
BIC_GMAX = 15          # G used for sigma2_hat in the BIC penalty (footnote 12)
# Where the G = 15 grouping for sigma2_hat comes from:
#   'own'     each sample's own G = 15 fit, as (S20) and footnote 12 describe
#             (and as BIC_G10_MC.m does);
#   'g0_10'   as BIC_G3_MC.m does for G0 = 3: it loads assignmentG010G15.txt, the
#             G = 15 groupings estimated on the G0 = 10 simulated samples, and
#             applies them to the G0 = 3 samples. Those groupings do not fit the
#             G0 = 3 data, so sigma2_hat is about twice as large, and the
#             penalty with it. This reproduces the published Table S.VIII
#             (G = 3 in 98% of samples); 'own' chooses G = 5-6 instead.
BIC_SIGMA2 = 'own'
CORES = 1
SEED = 20150501

# Published values for comparison (i.i.d. normal errors, 1,000 replications).
PUBLISHED = {
    3: {'mean': [0.391, 0.099, 0.163], 'misclass': 9.50,
        'se': [[0.035, 0.051, 0.068, 0.043], [0.0093, 0.0132, 0.0156, 0.0137], [0.013, 0.022, 0.030, 0.021]],
        'cov': [[0.847, 0.965, 0.970], [0.723, 0.883, 0.914], [0.693, 0.917, 0.900]]},
    5: {'mean': [0.262, 0.086, 0.117], 'misclass': 9.68,
        'se': [[0.037, 0.068, 0.097, 0.058], [0.0088, 0.0135, 0.0160, 0.0112], [0.011, 0.022, 0.035, 0.022]],
        'cov': [[0.848, 0.961, 0.973], [0.788, 0.932, 0.943], [0.710, 0.936, 0.928]]},
    10: {'mean': [0.286, 0.078, 0.110], 'misclass': 44.73,
         'se': [[0.037, 0.048, 0.091, 0.059], [0.0074, 0.0095, 0.0156, 0.0103], [0.009, 0.012, 0.026, 0.015]],
         'cov': [[0.798, 0.902, 0.992], [0.841, 0.912, 0.996], [0.783, 0.904, 0.986]]},
}


def cum(th):
    return th[1] / (1 - th[0])


def cum_grad(th):
    return np.array([th[1] / (1 - th[0]) ** 2, 1 / (1 - th[0])])


def simulate_panel(truth, dgp, sigma2_naive, rng):
    '''Simulate one data set from the DGP of DGP_final.m.
    Returns a balanced GFEPanel with the same layout as the data.'''
    p = truth.panel
    N, T = p.N, p.T
    ehat = truth.residuals.reshape(N, T).T                 # T x N GFE residuals
    if dgp == 'iid':
        # V / std(V) rescales the draws to have sample variance exactly 1
        V = rng.normal(size=N * T)
        err = (np.sqrt(sigma2_naive) * V / V.std(ddof=1)).reshape(N, T).T
    elif dgp == 'wild':
        err = ehat * rng.normal(size=N)[None, :]
    elif dgp == 'resample':
        err = ehat[:, rng.integers(0, N, size=N)]
    else:
        raise ValueError("DGP should be 'iid', 'wild', or 'resample'")
    lag_inc = p.X[:, 1].reshape(N, T).T
    delta = truth.alpha[:, truth.groups]                    # T x N: alpha_{g_i t}
    dm = np.zeros((T + 1, N))
    dm[0] = p.X[:, 0].reshape(N, T).T[0]                    # observed 1970 lagged democracy
    for t in range(T):
        dm[t + 1] = truth.theta[0] * dm[t] + truth.theta[1] * lag_inc[t] + delta[t] + err[t]
    # Back to the stacked-by-unit layout: transpose T x N to N x T, then ravel.
    return gfe_panel(dm[1:].T.ravel(),
                     np.column_stack([dm[:-1].T.ravel(), lag_inc.T.ravel()]),
                     N=N, T=T, unit_labels=p.unit_labels, time_labels=p.time_labels,
                     x_names=p.x_names)


def one_replication(args):
    '''One Monte Carlo replication (top-level, so worker processes can run it).'''
    r, truth, theta0, sigma2_naive, seed, truth10 = args
    rng = np.random.default_rng(seed)
    ps = simulate_panel(truth, DGP, sigma2_naive, rng)
    fit = gfe(ps, G0, algorithm='vns', seed=rng, **VNS_TUNING)
    est = np.array([fit.theta[0], fit.theta[1], cum(fit.theta)])

    # Misclassification, after the best relabeling of the estimated groups.
    mg = match_groups(fit.groups, truth.groups, G0)
    alpha_rel = np.empty((ps.T, G0))
    alpha_rel[:, mg['map']] = fit.alpha

    # Standard errors: large-T clustered (S.3) and fixed-T (Pollard).
    def se_of(V):
        return np.append(gfe_se_theta(fit, V), np.sqrt(gfe_delta(fit, V, cum_grad(fit.theta))))
    se_lt = se_of(gfe_vcov_largeT(fit))
    se_ft = se_of(gfe_vcov_fixedT(fit, gfe_bandwidth(fit)))
    z = norm.ppf(0.975)
    out = {'est': est, 'misclass': mg['misclassification'], 'alpha': alpha_rel,
           'se_largeT': se_lt, 'se_fixedT': se_ft,
           'cov_largeT': (np.abs(est - theta0) <= z * se_lt).astype(float),
           'cov_fixedT': (np.abs(est - theta0) <= z * se_ft).astype(float)}

    if BOOTSTRAP_B > 0:
        bs = gfe_bootstrap(fit, B=BOOTSTRAP_B, fun=cum, seed=rng.integers(10**9), **BOOT_TUNING)
        q = bs['percentile']
        out['se_boot'] = bs['se']
        out['cov_boot'] = ((q[0] < theta0) & (theta0 < q[1])).astype(float)

    if RUN_BIC:
        out['bic_Q'] = np.array([gfe(ps, G, algorithm='vns', seed=rng, **VNS_TUNING).objective
                                 for G in BIC_G])
        if BIC_SIGMA2 == 'g0_10':
            # Emulates BIC_G3_MC.m: a G = 15 grouping estimated on an independent
            # G0 = 10 sample, applied to this G0 = 3 sample.
            t10, s10 = truth10
            p10 = simulate_panel(t10, DGP, s10, rng)
            g15 = gfe(p10, BIC_GMAX, algorithm='vns', seed=rng, **VNS_TUNING).groups
            Qmax = gfe(ps, BIC_GMAX, groups=g15).objective
        else:
            Qmax = gfe(ps, BIC_GMAX, algorithm='vns', seed=rng, **VNS_TUNING).objective
        out['bic_sigma2'] = Qmax / (ps.N * ps.T - BIC_GMAX * ps.T - ps.N - ps.K)
    return out


def main():
    out_dir = ROOT / 'output'
    out_dir.mkdir(exist_ok=True)
    p = load_bm_panel()
    pub = load_bm_published()['groupings']
    N, T, K = p.N, p.T, p.K

    # 1. The DGP: parameters fixed at their GFE values on the data.
    truth = gfe(p, G0, groups=pub[f'G{G0}'])
    theta0 = np.array([truth.theta[0], truth.theta[1], cum(truth.theta)])
    sigma2_naive = truth.objective / (N * T - G0 * T - N - K)
    print(f'True values (G0 = {G0}): theta1 = {theta0[0]:.4f}, theta2 = {theta0[1]:.4f}, '
          f'cumulative = {theta0[2]:.4f}; sigma = {np.sqrt(sigma2_naive):.4f}')

    # 2-3. Run the replications (optionally in parallel; works on Windows
    # because of the `if __name__ == "__main__"` guard below).
    seeds = np.random.SeedSequence(SEED).spawn(N_REP)
    truth10 = None
    if RUN_BIC and BIC_SIGMA2 == 'g0_10':
        t10 = gfe(p, 10, groups=pub['G10'])
        truth10 = (t10, t10.objective / (N * T - 10 * T - N - K))
    jobs = [(r, truth, theta0, sigma2_naive, seeds[r], truth10) for r in range(N_REP)]
    t0 = time.time()
    if CORES > 1:
        with ProcessPoolExecutor(max_workers=CORES) as ex:
            sims = list(ex.map(one_replication, jobs))
    else:
        sims = []
        for j in jobs:
            sims.append(one_replication(j))
            if (j[0] + 1) % 10 == 0:
                print(f'replication {j[0] + 1} of {N_REP}')
    print(f'Elapsed: {time.time() - t0:.0f}s')

    # 4. Tables S.III / S.VI (bias) and S.IV / S.VII (inference).
    def stack(name):
        return np.vstack([s[name] for s in sims])
    pars = ['theta1', 'theta2', 'cumulative']
    est = stack('est')
    bias_tab = pd.DataFrame({'parameter': pars, 'true': theta0, 'gfe_mean': est.mean(axis=0),
                             'gfe_median': np.median(est, axis=0), 'mc_sd': est.std(axis=0, ddof=1)})
    print(f'\nBias of the GFE estimator (G0 = {G0}, DGP = {DGP}, {N_REP} replications)')
    print(bias_tab.round(4).to_string(index=False))
    print(f'Average misclassification frequency: {100 * np.mean([s["misclass"] for s in sims]):.2f}%')

    inf_tab = pd.DataFrame({
        'parameter': pars,
        'se_largeT_median': np.median(stack('se_largeT'), axis=0),
        'se_fixedT_median': np.median(stack('se_fixedT'), axis=0),
        'se_boot_median': np.median(stack('se_boot'), axis=0) if BOOTSTRAP_B > 0 else np.nan,
        'mc_sd': est.std(axis=0, ddof=1),
        'coverage_largeT': stack('cov_largeT').mean(axis=0),
        'coverage_fixedT': stack('cov_fixedT').mean(axis=0),
        'coverage_boot': stack('cov_boot').mean(axis=0) if BOOTSTRAP_B > 0 else np.nan})
    print('\nInference: median standard errors, Monte Carlo s.d., and coverage of 95% intervals')
    print(inf_tab.round(4).to_string(index=False))
    if DGP == 'iid' and G0 in PUBLISHED:
        pb = PUBLISHED[G0]
        print(f'\nPublished (Tables S.III and S.IV): GFE mean {pb["mean"]}; misclassified {pb["misclass"]}%')
        print('  median s.e. (large-T, fixed-T, bootstrap) and MC s.d., by parameter:', pb['se'])
        print('  coverage (large-T, fixed-T, bootstrap), by parameter:', pb['cov'])
    tag = f'G{G0}_{DGP}'
    pd.concat([bias_tab, inf_tab.drop(columns=['parameter', 'mc_sd'])], axis=1) \
        .to_csv(out_dir / f'montecarlo_{tag}.csv', index=False)

    # 5. Figure S.1: true and mean estimated group-specific time effects.
    alpha_mean = np.mean([s['alpha'] for s in sims], axis=0)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(range(1, T + 1), truth.alpha, color='black', lw=2)
    ax.plot(range(1, T + 1), alpha_mean, color='0.4', lw=2, ls='--')
    ax.set_xlabel('t')
    ax.set_ylabel('group-specific time effects')
    ax.legend([plt.Line2D([], [], color='black', lw=2), plt.Line2D([], [], color='0.4', lw=2, ls='--')],
              ['true', 'mean of estimates'], loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f'figureS1_alpha_bias_{tag}.pdf')
    plt.close(fig)

    # 6. Table S.VIII: BIC choice of the number of groups (optional). As in
    # BIC_G3_MC.m, sigma2_hat is averaged over the simulated samples
    # (sigma22 = mean(sigma2)) and the same value is used in every sample.
    if RUN_BIC:
        NT = N * T
        Qm = stack('bic_Q')
        s2 = np.mean([s['bic_sigma2'] for s in sims])
        bic = Qm / NT + s2 * np.log(NT) * (np.array(BIC_G) * T + N + K) / NT
        choice = np.array(BIC_G)[bic.argmin(axis=1)]
        print(f'\nBIC choice of G across {N_REP} samples (true G0 = {G0}; sigma2_hat = {s2:.4f}, source {BIC_SIGMA2!r}):')
        tab = pd.Series(choice).value_counts().reindex(BIC_G, fill_value=0)
        print(tab)
        tab.rename_axis('G').rename('samples').to_csv(out_dir / f'tableS8_bic_choice_G{G0}_{BIC_SIGMA2}.csv')


if __name__ == '__main__':
    main()
