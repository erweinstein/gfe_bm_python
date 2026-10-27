'''gfe/extensions.py -- Extensions of the baseline GFE model.

Python translation of the extensions in Bonhomme and Manresa (2015),
Supplementary Appendix Section S.4, and the corresponding replication files
(original code by Stephane Bonhomme and Elena Manresa):

  (1) Unit-specific heterogeneity: GFE in deviations from unit means,
      y_it - ybar_i = (x_it - xbar_i)' theta + mu_{g_i t} + (v_it - vbar_i)
      [GFE_deviations_estimates.m; GFE_replication_code.do with Replic = 5;
       equations (S.21)-(S.22)], and the first-differenced IV estimator that
      corrects the small-T bias with a lagged outcome
      [GFE_replication_code_IV.do; equation (S.30); Table S.XV].

  (2) Heterogeneous (group-specific) coefficients:
      y_it = x_it' theta_{g_i} + alpha_{g_i t} + v_it
      [Heterogeneous_coefficientsA.m (all slopes group-specific) and
       Heterogeneous_coefficientsB.m (common lagged-outcome coefficient,
       group-specific income coefficient); equation (S.32)].

  (3) Two layers of groups: G "large" groups, each split into H_g subgroups
      whose time profiles differ from the group profile only by a constant:
      alpha_{(g,h) t} = a_{g t} + b_{g h}
      [GFE_twolayers_computation.m and GFE_twolayers_estimates.m; Figure S.6].

Python translation Copyright (c) 2026 Eliot R. Weinstein. Licensed under the
BSD 3-Clause License; see LICENSE and NOTICE.md (the license covers the
translation only, not Bonhomme and Manresa's method or original code).
'''

import numpy as np
import pandas as pd

from .core import gfe_panel, gfe_distances, _random_start
from .utils import pinv, ols_iv_cluster


# =============================================================================
# (1a) Deviations from unit means
# =============================================================================

def gfe_within(panel):
    '''Transform a panel into deviations from unit means (within
    transformation), so that GFE removes time-invariant unit effects
    (model (5) in the paper).'''
    T, N = panel.T, panel.N

    def demean(v):
        M = v.reshape(N, T).T                               # T x N
        mu = (M * panel.Dm).sum(axis=0) / panel.Dm.sum(axis=0)
        return ((M - mu[None, :]) * panel.Dm).T.ravel()

    X = (np.column_stack([demean(panel.X[:, k]) for k in range(panel.K)])
         if panel.K > 0 else None)
    return gfe_panel(demean(panel.y), X, N, T, d=panel.d,
                     unit_labels=panel.unit_labels, time_labels=panel.time_labels,
                     x_names=panel.x_names)


# =============================================================================
# (1b) First-differenced IV with estimated groups (Table S.XV)
# =============================================================================

def gfe_fd_iv(df, id='id', year='year', y='y', x='x', group='group'):
    '''First-differenced 2SLS with group x time effects, as in
    GFE_replication_code_IV.do (Replic = 5):

        Delta y_it = theta_1 Delta y_{i,t-1} + theta_2 Delta x_{i,t-1} + mu_{g_i t} + Delta v_it,

    instrumenting Delta y_{i,t-1} with y_{i,t-2}, and including (estimated)
    group x period dummies as exogenous regressors. Standard errors are
    clustered by unit (vce(cluster code_numeric)), without a small-sample
    factor, as in Stata's ivregress; this reproduces Table S.XV.

    As in the .do file, lags are taken within the estimation sample (the
    1970-2000 balanced panel), so the first two periods are lost.

    df : long DataFrame with columns id, year, y (outcome level), x (covariate
        level, e.g. log income), and group (estimated GFE group).
    '''
    df = df.sort_values([id, year]).reset_index(drop=True)
    # groupby().shift() takes lags within each unit, like Stata's L. operator
    # after tsset (but by position, as in the .do file's [_n-1] lags).
    lag = df.groupby(id)
    L1y, L2y = lag[y].shift(1), lag[y].shift(2)
    L1x, L2x = lag[x].shift(1), lag[x].shift(2)
    dd = pd.DataFrame({'id': df[id], 'year': df[year], 'group': df[group],
                       'delta_y': df[y] - L1y,
                       'lag_delta_y': L1y - L2y,
                       'lag_delta_x': L1x - L2x,
                       'lag2_y': L2y}).dropna().reset_index(drop=True)
    cell = pd.get_dummies(dd['group'].astype(str) + ' ' + dd['year'].astype(str),
                          dtype=float).to_numpy()
    X = np.column_stack([dd['lag_delta_y'], dd['lag_delta_x'], cell])
    Z = np.column_stack([dd['lag2_y'], dd['lag_delta_x'], cell])
    fit = ols_iv_cluster(dd['delta_y'].to_numpy(), X, Z, cluster=dd['id'].to_numpy())
    return {'coef': fit['coef'][:2], 'se': fit['se'][:2], 'vcov': fit['vcov'][:2, :2],
            'n': len(dd), 'n_clusters': fit['n_clusters'],
            'names': ['lag_delta_y', 'lag_delta_x']}


# =============================================================================
# (2) Heterogeneous coefficients (Heterogeneous_coefficientsA/B.m)
# =============================================================================

def gfe_hetero(panel, G, hetero=None, n_starts=1000, max_iter=1000, polish_top=0,
               start_groups=None, seed=None):
    '''GFE with group-specific slope coefficients, computed with "a
    counterpart to Algorithm 1" (footnote 17 of the Supplementary Appendix),
    as in Heterogeneous_coefficientsA.m / B.m:

        starting values: each unit assigned to a random group;
        update:     OLS of y on [group x time dummies, common covariates,
                    heterogeneous covariates x group dummies]   (pinv, as in MATLAB)
        assignment: each unit to the group with the smallest sum of squared
                    residuals, using that group's slopes and time effects;
        repeat until the parameters no longer change; keep the best of n_starts.

    hetero : 0-based indices of the covariates with group-specific
        coefficients (A.m: all of them; B.m: only income, i.e. hetero=[1]).
    polish_top : optional refinement (not in the original code): single-unit
        local search, as in Step 5 of Algorithm 2, applied to the best
        `polish_top` distinct solutions. Useful when the number of starts is
        far below the 1,000,000 used in the paper.
    start_groups : optional list of 0-based starting groupings (replaces the
        random starts).
    '''
    rng = np.random.default_rng(seed)
    N, T, K = panel.N, panel.T, panel.K
    hetero = list(range(K)) if hetero is None else list(hetero)
    common = [k for k in range(K) if k not in hetero]
    Xc, Xh = panel.X[:, common], panel.X[:, hetero]
    kh, kc = len(hetero), len(common)
    rows = np.arange(N * T)

    def design(groups):
        Dg = np.zeros((N * T, G * T))
        Dg[rows, groups[panel.unit] * T + panel.time] = 1.0
        # heterogeneous slopes: Xh interacted with group dummies (kron(X_i, gi)
        # in the MATLAB code), ordered covariate-major: (k1:g1..gG, k2:g1..gG, ...)
        Xhg = np.zeros((N * T, kh * G))
        for k in range(kh):
            Xhg[rows, k * G + groups[panel.unit]] = Xh[:, k]
        return np.hstack([Dg, Xc, Xhg])

    def unpack(par):
        alpha = par[:G * T].reshape(G, T).T
        theta_c = par[G * T:G * T + kc]
        beta = par[G * T + kc:G * T + kc + kh * G].reshape(kh, G).T   # row g: slopes of group g
        return alpha, theta_c, beta

    def ssr_by_group(par):
        alpha, theta_c, beta = unpack(par)
        out = np.empty((N, G))
        for g in range(G):
            r = panel.y - alpha[panel.time, g]
            if kc:
                r = r - Xc @ theta_c
            r = r - Xh @ beta[g]
            out[:, g] = (r ** 2).reshape(N, T).sum(axis=1)
        return out

    def ssr_of(groups):
        if len(np.unique(groups)) < G:
            return np.inf
        Xt = design(groups)
        coef = np.linalg.lstsq(Xt, panel.y, rcond=None)[0]
        return float(np.sum((panel.y - Xt @ coef) ** 2))

    def run_alg1(groups):
        for _ in range(max_iter):
            if len(np.unique(groups)) < G:
                return {'objective': np.inf, 'groups': groups}
            par = pinv(design(groups)) @ panel.y
            new_groups = np.argmin(ssr_by_group(par), axis=1)
            if np.array_equal(new_groups, groups):
                break
            groups = new_groups
        return {'objective': ssr_of(groups), 'groups': groups}

    # Random starts (or user-supplied starting groupings).
    starts = n_starts if start_groups is None else len(start_groups)
    sols, objs = [], np.empty(starts)
    for s in range(starts):
        g0 = (rng.integers(0, G, size=N) if start_groups is None
              else np.asarray(start_groups[s], dtype=int))
        sols.append(run_alg1(g0))                          # random initial assignment
        objs[s] = sols[s]['objective']

    if polish_top > 0:
        keys = [tuple(s['groups']) for s in sols]
        seen, cand = set(), []
        for s in np.argsort(objs):
            if np.isfinite(objs[s]) and keys[s] not in seen:
                seen.add(keys[s])
                cand.append(s)
            if len(cand) >= polish_top:
                break
        for s in cand:
            g, obj = sols[s]['groups'].copy(), sols[s]['objective']
            while True:
                improved = False
                for i in range(N):
                    if np.sum(g == g[i]) <= 1:
                        continue
                    for h in range(G):
                        if h == g[i]:
                            continue
                        g2 = g.copy()
                        g2[i] = h
                        o = ssr_of(g2)
                        if o < obj - 1e-10 * obj:
                            g, obj, improved = g2, o, True
                if not improved:
                    break
                a1 = run_alg1(g)                               # re-run the iterations
                if a1['objective'] < obj:
                    g, obj = a1['groups'], a1['objective']
            sols[s] = {'objective': obj, 'groups': g}
            objs[s] = obj

    b = int(np.argmin(objs))
    groups = sols[b]['groups']
    Xtot = design(groups)
    par = pinv(Xtot) @ panel.y

    # Large-T clustered standard errors as in the MATLAB code (including its
    # degrees-of-freedom factor NT/(NT - T - K)).
    e = panel.y - Xtot @ par
    bread = pinv(Xtot.T @ Xtot)
    score = (Xtot * e[:, None]).reshape(N, T, -1).sum(axis=1)
    V = bread @ (score.T @ score) @ bread * N * T / (N * T - T - K)
    alpha, theta_c, beta = unpack(par)
    _, se_c, se_beta = unpack(np.sqrt(np.maximum(np.diag(V), 0)))
    names = panel.x_names
    return {'groups': groups, 'objective': float(objs[b]), 'alpha': alpha,
            'theta_common': pd.Series(theta_c, index=[names[k] for k in common]),
            'se_theta_common': pd.Series(se_c, index=[names[k] for k in common]),
            'beta': pd.DataFrame(beta, index=[f'g{g + 1}' for g in range(G)],
                                 columns=[names[k] for k in hetero]),
            'se_beta': pd.DataFrame(se_beta, index=[f'g{g + 1}' for g in range(G)],
                                    columns=[names[k] for k in hetero]),
            'vcov': V, 'objectives_by_start': objs, 'G': G, 'panel': panel}


# =============================================================================
# (3) Two layers of groups (GFE_twolayers_computation.m / _estimates.m)
# =============================================================================

def _twolayer_constraints(H, T, K):
    '''Linear restrictions R c = 0 on the stacked subgroup time effects
    c = (c_{s,t}), s = 0..sum(H)-1 (subgroup-major), imposing
        c_{(g,h),t} - c_{(g,1),t} - c_{(g,h),1} + c_{(g,1),1} = 0,  h >= 2, t >= 2,
    i.e. alpha_{(g,h)t} = a_{gt} + b_{gh} (Mat_cons in the MATLAB code).'''
    S = sum(H)
    start = np.concatenate([[0], np.cumsum(H)])[:len(H)]   # subgroups before group g
    rows = []
    for g, Hg in enumerate(H):
        s1 = start[g]
        for h in range(1, Hg):
            sh = start[g] + h
            for t in range(1, T):
                r = np.zeros(S * T + K)
                r[sh * T + t] = 1
                r[s1 * T + t] = -1
                r[sh * T] = -1
                r[s1 * T] = 1
                rows.append(r)
    return np.array(rows) if rows else np.zeros((0, S * T + K))


def gfe_twolayer(panel, H, n_starts=1000, groups=None, max_iter=1000, seed=None):
    '''Two-layer GFE (Figure S.6, lower panel; Table S.XIII rightmost column).

    Units are assigned to one of sum(H) subgroups; the subgroup time profiles
    are constrained so that, within layer-1 group g, they differ only by
    constants. Computation follows GFE_twolayers_computation.m: Algorithm 1
    with the update step replaced by constrained least squares,
        par = P pinv(Xtot P) y,  P = I - pinv(R) R.

    H : list of subgroup counts, e.g. [5, 2, 2].
    groups : optional 0-based subgroup labels (0..sum(H)-1) to evaluate directly.
    '''
    rng = np.random.default_rng(seed)
    N, T, K, S = panel.N, panel.T, panel.K, sum(H)
    R = _twolayer_constraints(H, T, K)
    P = np.eye(S * T + K) - pinv(R) @ R
    rows = np.arange(N * T)

    def design(g):
        Dg = np.zeros((N * T, S * T))
        Dg[rows, g[panel.unit] * T + panel.time] = 1.0
        return np.hstack([Dg, panel.X])

    def fit_given(g):
        Xtot = design(g)
        par = P @ pinv(Xtot @ P) @ panel.y
        return {'par': par, 'c': par[:S * T].reshape(S, T).T, 'theta': par[S * T:],
                'objective': float(np.sum((panel.y - Xtot @ par) ** 2))}

    if groups is None:
        best = {'objective': np.inf}
        for _ in range(n_starts):
            theta, cc = _random_start(panel, S, rng)           # S random "centers"
            g = np.full(N, -1)
            for _ in range(max_iter):
                new_g = np.argmin(gfe_distances(panel, theta, cc), axis=1)
                if np.array_equal(new_g, g):
                    break
                g = new_g
                if len(np.unique(g)) < S:                      # an empty subgroup: stop
                    break
                f = fit_given(g)
                theta, cc = f['theta'], f['c']
            if len(np.unique(g)) < S:
                continue
            f = fit_given(g)
            if f['objective'] < best['objective']:
                best = dict(f, groups=g)
    else:
        g = np.asarray(groups, dtype=int)
        best = dict(fit_given(g), groups=g)
    layer_of = np.repeat(np.arange(len(H)), H)                 # subgroup -> layer-1 group
    within = np.concatenate([np.arange(h) for h in H])         # subgroup -> index within group
    best.update(H=list(H), layer1=layer_of[best['groups']], layer2=within[best['groups']],
                panel=panel)
    return best
