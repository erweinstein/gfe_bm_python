'''gfe/core.py -- The grouped fixed-effects (GFE) estimator.

A Python translation of the estimator and computational algorithms in

    Bonhomme, Stephane and Elena Manresa (2015), "Grouped Patterns of
    Heterogeneity in Panel Data," Econometrica 83(3), 1147-1184.
    doi:10.3982/ECTA11319

The method, the algorithms, and the original MATLAB / Fortran / Stata code are
by Stephane Bonhomme and Elena Manresa. This file translates their replication
code (by way of my R translation, gfe_bm_R, which follows it line by line)
into Python. Any errors introduced in translation are mine, not theirs.

Original files this module follows:
    Application/Data_paths.m          Algorithm 1 (iterative), MATLAB version
    Application/Fortran codes/*.f90   Algorithms 1 and 2 (VNS), with and
                                      without covariates, balanced/unbalanced
    Application/Exact algorithm/heuristics_sub.f90   Algorithms 1-2 with two
                                      covariates (used by the exact B&B)
    Supplementary Appendix, Section S.1 (algorithm statements)

The model (equation (1) in the paper):

    y_it = x_it' theta + alpha_{g_i t} + v_it,   i = 1..N, t = 1..T,

where g_i in {1, ..., G} is unit i's (unknown) group and alpha_{gt} is a
group-specific time effect. The GFE estimator solves

    (theta_hat, alpha_hat, gamma_hat) =
        argmin_{theta, alpha, gamma}  sum_i sum_t (y_it - x_it' theta - alpha_{g_i t})^2

over all groupings gamma = {g_1, ..., g_N}. For a FIXED grouping the problem is
ordinary least squares on group x time dummies; for FIXED (theta, alpha) each
unit is simply assigned to the group whose time profile is closest. Algorithm 1
alternates between those two steps (it is k-means when K = 0).

Two translation notes that matter everywhere in this package:

  * Data layout. As in the original MATLAB/Fortran code, y is a vector of
    length N*T stacked unit by unit: element (i - 1)*T + t in MATLAB's 1-based
    indexing, which is element i*T + t here because Python counts from 0.
    X is an (N*T) x K array in the same order.

  * Group labels. MATLAB, Fortran, and R label groups 1..G. Python indexing
    starts at 0, so inside this package groups are labeled 0..G-1. Wherever a
    grouping comes in from, or goes out to, the outside world (the published
    groupings in data/, printed output), the conversion is done explicitly and
    commented. Mixing the two conventions is the single easiest way to break a
    MATLAB-to-Python port (I learned this the hard way porting rq.m).

R translation Copyright (C) 2026 Eliot R. Weinstein; Python translation
Copyright (C) 2026 Eliot R. Weinstein. Licensed under the GNU GPL, version 2 or
(at your option) any later version; see LICENSE. The license covers the
translation only, not Bonhomme and Manresa's method or original code.
'''

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# =============================================================================
# 0. Panel container
# =============================================================================

@dataclass
class GFEPanel:
    '''The panel object used by every GFE routine.

    I considered using a plain dictionary (closest to R's list) but a small
    dataclass documents its own fields, gives a readable repr, and costs
    nothing. It holds no methods with state of their own; it is just the data
    plus the precomputed arrays the algorithms need.

    Attributes
    ----------
    y : (N*T,) array, stacked unit by unit.
    X : (N*T, K) array, same row order (K may be 0).
    d : (N*T,) array of 0/1 observation indicators (unbalanced panels).
    N, T, K : ints.
    Dm : (T, N) array of observation indicators (column i = unit i).
    Z : (N, T, K+1) array, Z[i, t] = d_it * (y_it, x_it').
    W : (K+1, K+1) total cross products sum_it z_it z_it'.
    unit, time : (N*T,) 0-based unit and period index of each row.
    unit_labels, time_labels : labels for printing (e.g. country codes, years).
    x_names : names of the covariates.
    balanced : True if every cell is observed.
    '''
    y: np.ndarray
    X: np.ndarray
    d: np.ndarray
    N: int
    T: int
    K: int
    Dm: np.ndarray
    Z: np.ndarray
    W: np.ndarray
    unit: np.ndarray
    time: np.ndarray
    unit_labels: list
    time_labels: list
    x_names: list
    balanced: bool
    data: pd.DataFrame = field(default=None, repr=False)

    def __repr__(self):
        return (f'GFEPanel(N={self.N}, T={self.T}, K={self.K}, '
                f'balanced={self.balanced}, x={self.x_names})')


def gfe_panel(y, X=None, N=None, T=None, d=None, unit_labels=None,
              time_labels=None, x_names=None):
    '''Build a GFEPanel from stacked arrays.

    Parameters
    ----------
    y : array of length N*T, stacked unit by unit.
    X : (N*T, K) array or None when there are no covariates.
    N, T : number of units and of time periods (T = maximal span).
    d : optional 0/1 array of observed cells. As in GFE_code.do, unobserved
        cells carry y = x = 0 and d = 0, and drop out of every sum.
    '''
    y = np.asarray(y, dtype=float).copy()
    if N is None or T is None:
        raise ValueError('Please give both N and T (the panel dimensions).')
    if y.shape != (N * T,):
        raise ValueError(f'y should have length N*T = {N * T}, not {y.shape}.')
    if X is None:
        X = np.zeros((N * T, 0))
    X = np.asarray(X, dtype=float).copy()
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != N * T:
        raise ValueError('X should have N*T rows, stacked like y.')
    if d is None:
        d = np.ones(N * T)
    d = np.asarray(d, dtype=float)
    if not np.all(np.isin(d, (0, 1))):
        raise ValueError('d should contain only 0s and 1s.')

    # Convention of GFE_code.do: unobserved cells are set to zero.
    y[d == 0] = 0.0
    X[d == 0, :] = 0.0
    if np.isnan(y).any() or np.isnan(X).any():
        raise ValueError('NA in an observed cell; set d = 0 there.')

    K = X.shape[1]
    # Unit-by-unit array for the sufficient-statistics representation of the
    # objective (see gfe_stats below): Z[i, t, :] = d_it * (y_it, x_it').
    # (In R this was a T x (K+1) x N array; putting the unit first is the
    # natural, C-ordered layout for numpy, so Z[i] is unit i's T x (K+1) block.)
    Z = np.concatenate([y[:, None], X], axis=1).reshape(N, T, K + 1)
    YX = np.concatenate([y[:, None], X], axis=1)

    return GFEPanel(
        y=y, X=X, d=d, N=N, T=T, K=K,
        Dm=d.reshape(N, T).T.copy(),
        Z=Z,
        W=YX.T @ YX,
        unit=np.repeat(np.arange(N), T),
        time=np.tile(np.arange(T), N),
        unit_labels=list(unit_labels) if unit_labels is not None else [str(i + 1) for i in range(N)],
        time_labels=list(time_labels) if time_labels is not None else list(range(1, T + 1)),
        x_names=list(x_names) if x_names is not None else [f'x{k + 1}' for k in range(K)],
        balanced=bool(np.all(d == 1)),
    )


def gfe_panel_from_df(df, id, time, y, x=()):
    '''Build a GFEPanel from a long pandas DataFrame.

    Rows with a missing outcome or covariate become unobserved cells (d = 0),
    as in the "AUTOMATIC CODE" part of GFE_code.do. Units are sorted by id.
    '''
    x = list(x)
    ids = np.sort(df[id].unique())
    times = np.sort(df[time].unique())
    N, T = len(ids), len(times)
    if df.duplicated([id, time]).any():
        raise ValueError('duplicate (id, time) rows')
    # A full N x T grid, merged with the data: the pandas equivalent of the
    # expand.grid + match() idiom in the R version.
    full = pd.MultiIndex.from_product([ids, times], names=[id, time]).to_frame(index=False)
    merged = full.merge(df[[id, time, y] + x], on=[id, time], how='left')
    yy = merged[y].to_numpy(dtype=float)
    XX = merged[x].to_numpy(dtype=float) if x else None
    obs = ~np.isnan(yy)
    if x:
        obs &= ~np.isnan(XX).any(axis=1)
        XX[~obs, :] = 0.0
    yy[~obs] = 0.0
    return gfe_panel(yy, XX, N, T, d=obs.astype(float),
                     unit_labels=[str(i) for i in ids], time_labels=list(times),
                     x_names=x)


# =============================================================================
# 1. Update step: OLS given a grouping
# =============================================================================

def gfe_update(panel, groups, G):
    '''Least-squares estimates of (theta, alpha) for a given grouping.

    This is the "update" step of Algorithm 1. With group x time dummies, OLS is
    equivalent to within-(group, time) demeaning, exactly as in the loops that
    build Ybar_gt and Xbar_gt in GFE_estimates.m:

        theta_hat = [sum_it (x_it - xbar_{g_i t})(x_it - xbar_{g_i t})']^{-1}
                    sum_it (x_it - xbar_{g_i t})(y_it - ybar_{g_i t})
        alpha_hat_gt = ybar_gt - xbar_gt' theta_hat

    Parameters
    ----------
    groups : (N,) int array of 0-based group labels.

    Returns
    -------
    dict with theta (K,), alpha (T x G, NaN where a group-period cell is
    empty), objective, residuals (N*T,), n_gt (T x G), ok.
    '''
    p = panel
    T, K, d = p.T, p.K, p.d
    groups = np.asarray(groups, dtype=int)
    gt = groups[p.unit] * T + p.time            # cell index 0..G*T-1 (group-major)

    # Group-time sums of (d, d*y, d*x). np.bincount with weights is the numpy
    # counterpart of R's rowsum(); minlength makes sure that every cell is
    # represented, even an empty one.
    n_gt = np.bincount(gt, weights=d, minlength=G * T)
    sum_y = np.bincount(gt, weights=p.y, minlength=G * T)
    sum_x = np.column_stack([np.bincount(gt, weights=p.X[:, k], minlength=G * T)
                             for k in range(K)]) if K > 0 else np.zeros((G * T, 0))
    ok = bool(np.all(n_gt[gt[d == 1]] > 0))

    with np.errstate(invalid='ignore', divide='ignore'):
        denom = np.where(n_gt > 0, n_gt, np.nan)
        ybar = sum_y / denom
        xbar = sum_x / denom[:, None]

    # Demeaned data (unobserved cells contribute zero).
    yd = np.where(d == 1, p.y - ybar[gt], 0.0)
    if K > 0:
        Xd = np.where(d[:, None] == 1, p.X - xbar[gt], 0.0)
        try:
            theta = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)
        except np.linalg.LinAlgError:
            theta = np.full(K, np.nan)
        if np.isnan(theta).any():
            ok = False
        e = yd - Xd @ theta
        alpha = ybar - xbar @ theta
    else:
        theta = np.zeros(0)
        e = yd
        alpha = ybar

    return {'theta': theta,
            # alpha comes out group-major (g0t0, g0t1, ...); reshape to G x T and
            # transpose so that column g is the time profile of group g, as in R.
            'alpha': alpha.reshape(G, T).T,
            'objective': float(np.sum(e ** 2)),
            'residuals': e,
            'n_gt': n_gt.reshape(G, T).T,
            'ok': ok}


# =============================================================================
# 2. Assignment step: closest group profile
# =============================================================================

def gfe_distances(panel, theta, alpha):
    '''Distance of every unit to every group profile, for given (theta, alpha):

        dist[i, g] = sum_t d_it (y_it - x_it' theta - alpha_gt)^2.
    '''
    p = panel
    r = p.y - (p.X @ theta if p.K > 0 else 0.0)
    R = r.reshape(p.N, p.T).T                       # T x N: column i = unit i's path
    G = alpha.shape[1]
    dist = np.empty((p.N, G))
    for g in range(G):
        a = alpha[:, g]
        if np.isnan(a).any():
            # A group with an empty period cannot host units observed in that period.
            bad = np.isnan(a)
            a = np.where(bad, 0.0, a)
            dist[:, g] = np.sum(p.Dm * (R - a[:, None]) ** 2, axis=0)
            dist[p.Dm[bad, :].sum(axis=0) > 0, g] = np.inf
        else:
            dist[:, g] = np.sum(p.Dm * (R - a[:, None]) ** 2, axis=0)
    return dist


def gfe_assign(panel, theta, alpha):
    '''Assignment step of Algorithm 1, with the empty-group repair of the
    Fortran code: if a group receives no unit, the unit farthest from its
    assigned profile (taken from a group with more than one member) is moved
    into it.
    '''
    dist = gfe_distances(panel, theta, alpha)
    G = alpha.shape[1]
    # np.argmin returns the FIRST minimum, like Fortran's minloc (and like
    # max.col(..., ties.method = "first") in the R version).
    groups = np.argmin(dist, axis=1)
    best = dist[np.arange(panel.N), groups]
    for g in range(G):
        if not np.any(groups == g):
            size = np.bincount(groups, minlength=G)
            cand = np.flatnonzero(size[groups] > 1)
            i = cand[np.argmax(best[cand])]
            groups[i] = g
            best[i] = 0.0
    return groups


# =============================================================================
# 3. Objective through sufficient statistics (used by the local search)
# =============================================================================
#
# For a grouping gamma, let z_it = (y_it, x_it')', S_gt = sum_{i in g} d_it z_it,
# n_gt = sum_{i in g} d_it, and
#
#     M(gamma) = sum_it d_it z_it z_it' - sum_{g,t} S_gt S_gt' / n_gt .
#
# M is the within-(group, time) cross-product matrix of (y, x). Concentrating
# out theta,
#
#     Q(gamma) = M_yy - M_yx M_xx^{-1} M_xy ,
#
# which equals the GFE objective at the best (theta, alpha) for gamma. Moving
# one unit changes only two groups' S_g, so the effect of a move on M can be
# computed in O(T (K+1)^2) operations. This is the same "objective as a
# function of the partition only" idea behind the matrices AMAT, BMAT1-2, and
# CMAT1-3 in Exact algorithm/Brusco_cov.f90.

def _between(Sg, ng):
    '''Between-cell cross products of one group: sum_t S_gt S_gt' / n_gt.'''
    ok = ng > 0
    if not np.any(ok):
        return np.zeros((Sg.shape[1], Sg.shape[1]))
    Ss = Sg[ok] / np.sqrt(ng[ok])[:, None]
    return Ss.T @ Ss


def gfe_stats(panel, groups, G):
    '''Group sums S (G x T x (K+1)), counts n (G x T), and between matrices B.'''
    p = panel
    K1 = p.K + 1
    S = np.zeros((G, p.T, K1))
    n = np.zeros((G, p.T))
    for g in range(G):
        idx = np.flatnonzero(groups == g)
        if idx.size:
            S[g] = p.Z[idx].sum(axis=0)
            n[g] = p.Dm[:, idx].sum(axis=1)
    B = np.stack([_between(S[g], n[g]) for g in range(G)])
    return {'S': S, 'n': n, 'B': B}


def _obj_from_M(M):
    '''Concentrated objective from the within cross-product matrix M.'''
    K = M.shape[0] - 1
    if K == 0:
        return M[0, 0]
    if K == 1:                                   # closed forms for speed (K = 1, 2)
        if M[1, 1] <= 1e-12 * abs(M[0, 0]):
            return np.inf
        return M[0, 0] - M[1, 0] ** 2 / M[1, 1]
    if K == 2:
        a, b, c = M[1, 1], M[1, 2], M[2, 2]
        det = a * c - b * b
        if det <= 1e-12 * a * c:
            return np.inf
        u, v = M[1, 0], M[2, 0]
        return M[0, 0] - (c * u * u - 2 * b * u * v + a * v * v) / det
    Mxx = M[1:, 1:]
    if np.linalg.cond(Mxx) > 1e12:
        return np.inf
    return M[0, 0] - M[1:, 0] @ np.linalg.solve(Mxx, M[1:, 0])


def gfe_objective(panel, groups, G):
    '''GFE objective Q(gamma) for a given grouping (theta, alpha concentrated out).'''
    st = gfe_stats(panel, np.asarray(groups), G)
    return _obj_from_M(panel.W - st['B'].sum(axis=0))


# =============================================================================
# 4. Algorithm 1: the simple iterative algorithm
# =============================================================================

def _alg1_run(panel, G, theta0=None, alpha0=None, groups0=None, max_iter=1000):
    '''Run Algorithm 1 to convergence from one starting value.

    Steps (paper, Section 2; Supplementary Appendix S.1):
        assignment: g_i = argmin_g sum_t (y_it - x_it' theta - alpha_gt)^2
        update:     (theta, alpha) = OLS of y on x and group x time dummies
    until the grouping no longer changes (equivalently, until the parameter
    change `deltapar` in the original code is exactly zero).

    Either (theta0, alpha0) or a starting grouping groups0 (used inside VNS,
    where Algorithm 1 restarts from a partition) must be given.
    '''
    if groups0 is not None:
        groups = np.asarray(groups0, dtype=int).copy()
        upd = gfe_update(panel, groups, G)
        if not upd['ok']:
            return {'objective': np.inf, 'groups': groups}
        theta, alpha = upd['theta'], upd['alpha']
    else:
        theta, alpha = theta0, alpha0
        groups = np.full(panel.N, -1)
    it = 0
    for it in range(1, max_iter + 1):
        new_groups = gfe_assign(panel, theta, alpha)
        if np.array_equal(new_groups, groups):        # deltapar == 0
            break
        groups = new_groups
        upd = gfe_update(panel, groups, G)
        if not upd['ok']:                             # empty cell: restart
            return {'objective': np.inf, 'groups': groups}
        theta, alpha = upd['theta'], upd['alpha']
    upd = gfe_update(panel, groups, G)
    return {'theta': upd['theta'], 'alpha': upd['alpha'], 'groups': groups,
            'objective': upd['objective'], 'iterations': it}


def _random_start(panel, G, rng, theta_sd=1.0):
    '''Random starting value, as in Data_paths.m and heuristics_sub.f90:
        1. draw theta0 ~ N(0, I_K);
        2. draw G distinct units at random and set
           alpha0_gt = y_{i_g t} - x_{i_g t}' theta0.
    '''
    theta0 = rng.normal(scale=theta_sd, size=panel.K)
    r = panel.y - (panel.X @ theta0 if panel.K > 0 else 0.0)
    R = r.reshape(panel.N, panel.T).T
    centers = rng.choice(panel.N, size=G, replace=False)
    return theta0, R[:, centers].copy()


def gfe_alg1(panel, G, n_starts=1000, theta_sd=1.0, max_iter=1000, seed=None,
             verbose=False):
    '''Algorithm 1 with many random starting values.

    Returns the solution with the lowest objective, together with the vector of
    objectives reached from every start. Inspecting that vector is the check
    the authors recommend in GFE_code.do ("Inspection of this file provides an
    idea of the numerical stability of the solution"): a reliable solution is
    reached from many different starting values.

    `seed` may be an int, None, or a numpy Generator (so callers can share one
    random stream across calls, the counterpart of set.seed() in R).
    '''
    rng = np.random.default_rng(seed)
    best = {'objective': np.inf}
    objs = np.empty(n_starts)
    for s in range(n_starts):
        theta0, alpha0 = _random_start(panel, G, rng, theta_sd)
        fit = _alg1_run(panel, G, theta0=theta0, alpha0=alpha0, max_iter=max_iter)
        objs[s] = fit['objective']
        if fit['objective'] < best['objective']:
            best = fit
        if verbose and (s + 1) % 100 == 0:
            print(f'Algorithm 1: start {s + 1}, best objective {best["objective"]:.8g}')
    return _as_gfe(panel, G, best['groups'], objs, 'Algorithm 1 (iterative)',
                   {'n_starts': n_starts})


# =============================================================================
# 5. Algorithm 2: Variable Neighborhood Search (VNS)
# =============================================================================

def gfe_local_search(panel, groups, G, tol=1e-10):
    '''Local search (Step 5 of Algorithm 2).

    Systematically checks every reassignment of a unit i to another group g
    and keeps it whenever the objective decreases, until no single
    reassignment improves the objective. The resulting grouping is a "local
    minimum" in the sense of the Supplementary Appendix: no single unit can be
    moved profitably, which Algorithm 1 alone does not guarantee.

    For each unit we move it to the best improving group (best-improvement per
    unit). The original Fortran tries groups in order and accepts any
    improvement; both satisfy the stopping rule above.
    '''
    p = panel
    groups = np.asarray(groups, dtype=int).copy()
    st = gfe_stats(p, groups, G)
    S, n, B = st['S'], st['n'], st['B']
    M = p.W - B.sum(axis=0)
    obj = _obj_from_M(M)
    size = np.bincount(groups, minlength=G)
    while True:
        improved = False
        for i in range(p.N):
            g0 = groups[i]
            if size[g0] <= 1:                         # never empty a group
                continue
            zi, di = p.Z[i], p.Dm[:, i]
            Sg0, ng0 = S[g0] - zi, n[g0] - di
            Bg0_new = _between(Sg0, ng0)
            best_obj, best_g, best_parts = obj, g0, None
            for g in range(G):
                if g == g0:
                    continue
                Sg, ng = S[g] + zi, n[g] + di
                Bg_new = _between(Sg, ng)
                Mnew = M + B[g0] + B[g] - Bg0_new - Bg_new
                o = _obj_from_M(Mnew)
                if o < best_obj - tol * max(1.0, abs(best_obj)):
                    best_obj, best_g = o, g
                    best_parts = (Mnew, Sg, ng, Bg_new)
            if best_g != g0:
                g = best_g
                Mnew, Sg, ng, Bg_new = best_parts
                S[g0], n[g0], B[g0] = Sg0, ng0, Bg0_new
                S[g], n[g], B[g] = Sg, ng, Bg_new
                M, obj = Mnew, best_obj
                groups[i] = g
                size[g0] -= 1
                size[g] += 1
                improved = True
        if not improved:
            break
    return {'groups': groups, 'objective': float(obj)}


def _vns_jump(groups, G, n, rng):
    '''Neighborhood jump (Step 3 of Algorithm 2): relocate n randomly selected
    units to n randomly selected (different) groups, never emptying a group.'''
    groups = groups.copy()
    for _ in range(n):
        size = np.bincount(groups, minlength=G)
        movable = np.flatnonzero(size[groups] > 1)
        if movable.size == 0:
            break
        i = rng.choice(movable)
        others = [g for g in range(G) if g != groups[i]]
        groups[i] = rng.choice(others)
    return groups


def gfe_vns(panel, G, n_starts=10, neighmax=10, itermax=10, theta_sd=1.0,
            max_iter=1000, seed=None, verbose=False):
    '''Algorithm 2: Variable Neighborhood Search.

    Supplementary Appendix, Section S.1 (extending Pacheco and Valencia (2003)
    and Brusco and Steinley (2007) to allow for covariates):

      1. From a starting value, obtain an initial grouping gamma*.
         (As in the Fortran code, we run Algorithm 1 to convergence from a
         random start.)  Set j = 0.
      2. Set n = 1.
      3. Neighborhood jump: relocate n random units to n random groups (gamma').
         One update step gives (theta', alpha').
      4. Apply Algorithm 1 from (theta', alpha').
      5. Local search: reassign single units while the objective decreases.
      6. If the objective improves on gamma*, set gamma* to the new grouping
         and go to 2; otherwise set n = n + 1.
      7. If n <= neighmax go to 3; otherwise go to 8.
      8. Set j = j + 1. If j > itermax stop; otherwise go to 2.

    Tuning parameters (Ns; neighmax; itermax): the paper uses (10; 10; 10) in
    the main estimates and (5; 10; 5) for bootstrap and Monte Carlo exercises.
    Rule of thumb from the authors: check that different starting values
    yield the exact same solution.
    '''
    rng = np.random.default_rng(seed)
    tol = 1e-10
    best = {'objective': np.inf}
    objs = np.empty(n_starts)
    for s in range(n_starts):
        # Step 1: initial grouping from Algorithm 1 (one random start).
        theta0, alpha0 = _random_start(panel, G, rng, theta_sd)
        cur = _alg1_run(panel, G, theta0=theta0, alpha0=alpha0, max_iter=max_iter)
        if not np.isfinite(cur['objective']):
            objs[s] = np.inf
            continue
        cur = gfe_local_search(panel, cur['groups'], G)
        j = 0
        while j <= itermax:
            n = 1                                                        # Step 2
            while n <= neighmax:                                         # Step 7
                prop = _vns_jump(cur['groups'], G, n, rng)               # Step 3
                a1 = _alg1_run(panel, G, groups0=prop, max_iter=max_iter)  # Step 4
                if not np.isfinite(a1['objective']):
                    n += 1
                    continue
                ls = gfe_local_search(panel, a1['groups'], G)            # Step 5
                if ls['objective'] < cur['objective'] - tol * max(1.0, abs(cur['objective'])):
                    cur = ls                                             # Step 6
                    n = 1
                else:
                    n += 1
            j += 1                                                       # Step 8
        objs[s] = cur['objective']
        if cur['objective'] < best['objective']:
            best = cur
        if verbose:
            print(f'VNS: start {s + 1}/{n_starts}, objective {cur["objective"]:.10g}, '
                  f'best {best["objective"]:.10g}')
    return _as_gfe(panel, G, best['groups'], objs, 'Algorithm 2 (VNS)',
                   {'n_starts': n_starts, 'neighmax': neighmax, 'itermax': itermax})


# =============================================================================
# 6. Main entry point and the fit object
# =============================================================================

@dataclass
class GFEFit:
    '''Result of a GFE estimation (the "gfe" S3 object of the R version).

    groups are 0-based; use `groups + 1` for the 1..G labels of the paper and
    the original code.
    '''
    theta: np.ndarray
    alpha: np.ndarray            # T x G
    groups: np.ndarray           # (N,) 0-based
    objective: float
    residuals: np.ndarray
    G: int
    N: int
    T: int
    K: int
    group_sizes: np.ndarray
    objectives_by_start: np.ndarray
    algorithm: str
    tuning: dict
    panel: GFEPanel = field(repr=False)

    def theta_series(self):
        '''theta as a labeled pandas Series.'''
        return pd.Series(self.theta, index=self.panel.x_names)

    def alpha_frame(self):
        '''alpha as a labeled T x G DataFrame (columns g1..gG, as in the paper).'''
        return pd.DataFrame(self.alpha, index=self.panel.time_labels,
                            columns=[f'g{g + 1}' for g in range(self.G)])

    def __repr__(self):
        lines = ['Grouped fixed-effects estimator (Bonhomme and Manresa, 2015)',
                 f'  N = {self.N}  T = {self.T}  K = {self.K}  G = {self.G}',
                 f'  computed with: {self.algorithm}',
                 f'  objective: {self.objective:.10g}',
                 f'  group sizes: {list(self.group_sizes)}']
        if self.K > 0:
            lines.append('  theta: ' + ', '.join(f'{nm} = {v:.4f}' for nm, v in
                                                zip(self.panel.x_names, self.theta)))
        ob = self.objectives_by_start
        if len(ob) > 1:
            m = np.min(ob)
            share = np.mean(np.abs(ob - m) < 1e-8 * max(1.0, abs(m)))
            lines.append(f'  starts reaching the best objective: {round(share * len(ob))} '
                         f'of {len(ob)} ({100 * share:.0f}%)')
        return '\n'.join(lines)


def _as_gfe(panel, G, groups, objs, algorithm, tuning):
    groups = np.asarray(groups, dtype=int)
    upd = gfe_update(panel, groups, G)
    return GFEFit(theta=upd['theta'], alpha=upd['alpha'], groups=groups,
                  objective=upd['objective'], residuals=upd['residuals'],
                  G=G, N=panel.N, T=panel.T, K=panel.K,
                  group_sizes=np.bincount(groups, minlength=G),
                  objectives_by_start=np.asarray(objs, dtype=float),
                  algorithm=algorithm, tuning=tuning, panel=panel)


def gfe(panel, G, algorithm='vns', groups=None, **kwargs):
    '''Grouped fixed-effects estimation.

    Parameters
    ----------
    panel : GFEPanel.
    G : number of groups.
    algorithm : 'vns' (Algorithm 2, default) or 'alg1' (Algorithm 1).
    groups : optional 0-based grouping to evaluate directly (no search), e.g.
        the published groupings returned by load_bm_published().
    **kwargs : tuning parameters passed to gfe_vns() or gfe_alg1()
        (n_starts, neighmax, itermax, seed, ...).
    '''
    if algorithm not in ('vns', 'alg1'):
        raise ValueError("algorithm should be 'vns' or 'alg1'")
    if G == 1:
        groups = np.zeros(panel.N, dtype=int)
    if groups is not None:
        groups = np.asarray(groups, dtype=int)
        if groups.min() < 0 or groups.max() >= G:
            raise ValueError('groups should be 0-based labels in 0..G-1 '
                             '(subtract 1 from 1-based labels).')
        upd = gfe_update(panel, groups, G)
        return _as_gfe(panel, G, groups, [upd['objective']], 'given grouping', {})
    if algorithm == 'vns':
        return gfe_vns(panel, G, **kwargs)
    return gfe_alg1(panel, G, **kwargs)


def gfe_relabel(fit, order_by=None):
    '''Relabel groups in order of a chosen criterion (by default, the
    first-period value of alpha), so that solutions from different runs can be
    compared. Group labels are arbitrary in the GFE problem.'''
    order_by = fit.alpha[0, :] if order_by is None else np.asarray(order_by)
    o = np.argsort(order_by, kind='stable')
    mapping = np.empty(fit.G, dtype=int)
    mapping[o] = np.arange(fit.G)
    groups = mapping[fit.groups]
    return _as_gfe(fit.panel, fit.G, groups, fit.objectives_by_start,
                   fit.algorithm, fit.tuning)
