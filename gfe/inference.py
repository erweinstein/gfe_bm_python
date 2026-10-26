'''gfe/inference.py -- Standard errors for the GFE estimator.

Python translation of the inference procedures in Bonhomme and Manresa (2015):

  (1) Large-N,T variance, clustered at the unit level (Arellano, 1987):
      GFE_estimates.m ("Large - T variance") and Supplementary Appendix
      equations (S.2)-(S.3). Group membership is treated as known, which
      Corollary 1 of the paper justifies as N and T grow.

  (2) Large-N, fixed-T variance: fixedT_function.m, "an extension of
      Pollard's (1982) fixed-T variance formula" to models with covariates
      (Supplementary Appendix S.2.2, equations (S.7)-(S.16)). It adds the
      contribution of units near the frontier between two groups, which the
      large-T formula ignores.

  (3) Bootstrap: resampling unit-specific blocks (y_i, x_i) with replacement
      and re-estimating the GFE model in each bootstrap sample (Supplementary
      Appendix S.2.2.2; Bootstrap_version.exe / MC_bootstrap.exe).

The original code is by Stephane Bonhomme and Elena Manresa.

R translation Copyright (C) 2026 Eliot R. Weinstein; Python translation
Copyright (C) 2026 Eliot R. Weinstein. Licensed under the GNU GPL, version 2 or
(at your option) any later version; see LICENSE. The license covers the
translation only, not Bonhomme and Manresa's method or original code.
'''

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

from .core import gfe, gfe_panel


def _gfe_design(fit):
    '''Regressor matrix of the "known groups" regression: [group x time
    dummies, X] (Xtot = [kron(gi, eye(T)) X] in GFE_estimates.m). Columns:
    alpha_{g,t} in group-major order (g1t1, ..., g1tT, g2t1, ...), then theta.'''
    p = fit.panel
    T, G = p.T, fit.G
    D = np.zeros((p.N * T, G * T))
    D[np.arange(p.N * T), fit.groups[p.unit] * T + p.time] = 1.0
    D *= p.d[:, None]
    return np.hstack([D, p.X])


def _theta_index(fit):
    return np.arange(fit.G * fit.T, fit.G * fit.T + fit.K)


def gfe_vcov_largeT(fit):
    '''Large-T variance clustered at the unit level (GFE_estimates.m):

        V = (Xtot'Xtot)^{-1} [ sum_i Xtot_i' e_i e_i' Xtot_i ] (Xtot'Xtot)^{-1}
              * NT / (NT - GT - K),

    with e the GFE residuals. The degrees-of-freedom factor is the one used in
    the original code. Returns the full (GT + K) x (GT + K) matrix (NaN rows
    and columns for empty group-period cells); the theta block is the
    large-N,T variance of theta_hat in (S.3).
    '''
    p = fit.panel
    Xtot = _gfe_design(fit)
    coef = np.concatenate([fit.alpha.T.ravel(), fit.theta])   # group-major alpha, then theta
    e = p.d * (p.y - Xtot @ np.nan_to_num(coef))
    keep = np.any(Xtot != 0, axis=0)             # drop empty group-period cells
    Xk = Xtot[:, keep]
    bread = np.linalg.inv(Xk.T @ Xk)
    # Meat: sum over units of (X_i' e_i)(X_i' e_i)'. Summing the rows of
    # Xk * e within each unit is R's rowsum(); with the stacked layout it is
    # just a reshape to (N, T, columns) and a sum over T.
    score = (Xk * e[:, None]).reshape(p.N, p.T, -1).sum(axis=1)
    meat = score.T @ score
    nobs = p.d.sum()
    V = bread @ meat @ bread * nobs / (nobs - fit.G * p.T - p.K)
    out = np.full((Xtot.shape[1], Xtot.shape[1]), np.nan)
    out[np.ix_(keep, keep)] = V
    return out


def gfe_se_theta(fit, V):
    '''Standard errors of theta from a variance matrix.'''
    k = _theta_index(fit)
    return np.sqrt(np.diag(V)[k])


def gfe_delta(fit, V, grad):
    '''Delta-method variance of a smooth function h(theta), e.g. the cumulative
    income effect theta_2 / (1 - theta_1) in the application. `grad` is the
    gradient of h at theta_hat (length K).'''
    k = _theta_index(fit)
    grad = np.asarray(grad, dtype=float)
    return float(grad @ V[np.ix_(k, k)] @ grad)


# =============================================================================
# Fixed-T variance (fixedT_function.m)
# =============================================================================

def gfe_vcov_fixedT(fit, eps):
    '''Large-N, fixed-T variance of (alpha, theta): Gamma^{-1} V Gamma^{-1}.

    Faithful translation of fixedT_function.m. With a_g the T-vector of group
    g's time effects and r_i = y_i - x_i theta (a T-vector):

      * frontier weight of unit i between groups g and h (g != h):
            w_igh = phi( u_igh / eps ) / eps
                    * 1{ g and h are both at least as close to r_i as every other group l },
            u_igh = (a_h - a_g)' (r_i - (a_g + a_h)/2) / ||a_h - a_g||,
        i.e. a Gaussian kernel in the signed distance from r_i to the
        hyperplane equidistant from a_g and a_h (I_igh(eps_N) in (S.13)-(S.16)).
      * Gamma_xx = X'X/N - (1/2N) sum_{g != h} sum_i w_igh x_i' (a_h-a_g)(a_h-a_g)' x_i / ||a_h-a_g||
      * Gamma_xg = (1/N) sum_{i in g} x_i' + (1/N) sum_{h != g} sum_i w_igh x_i'(a_g-a_h)(r_i-a_g)' / ||a_h-a_g||
      * Gamma_gg = (n_g/N) I_T - (1/N) sum_{h != g} sum_i w_igh (r_i-a_g)(r_i-a_g)' / ||a_h-a_g||
      * Gamma_gh = (1/N) sum_i w_igh (r_i-a_g)(r_i-a_h)' / ||a_h-a_g||       (g != h)
      * V = sum_i W_i' e_i e_i' W_i with W_i = [g-dummies (x) I_T, x_i] (clustered).

    As in the original, Gamma is multiplied by N and V is not divided by N, so
    Gamma^{-1} V Gamma^{-1} is directly the variance of the estimator.

    Notes on the original code, kept as is:
      - the "1/2" in Gamma_xx follows equation (S.13);
      - the blocks Gamma_gh are filled for g < h and then symmetrized;
      - alpha is recomputed from the given grouping and theta (delta_aux).

    eps : bandwidth epsilon_N (kernel = standard normal density).
    '''
    p = fit.panel
    if not p.balanced:
        raise ValueError('gfe_vcov_fixedT() is written for balanced panels, as in the original.')
    N, T, K, G = p.N, p.T, p.K, fit.G
    theta, groups = fit.theta, fit.groups
    r = p.y - (p.X @ theta if K > 0 else 0.0)
    R = r.reshape(N, T).T                                  # T x N, column i: r_i
    a = np.column_stack([R[:, groups == g].mean(axis=1) for g in range(G)])   # T x G
    Xu = p.X.reshape(N, T, K)                              # Xu[i] = x_i (T x K)

    # Squared distances of every unit to every group profile.
    dist = np.column_stack([np.sum((R - a[:, [g]]) ** 2, axis=0) for g in range(G)])

    # Frontier weights w[i, g, h].
    w = np.zeros((N, G, G))
    for g in range(G):
        for h in range(G):
            if h == g:
                continue
            ind = np.ones(N)
            for l in range(G):
                if l != g and l != h:
                    ind *= (dist[:, g] <= dist[:, l]) * (dist[:, h] <= dist[:, l])
            dvec = a[:, h] - a[:, g]
            nrm = np.sqrt(np.sum(dvec ** 2))
            u = (dvec / nrm) @ (R - ((a[:, g] + a[:, h]) / 2)[:, None])
            w[:, g, h] = ind * norm.pdf(u / eps) / eps

    # Gamma blocks.
    G_xx = p.X.T @ p.X / N if K > 0 else np.zeros((0, 0))
    G_xg = np.zeros((G, K, T))
    G_gg = np.zeros((G, T, T))
    G_gh = np.zeros((G, G, T, T))
    for g in range(G):
        G_gg[g] = np.sum(groups == g) * np.eye(T) / N
        if K > 0:
            G_xg[g] = Xu[groups == g].sum(axis=0).T / N
    for g in range(G):
        for h in range(G):
            if h == g:
                continue
            ag, ah = a[:, g], a[:, h]
            nrm = np.sqrt(np.sum((ah - ag) ** 2))
            for i in np.flatnonzero(w[:, g, h] != 0):
                wi = w[i, g, h]
                ri = R[:, i]
                if K > 0:
                    x = Xu[i]
                    G_xx = G_xx + wi * x.T @ (-0.5 * np.outer(ah - ag, ah - ag)) @ x / nrm / N
                    G_xg[g] = G_xg[g] + wi * x.T @ np.outer(ag - ah, ri - ag) / nrm / N
                G_gg[g] = G_gg[g] - wi * np.outer(ri - ag, ri - ag) / nrm / N
                G_gh[g, h] = G_gh[g, h] + wi * np.outer(ri - ag, ri - ah) / nrm / N

    # Assemble BigGamma ((GT + K) x (GT + K)): alpha blocks first, then theta.
    dimG = G * T + K
    BG = np.zeros((dimG, dimG))
    kk = slice(G * T, G * T + K)

    def idx(g):
        return slice(g * T, (g + 1) * T)

    if K > 0:
        BG[kk, kk] = G_xx
        for g in range(G):
            BG[idx(g), kk] = G_xg[g].T
    for g in range(G):
        for h in range(g, G):
            BG[idx(g), idx(h)] = G_gg[g] if g == h else G_gh[g, h]
    lower = np.tril_indices(dimG, -1)
    BG[lower] = BG.T[lower]                     # symmetrize from the upper triangle
    BG = N * BG

    # Clustered "meat" V = Xtot' Omega Xtot (Omega block-diagonal by unit).
    Xtot = _gfe_design(fit)
    e = p.y - Xtot @ np.concatenate([a.T.ravel(), theta])
    score = (Xtot * e[:, None]).reshape(N, T, -1).sum(axis=1)
    Vm = score.T @ score

    BGinv = np.linalg.inv(BG)
    return BGinv @ Vm @ BGinv


def gfe_bandwidth(fit, snap_to_grid=False, grid=np.linspace(0.05, 0.15, 10)):
    '''Bandwidth rule for the fixed-T variance (GFE_estimates.m; footnote 10 of
    the Supplementary Appendix): a Silverman-type rule of thumb,

        eps_N = 1.06 * min( sigma_Y, IQR(y - x theta) / 1.34 ) * N^{-1/5},

    where sigma_Y is the smallest, over pairs of groups (g, h), of the standard
    deviation of the residual paths projected on the unit normal vector
    (a_g - a_h) / ||a_g - a_h||.

    GFE_estimates.m, as distributed, then picks the value closest to eps_N on
    the grid linspace(0.05, 0.15, 10) (snap_to_grid=True). The fixed-T
    standard errors published in Table S.XI are reproduced by the unsnapped
    rule (the default here).
    '''
    p = fit.panel
    T, N, G = p.T, p.N, fit.G
    r = p.y - (p.X @ fit.theta if p.K > 0 else 0.0)
    R = r.reshape(N, T).T
    sig = []
    for g in range(G - 1):
        for h in range(g + 1, G):
            nv = fit.alpha[:, g] - fit.alpha[:, h]
            nv = nv / np.sqrt(np.sum(nv ** 2))
            sig.append(np.std(nv @ R, ddof=1))          # ddof=1, like R's sd() and MATLAB's std()
    sig_Y = min(sig) if sig else np.std(r, ddof=1)
    # MATLAB's quantile() uses the "Hazen" definition (R's type = 5).
    q25, q75 = np.percentile(r, [25, 75], method='hazen')
    bw = 1.06 * min(sig_Y, (q75 - q25) / 1.34) * N ** (-1 / 5)
    if snap_to_grid:
        return float(grid[np.argmin(np.abs(bw - grid))])
    return float(bw)


# =============================================================================
# Bootstrap
# =============================================================================

def _one_bootstrap(args):
    '''One bootstrap replication (top-level so it can run in a worker process).'''
    p, G, algorithm, units, fun, seed, kwargs = args
    T = p.T
    rows = (units[:, None] * T + np.arange(T)[None, :]).ravel()
    pb = gfe_panel(p.y[rows], p.X[rows], N=p.N, T=T, d=p.d[rows], x_names=p.x_names)
    fb = gfe(pb, G, algorithm=algorithm, seed=seed, **kwargs)
    extra = [] if fun is None else list(np.atleast_1d(fun(fb.theta)))
    return np.concatenate([fb.theta, extra])


def gfe_bootstrap(fit, B=100, algorithm='vns', fun=None, seed=None, cores=1, **kwargs):
    '''Unit-level (block) bootstrap of the GFE estimator.

    Resamples units (y_i, x_i) with replacement, re-estimates the GFE model in
    each bootstrap sample, and returns the draws of theta (and of any function
    of theta supplied in `fun`). Standard errors are the standard deviations
    of the draws (as std_cluster_b in GFE_estimates.m); percentile intervals
    use the 2.5% and 97.5% quantiles (Processing_bootstrap_G10_dgp1.m).

    Bootstrap consistency for minimum sum-of-squares partitioning is due to
    Arcones and Gine (1992), building on Pollard (1982); see Section S.2.2.2.

    The paper's bootstrap used Algorithm 2 with (5; 10; 5) in the application,
    and Algorithm 1 with 1,000 starts in the Monte Carlo exercises.

    cores > 1 runs replications in worker processes. On Windows, call this
    from inside an `if __name__ == "__main__":` block (the scripts do). Each
    replication gets its own child seed, so results do not depend on `cores`.
    `fun` must then be a top-level (picklable) function, not a lambda.
    '''
    p = fit.panel
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    draws_units = [rng.integers(0, p.N, size=p.N) for _ in range(B)]
    child_seeds = ss.spawn(B)
    jobs = [(p, fit.G, algorithm, u, fun, cs, kwargs) for u, cs in zip(draws_units, child_seeds)]
    if cores > 1:
        with ProcessPoolExecutor(max_workers=cores) as ex:
            res = list(ex.map(_one_bootstrap, jobs))
    else:
        res = [_one_bootstrap(j) for j in jobs]
    draws = np.vstack(res)
    return {'draws': draws,
            'se': draws.std(axis=0, ddof=1),
            'percentile': np.percentile(draws, [2.5, 97.5], axis=0),
            'B': B, 'algorithm': algorithm}
