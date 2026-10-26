'''gfe/utils.py -- Helpers: group matching, group means, plots, linear algebra.

R translation Copyright (C) 2026 Eliot R. Weinstein; Python translation
Copyright (C) 2026 Eliot R. Weinstein. Licensed under the GNU GPL, version 2 or
(at your option) any later version; see LICENSE. The license covers the
translation only, not Bonhomme and Manresa's method or original code.
'''

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Matching estimated groups to true (or published) groups
# =============================================================================

def match_groups(est, true, G=None):
    '''Relabel estimated groups to best match true groups.

    Group labels are arbitrary, so comparing two groupings means finding the
    relabeling that maximizes agreement. The paper takes the minimum
    misclassification over all G! permutations (over 500,000 random ones when
    G = 10). This is an assignment problem, which the Hungarian algorithm
    solves exactly; in R I wrote it out by hand, but in Python scipy already
    provides it as scipy.optimize.linear_sum_assignment, so there is no reason
    to reinvent it.

    Parameters
    ----------
    est, true : (N,) arrays of 0-based labels.

    Returns
    -------
    dict with groups (est relabeled), map (map[old] = new), and
    misclassification (share of units with relabeled group != true group).
    '''
    est = np.asarray(est, dtype=int)
    true = np.asarray(true, dtype=int)
    if G is None:
        G = int(max(est.max(), true.max())) + 1
    overlap = np.zeros((G, G))
    np.add.at(overlap, (est, true), 1)
    row, col = linear_sum_assignment(-overlap)       # maximize agreement
    mapping = np.empty(G, dtype=int)
    mapping[row] = col
    relabeled = mapping[est]
    return {'groups': relabeled, 'map': mapping,
            'misclassification': float(np.mean(relabeled != true))}


# =============================================================================
# Group averages of the data (MYbar_gt and MXbar_gt in GFE_estimates.m)
# =============================================================================

def group_means(fit, v, groups=None, G=None):
    '''T x G array of group-by-period means of a variable stacked like y.

    `fit` only needs .panel; `groups`/`G` default to fit.groups/fit.G, so the
    same function serves the extensions (two-layer, heterogeneous slopes).'''
    p = fit.panel
    groups = fit.groups if groups is None else np.asarray(groups)
    G = fit.G if G is None else G
    M = np.asarray(v, dtype=float).reshape(p.N, p.T).T
    D = p.Dm
    out = np.empty((p.T, G))
    for g in range(G):
        idx = groups == g
        out[:, g] = (M * D)[:, idx].sum(axis=1) / D[:, idx].sum(axis=1)
    return out


# =============================================================================
# Figures in the style of the paper (black, white, and grays)
# =============================================================================

def plot_profiles(ax, M, years, ylabel='time effects', xlabel='years', ylim=None,
                  colors=None, linestyles=None, lw=3, labels=None):
    '''Plot T x G profiles (group time effects, or group averages of a
    variable) on a matplotlib Axes, in the grayscale line style of
    GFE_estimates.m (thick black and gray lines, solid / dashed / dash-dot).'''
    G = M.shape[1]
    grays = [0, 0.75, 0.4, 0.6, 0.2, 0.5, 0.3, 0.65, 0.1, 0.85]
    if colors is None:
        colors = [str(grays[g % 10]) for g in range(G)]      # matplotlib reads '0.4' as gray
    if linestyles is None:
        linestyles = [['-', '--', '-.'][g % 3] for g in range(G)]
    for g in range(G):
        ax.plot(years, M[:, g], color=colors[g], linestyle=linestyles[g], lw=lw,
                label=None if labels is None else labels[g])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


def plot_by_G(ax, Gs, est, se, ylabel, ylim=None):
    '''Parameter estimates against the number of groups, with +/- 1.96 se
    bands (the "Figure of parameter estimates" block in GFE_estimates.m).'''
    est, se = np.asarray(est), np.asarray(se)
    ax.plot(Gs, est, 'o-', color='black', lw=3, mfc='white', ms=8)
    ax.plot(Gs, est - 1.96 * se, ':', color='black', lw=2)
    ax.plot(Gs, est + 1.96 * se, ':', color='black', lw=2)
    ax.set_xticks(list(Gs))
    ax.set_xlabel('number of groups G')
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    return ax


# =============================================================================
# Linear algebra and regression helpers
# =============================================================================

def pinv(A):
    '''Moore-Penrose pseudo-inverse (MATLAB pinv). numpy's default tolerance
    rule is the same as MATLAB's: max(size(A)) * eps(norm(A)).'''
    return np.linalg.pinv(A, rcond=max(A.shape) * np.finfo(float).eps)


def ols_iv_cluster(y, X, Z=None, cluster=None, small='none'):
    '''OLS / 2SLS with standard errors clustered by `cluster`.

    2SLS: y on X (exogenous and endogenous regressors), instruments Z
    (exogenous regressors plus excluded instruments). With Z = None this is OLS.

    small : small-sample factor applied to the cluster-robust variance:
        'none' (Stata's ivregress, vce(cluster), without the `small` option;
        this reproduces the standard errors in Table S.XV), 'clusters'
        (M/(M-1)), or 'regress' (M/(M-1) * (n-1)/(n-k), as Stata's regress,
        vce(cluster)).
    '''
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = X if Z is None else np.asarray(Z, dtype=float)
    Pz_X = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)       # first-stage fitted X
    b = np.linalg.solve(Pz_X.T @ X, Pz_X.T @ y)
    e = y - X @ b
    bread = np.linalg.inv(Pz_X.T @ Pz_X)
    # Cluster sums of the scores: group rows by cluster label (R's rowsum()).
    labels, inv = np.unique(np.asarray(cluster), return_inverse=True)
    score = np.zeros((len(labels), X.shape[1]))
    np.add.at(score, inv, Pz_X * e[:, None])
    M, n, k = len(labels), len(y), X.shape[1]
    adj = {'none': 1.0, 'clusters': M / (M - 1),
           'regress': M / (M - 1) * (n - 1) / (n - k)}[small]
    V = bread @ (score.T @ score) @ bread * adj
    return {'coef': b, 'vcov': V, 'se': np.sqrt(np.diag(V)), 'residuals': e,
            'n_clusters': M}
