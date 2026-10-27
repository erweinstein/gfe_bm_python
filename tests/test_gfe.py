'''tests/test_gfe.py -- Unit tests for the Python translation of
Bonhomme and Manresa (2015), "Grouped Patterns of Heterogeneity in Panel
Data," Econometrica 83(3), 1147-1184.

Each test checks one building block against an independent computation (brute
force, a direct least-squares fit, a direct matrix formula) or against a
number published in the paper. Run from the project folder with

    python -m pytest tests

Validation against all the published tables is in
scripts/01_validate_against_published.py.

Python translation Copyright (c) 2026 Eliot R. Weinstein. Licensed under the
BSD 3-Clause License; see LICENSE and NOTICE.md (the license covers the
translation only, not Bonhomme and Manresa's method or original code).
'''

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import (gfe_panel, gfe_update, gfe_objective, gfe_local_search, gfe,  # noqa: E402
                 gfe_exact_bb, gfe_bruteforce, gfe_vcov_largeT, gfe_within,
                 gfe_twolayer, match_groups, load_bm_panel, load_bm_published)


def sim_panel(N, T, G, K=2, seed=1, unbalanced=False):
    '''A small simulated GFE panel with K covariates.'''
    rng = np.random.default_rng(seed)
    g0 = rng.integers(0, G, size=N)
    a0 = rng.normal(scale=1.5, size=(T, G))
    X = rng.normal(size=(N * T, K))
    y = X @ np.linspace(0.5, -0.3, K) + a0[np.tile(np.arange(T), N), np.repeat(g0, T)] \
        + rng.normal(size=N * T)
    d = (rng.uniform(size=N * T) > 0.15).astype(float) if unbalanced else None
    return gfe_panel(y, X, N, T, d=d)


def dummies(labels):
    '''Indicator matrix for an integer label vector.'''
    _, inv = np.unique(labels, return_inverse=True)
    return np.eye(inv.max() + 1)[inv]


# 1. The sufficient-statistics objective equals the OLS objective ------------
@pytest.mark.parametrize('draw', range(5))
def test_objective_matches_update(draw):
    p = sim_panel(40, 6, 3)
    g = np.random.default_rng(draw).integers(0, 3, size=p.N)
    assert abs(gfe_objective(p, g, 3) - gfe_update(p, g, 3)['objective']) < 1e-8


# 2. The update step is OLS on group x time dummies (unbalanced panel) -------
def test_update_is_ols_unbalanced():
    p = sim_panel(30, 5, 3, unbalanced=True)
    g = np.random.default_rng(0).integers(0, 3, size=p.N)
    up = gfe_update(p, g, 3)
    obs = p.d == 1
    D = dummies(g[p.unit] * p.T + p.time)[obs]
    Xfull = np.hstack([p.X[obs], D])
    coef, *_ = np.linalg.lstsq(Xfull, p.y[obs], rcond=None)
    e = p.y[obs] - Xfull @ coef
    assert np.max(np.abs(up['theta'] - coef[:2])) < 1e-8
    assert abs(up['objective'] - np.sum(e ** 2)) < 1e-8


# 3. Branch and bound = brute force; VNS finds the same optimum -------------
@pytest.mark.parametrize('G', [2, 3])
def test_exact_and_vns_reach_global_minimum(G):
    p = sim_panel(9, 4, G, seed=10 + G)
    bf = gfe_bruteforce(p, G)
    ex = gfe_exact_bb(p, G, seed=1)
    vn = gfe(p, G, algorithm='vns', n_starts=5, neighmax=5, itermax=5, seed=1)
    assert abs(ex.objective - bf['objective']) < 1e-8
    assert abs(vn.objective - bf['objective']) < 1e-8


# 4. Local search never increases the objective, and ends at a local minimum -
def test_local_search():
    p = sim_panel(40, 6, 3, seed=3)
    g = np.random.default_rng(3).integers(0, 3, size=p.N)
    ls = gfe_local_search(p, g, 3)
    assert ls['objective'] <= gfe_objective(p, g, 3) + 1e-10
    for i in range(p.N):
        for h in range(3):
            if h == ls['groups'][i]:
                continue
            g2 = ls['groups'].copy()
            g2[i] = h
            if np.all(np.bincount(g2, minlength=3) > 0):
                assert gfe_objective(p, g2, 3) >= ls['objective'] - 1e-9


# 5. Hungarian matching = minimum over all G! relabelings --------------------
@pytest.mark.parametrize('draw', range(5))
def test_match_groups_is_optimal(draw):
    rng = np.random.default_rng(draw)
    G = 5
    truth = rng.integers(0, G, size=60)
    est = np.where(rng.uniform(size=60) < 0.3, rng.integers(0, G, size=60),
                   rng.permutation(G)[truth])
    brute = min(np.mean(np.array(pm)[est] != truth) for pm in itertools.permutations(range(G)))
    assert abs(match_groups(est, truth, G)['misclassification'] - brute) < 1e-12


# 6. Large-T clustered variance = direct sandwich formula --------------------
def test_vcov_largeT_sandwich():
    p = sim_panel(50, 5, 2, seed=4)
    fit = gfe(p, 2, algorithm='alg1', n_starts=50, seed=4)
    D = np.zeros((p.N * p.T, 2 * p.T))
    D[np.arange(p.N * p.T), fit.groups[p.unit] * p.T + p.time] = 1
    Xt = np.hstack([D, p.X])
    coef, *_ = np.linalg.lstsq(Xt, p.y, rcond=None)
    e = p.y - Xt @ coef
    Bread = np.linalg.inv(Xt.T @ Xt)
    M = sum(np.outer(Xt[r].T @ e[r], Xt[r].T @ e[r])
            for r in (np.arange(p.T) + i * p.T for i in range(p.N)))
    V = Bread @ M @ Bread * (p.N * p.T) / (p.N * p.T - 2 * p.T - p.K)
    assert np.max(np.abs(gfe_vcov_largeT(fit) - V)) < 1e-10


# 7. Within transformation: G = 1 GFE in deviations = two-way fixed effects --
def test_within_is_twoway_fe():
    p = sim_panel(30, 6, 2, seed=5)
    fe = gfe(gfe_within(p), 1)
    Xfull = np.hstack([p.X, dummies(p.unit), dummies(p.time)[:, 1:]])
    coef, *_ = np.linalg.lstsq(Xfull, p.y, rcond=None)
    assert np.max(np.abs(fe.theta - coef[:2])) < 1e-8


# 8. Two-layer model: constrained LS = OLS on its own parametrization --------
def test_twolayer_constrained_ls():
    p = sim_panel(36, 5, 3, seed=6)
    H = [2, 2, 1]
    rng = np.random.default_rng(6)
    sg = rng.integers(0, sum(H), size=p.N)
    while len(np.unique(sg)) < sum(H):
        sg = rng.integers(0, sum(H), size=p.N)
    tl = gfe_twolayer(p, H, groups=sg)
    l1 = np.repeat(np.arange(len(H)), H)[sg]
    Xfull = np.hstack([p.X, dummies(l1[p.unit] * p.T + p.time), dummies(sg[p.unit])])
    coef, *_ = np.linalg.lstsq(Xfull, p.y, rcond=None)
    e = p.y - Xfull @ coef
    assert np.max(np.abs(tl['theta'] - coef[:2])) < 1e-8
    assert abs(tl['objective'] - np.sum(e ** 2)) < 1e-8


# 9. Published numbers: Table S.XI objectives for the authors' groupings ----
def test_published_objectives():
    p = load_bm_panel()
    pub = load_bm_published()['groupings']
    for G, obj in {2: 19.84685, 3: 16.59873, 4: 14.31867, 10: 7.74906}.items():
        assert abs(gfe(p, G, groups=pub[f'G{G}']).objective - obj) < 5e-6
