'''gfe/exact.py -- Exact GFE solution by repetitive branch and bound.

Python translation of Bonhomme and Manresa's extension of Brusco's (2006)
repetitive branch-and-bound algorithm for minimum within-cluster sums of
squares partitioning, modified to allow for covariates:

    Original: Bonhomme_Manresa_codes/Application/Exact algorithm/Brusco_cov.f90
              (with heuristics_sub.f90 for the starting heuristic and the
               COMPUTE_MATRIX subroutine for the pairwise matrices)
    Method:   Brusco, M. J. (2006), "A Repetitive Branch-and-Bound Procedure
              for Minimum Within-Cluster Sums of Squares Partitioning,"
              Psychometrika 71(2), 347-363.
    Use:      Supplementary Appendix of Bonhomme and Manresa (2015), Section
              S.1.2 and Table S.I: the exact solution certifies that the
              heuristics found the global minimum.

Speed. The authors ran this as compiled Fortran, and in R I needed C++ (Rcpp)
to get past N of about 20. In Python, the stage loop below is written in a
deliberately "Fortran-like" style (scalar loops over plain numpy arrays, no
Python objects) so that numba can compile it to machine code when numba is
installed (`pip install numba`). Without numba the same function runs as
ordinary Python: correct, and fine for the small problems in the tests, but
far too slow for N = 30.

Key identity. For a grouping gamma with n_g units in group g, the
within-group sum of squares of any T-vectors w_i equals

    sum_g sum_{i in g} || w_i - wbar_g ||^2 = sum_g (1/n_g) sum_{i<j in g} || w_i - w_j ||^2 .

Applying it to w_i = y_i - x_i theta and concentrating out theta gives the GFE
objective as a function of PAIRWISE quantities only (balanced panels):

    A_ij      = sum_t (y_it - y_jt)^2
    B_ij[k]   = sum_t (y_it - y_jt)(x_itk - x_jtk)
    C_ij[k,l] = sum_t (x_itk - x_jtk)(x_itl - x_jtl)

    a = sum_g (1/n_g) sum_{i<j in g} A_ij,  b and C defined alike,
    Q(gamma) = a - b' C^{-1} b .

These are the matrices AMAT, BMAT1-2, CMAT1-3 read by Brusco_cov.f90.

Why the bound is valid with covariates. Restricting a grouping to a subset of
units can only lower the within sum of squares for every theta, and
min_theta [f1(theta) + f2(theta)] >= min f1 + min f2. Hence, if the first P
units are assigned, the objective of ANY completion is at least
Q(partial assignment of the first P units) + Q*(remaining units), where
Q*(remaining) is the optimum for the remaining units on their own. The
"repetitive" part of Brusco's method computes these Q*(.) by solving the
problem for the last G+1, G+2, ..., N units in turn.

R translation Copyright (C) 2026 Eliot R. Weinstein; Python translation
Copyright (C) 2026 Eliot R. Weinstein. Licensed under the GNU GPL, version 2 or
(at your option) any later version; see LICENSE. The license covers the
translation only, not Bonhomme and Manresa's method or original code.
'''

import itertools
import time as _time

import numpy as np

from .core import gfe_panel, gfe_objective, gfe_alg1, gfe_vns, _as_gfe

try:                                    # optional speed-up
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                     # fall back to plain Python
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        '''No-op stand-in for numba.njit, so the decorated code still runs.'''
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


# =============================================================================
# The stage loop (one value of E), in the style of the Fortran original
# =============================================================================

@njit(cache=True)
def _partial_obj(v, K):
    '''Concentrated partial objective a - b' C^{-1} b from v = (a, b, vec(C)).
    Returns NaN if C is (numerically) singular; the Fortran code skips the
    bound test in that case. (Fortran tests ABS(DISC) > 1*10**(-6); with
    integer arithmetic 10**(-6) evaluates to 0 there, so the effective test is
    DISC != 0. We use a small relative tolerance instead.)'''
    if K == 0:
        return v[0]
    if K == 1:
        if v[2] > 1e-12:
            return v[0] - v[1] * v[1] / v[2]
        return np.nan
    if K == 2:
        c11, c21, c22 = v[3], v[4], v[6]
        dt = c11 * c22 - c21 * c21
        if not (dt > 1e-12 * c11 * c22):
            return np.nan
        return v[0] - (c22 * v[1] * v[1] - 2.0 * c21 * v[1] * v[2] + c11 * v[2] * v[2]) / dt
    # General K: Gaussian elimination with partial pivoting on [C | b]
    # (written out by hand, as in the C++ version, so numba can compile it).
    C = np.empty((K, K))
    b = np.empty(K)
    for k in range(K):
        b[k] = v[1 + k]
        for l in range(K):
            C[k, l] = v[1 + K + l * K + k]
    bw = b.copy()
    scale = 0.0
    for k in range(K):
        scale = max(scale, abs(C[k, k]))
    for c in range(K):
        piv = c
        best = abs(C[c, c])
        for r in range(c + 1, K):
            if abs(C[r, c]) > best:
                best = abs(C[r, c])
                piv = r
        if best <= 1e-12 * scale:
            return np.nan
        if piv != c:
            for l in range(K):
                tmp = C[c, l]
                C[c, l] = C[piv, l]
                C[piv, l] = tmp
            tmp = bw[c]
            bw[c] = bw[piv]
            bw[piv] = tmp
        for r in range(c + 1, K):
            f = C[r, c] / C[c, c]
            for l in range(c, K):
                C[r, l] -= f * C[c, l]
            bw[r] -= f * bw[c]
    sol = np.zeros(K)
    for r in range(K - 1, -1, -1):
        s = bw[r]
        for l in range(r + 1, K):
            s -= C[r, l] * sol[l]
        sol[r] = s / C[r, r]
    q = 0.0
    for k in range(K):
        q += b[k] * sol[k]
    return v[0] - q


@njit(cache=True)
def _bb_stage(PW, E, G, K, Z, opt):
    '''One stage of the branch and bound.

    The steps and their numbers follow Brusco_cov.f90 (GO TO 100, 200, ...).
    Objects 1..E of the subproblem are assigned in order; clusters are opened
    in order 1, 2, ..., G, which removes label symmetry.

    Translation note: inside this function I deliberately keep Fortran's
    1-based object depth P and cluster label M (arrays are sized E+1 and G+1
    and element 0 is unused). That makes the code match the original line by
    line, at the cost of a little memory, and it is the one place in the
    package where 1-based labels are used. The result XB is converted to
    0-based labels before it is returned.

    PW : (E, E, Qd) pairwise matrices for the objects of this stage.
    opt : opt[r] = optimum for the last r objects (BUP2 in the Fortran code).
    Returns (Z, found, XB, nodes).
    '''
    Qd = 1 + K + K * K
    P = 0
    Q = G                          # number of still-empty clusters
    M = 1
    n = np.zeros(G + 1, dtype=np.int64)
    x = np.zeros(E + 1, dtype=np.int64)
    XB = np.zeros(E, dtype=np.int64)
    sums = np.zeros((G + 1, Qd))   # cluster sums (SUMA, SUMB1-2, SUMC1-3)
    v = np.zeros(Qd)
    nodes = 0
    have = False
    step = 1
    while True:
        if step == 1:                             # STEP 1: increment search depth
            P += 1
            M = 1
            n[M] += 1
            if n[M] == 1:
                Q -= 1
            x[P] = M
            for j in range(1, P):                 # add pairs (P, j), j in cluster M
                if x[j] == M:
                    for q in range(Qd):
                        sums[M, q] += PW[P - 1, j - 1, q]
            step = 2
        elif step == 2:                           # STEP 2: feasibility
            step = 5 if E - P < Q else 3
        elif step == 3:                           # STEP 3: bound test
            nodes += 1
            used = G - Q
            for q in range(Qd):
                s = 0.0
                for m in range(1, used + 1):
                    s += sums[m, q] / n[m]
                v[q] = s
            ZB = _partial_obj(v, K)
            if (not np.isnan(ZB)) and Z <= ZB + opt[E - P]:
                step = 5
                continue
            if P != E:                            # STEP 4: update incumbent
                step = 1
                continue
            if not np.isnan(ZB):
                Z = ZB
                for j in range(E):
                    XB[j] = x[j + 1] - 1          # back to 0-based labels
                have = True
            step = 5
        elif step == 5:                           # STEP 5: action after fathoming
            # Branching right is pointless if M is the last cluster, or if P
            # is alone in M (moving it would only relabel a new empty cluster).
            step = 7 if (M == G or n[M] == 1) else 6
        elif step == 6:                           # STEP 6: branch right
            for j in range(1, P):
                if x[j] == M:
                    for q in range(Qd):
                        sums[M, q] -= PW[P - 1, j - 1, q]
            n[M] -= 1
            M += 1
            n[M] += 1
            if n[M] == 1:
                Q -= 1
            x[P] = M
            for j in range(1, P):
                if x[j] == M:
                    for q in range(Qd):
                        sums[M, q] += PW[P - 1, j - 1, q]
            step = 2
        else:                                     # STEP 7: retract depth
            x[P] = 0
            for j in range(1, P):
                if x[j] == M:
                    for q in range(Qd):
                        sums[M, q] -= PW[P - 1, j - 1, q]
            n[M] -= 1
            P -= 1
            if n[M] == 0:
                Q += 1
            if P > 0:
                M = x[P]
                step = 5
            else:
                break
    return Z, have, XB, nodes


# =============================================================================
# Driver
# =============================================================================

def _pairwise_matrices(panel, S):
    '''Pairwise matrices for the reordered units (COMPUTE_MATRIX), stacked in
    one array PW[i, j, q], q = 0..Qd-1:
        q = 0                  : A_ij
        q = 1 + k              : B_ij[k]       (k = 0..K-1)
        q = 1 + K + l*K + k    : C_ij[k, l]
    '''
    N, T, K = panel.N, panel.T, panel.K
    Ym = panel.y.reshape(N, T)[S]                       # N x T (reordered)
    Xa = panel.X.reshape(N, T, K)[S] if K > 0 else np.zeros((N, T, 0))

    def pair_cross(U, V):
        # sum_t (u_it - u_jt)(v_it - v_jt) for all pairs (i, j)
        su = np.sum(U * V, axis=1)
        return su[:, None] + su[None, :] - U @ V.T - V @ U.T

    Qd = 1 + K + K * K
    PW = np.zeros((N, N, Qd))
    PW[:, :, 0] = pair_cross(Ym, Ym)
    for k in range(K):
        PW[:, :, 1 + k] = pair_cross(Ym, Xa[:, :, k])
        for l in range(K):
            PW[:, :, 1 + K + l * K + k] = pair_cross(Xa[:, :, k], Xa[:, :, l])
    return PW


def _subset_panel(panel, units):
    '''Panel restricted to a subset of units (balanced panels).'''
    rows = (np.asarray(units)[:, None] * panel.T + np.arange(panel.T)[None, :]).ravel()
    return gfe_panel(panel.y[rows], panel.X[rows], N=len(units), T=panel.T,
                     unit_labels=[panel.unit_labels[u] for u in units],
                     time_labels=panel.time_labels, x_names=panel.x_names)


def gfe_exact_bb(panel, G, heuristic=None, stage_starts=50, seed=None, verbose=False):
    '''Exact GFE solution by repetitive branch and bound (balanced panels).

    Parameters
    ----------
    G : number of groups (the authors used G = 2, and G = 3 for N = 30).
    heuristic : optional GFEFit used for the object ordering and the upper
        bounds. By default a VNS fit is computed.
    stage_starts : Algorithm 1 starts used to get an upper bound at each
        stage (Brusco_cov.f90 uses heuristics_simple with 1,000 starts).
    verbose : print the optimum of each stage, as the Fortran code does.

    Returns a GFEFit with extra attributes stage_optima, nodes, elapsed, and
    heuristic_objective.
    '''
    if not panel.balanced:
        raise ValueError('gfe_exact_bb() requires a balanced panel.')
    rng = np.random.default_rng(seed)
    N, K = panel.N, panel.K
    t_start = _time.time()

    # STEP A. A good heuristic solution (Brusco_cov.f90: CALL heuristics, i.e.
    # VNS with (Nsim; neighmax; itermax) = (10; 10; 2)).
    if heuristic is None:
        heuristic = gfe_vns(panel, G, n_starts=10, neighmax=10, itermax=2, seed=rng)
    hg = np.asarray(heuristic.groups)

    # STEP B. Object ordering (Brusco_cov.f90, "observation sequencing").
    # Put clusters in increasing order of size, take their members in turns,
    # then reverse the sequence.
    members = [np.flatnonzero(hg == g) for g in range(G)]
    members.sort(key=len)
    S = [m[k] for k in range(max(len(m) for m in members)) for m in members if k < len(m)]
    S = np.array(S[::-1])

    # STEP C. Pairwise matrices for the reordered units.
    PW = _pairwise_matrices(panel, S)

    # opt[r] = optimal objective for the last r objects of the sequence (BUP2
    # in Brusco_cov.f90); with r <= G objects each can sit alone, so the
    # optimum is zero.
    opt = np.zeros(N + 1)
    nodes = 0
    XB_final = None

    # STEP D. Repetitive branch and bound: E = G+1, ..., N objects.
    for E in range(G + 1, N + 1):
        sub = np.arange(N - E, N)                  # positions of the last E objects

        # Upper bound for this stage (ZXB): the best of the heuristic grouping
        # restricted to these units and Algorithm 1 on the subset. A small
        # slack lets the search find a solution equal to the bound (Fortran:
        # ZXB + 0.00001).
        sub_units = S[sub]
        sub_panel = _subset_panel(panel, sub_units)
        restricted = hg[sub_units]
        _, restricted = np.unique(restricted, return_inverse=True)  # relabel 0..G'-1
        ub = gfe_objective(sub_panel, restricted, G) if restricted.max() == G - 1 else np.inf
        if stage_starts > 0:
            ub = min(ub, gfe_alg1(sub_panel, G, n_starts=stage_starts, seed=rng).objective)
        Z = ub + 1e-6 * max(1.0, abs(ub))

        Zs, found, XB, nd = _bb_stage(np.ascontiguousarray(PW[np.ix_(sub, sub)]),
                                      E, G, K, Z, opt)
        nodes += nd
        if not found:
            raise RuntimeError(f'Branch and bound found no solution below the upper '
                               f'bound at stage E = {E}.')
        opt[E] = Zs
        if verbose:
            print(f'NUMBER OF OBJECTS {E:3d}  Z = {Zs:.6f}  (nodes so far: {nodes})')
        if E == N:
            XB_final = XB

    # Map back to the original unit ordering (XREOR in Brusco_cov.f90).
    groups = np.empty(N, dtype=int)
    groups[S] = XB_final
    engine = 'numba' if HAVE_NUMBA else 'pure Python'
    fit = _as_gfe(panel, G, groups, [opt[N]],
                  f'exact branch and bound (Brusco, 2006; with covariates; {engine})',
                  {'stage_starts': stage_starts, 'engine': engine})
    fit.stage_optima = opt[1:]
    fit.nodes = nodes
    fit.elapsed = _time.time() - t_start
    fit.heuristic_objective = heuristic.objective
    return fit


def gfe_bruteforce(panel, G):
    '''Brute-force GFE solution by enumerating all groupings (tiny N only).
    Used in tests/ to check gfe_exact_bb() and the heuristics.'''
    N = panel.N
    if G ** (N - 1) > 2e6:
        raise ValueError('too many groupings to enumerate')
    best = {'objective': np.inf, 'groups': None}
    # Unit 0 is fixed in group 0 to remove part of the label symmetry.
    for rest in itertools.product(range(G), repeat=N - 1):
        g = np.array((0,) + rest)
        if len(np.unique(g)) < G:
            continue
        o = gfe_objective(panel, g, G)
        if o < best['objective']:
            best = {'objective': o, 'groups': g}
    return best
