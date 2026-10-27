'''gfe -- Grouped fixed-effects estimation in Python.

A Python translation of the replication code for

    Bonhomme, Stephane and Elena Manresa (2015), "Grouped Patterns of
    Heterogeneity in Panel Data," Econometrica 83(3), 1147-1184.
    doi:10.3982/ECTA11319

The method, the algorithms, and the original MATLAB, Fortran, and Stata code
are by Stephane Bonhomme and Elena Manresa. Please cite their paper when you
use this code (see CITATION.md).

Usage (from the project folder):

    from gfe import load_bm_panel, gfe
    p = load_bm_panel()
    fit = gfe(p, G=4, algorithm='vns', n_starts=10, neighmax=10, itermax=10, seed=1)
    print(fit)

Python translation Copyright (c) 2026 Eliot R. Weinstein. Licensed under the
BSD 3-Clause License; see LICENSE and NOTICE.md (the license covers the
translation only, not Bonhomme and Manresa's method or original code).
'''

from .core import (GFEPanel, GFEFit, gfe_panel, gfe_panel_from_df, gfe_update,
                   gfe_distances, gfe_assign, gfe_stats, gfe_objective, gfe_alg1,
                   gfe_local_search, gfe_vns, gfe, gfe_relabel)
from .exact import gfe_exact_bb, gfe_bruteforce, HAVE_NUMBA
from .inference import (gfe_vcov_largeT, gfe_se_theta, gfe_delta, gfe_vcov_fixedT,
                        gfe_bandwidth, gfe_bootstrap)
from .select import gfe_bic
from .extensions import gfe_within, gfe_fd_iv, gfe_hetero, gfe_twolayer
from .utils import match_groups, group_means, plot_profiles, plot_by_G, pinv, ols_iv_cluster
from .data import load_bm_panel, load_bm_published, ROOT, DATA

__version__ = '0.1.0'
