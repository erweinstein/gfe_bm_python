'''gfe/select.py -- Choosing the number of groups.

Python translation of the information criteria in Bonhomme and Manresa (2015),
Supplementary Appendix Section S.3.2, as computed in GFE_estimates.m,
GFE_deviations_estimates.m, and MonteCarlo/BIC_G3_MC.m (original code by
Stephane Bonhomme and Elena Manresa).

    BIC(G) = Q(G)/(NT) + sigma2_hat * (G T + N + K)/(NT) * ln(NT)          (S.20)

where Q(G) is the minimized GFE objective with G groups and sigma2_hat is a
consistent estimate of Var(v_it). Following footnote 12 of the Supplementary
Appendix (and the code), sigma2_hat is computed once, from the fit with the
largest number of groups considered, Gmax:

    sigma2_hat = Q(Gmax) / (NT - Gmax T - N - K).

BIC_G3_MC.m also computes an alternative with a steeper penalty,

    BIC2(G) = Q(G)/(NT) + sigma2_hat * G (N + T + K - G)/(NT) * ln(NT).

Both are returned. Consistency: G_hat is consistent for G0 if N and T grow at
the same rate; if T/N -> 0 the BIC need not be consistent (Section S.3.2).

R translation Copyright (C) 2026 Eliot R. Weinstein; Python translation
Copyright (C) 2026 Eliot R. Weinstein. Licensed under the GNU GPL, version 2 or
(at your option) any later version; see LICENSE. The license covers the
translation only, not Bonhomme and Manresa's method or original code.
'''

import numpy as np
import pandas as pd


def gfe_bic(fits):
    '''Information criteria for a list of GFEFit objects with different G.

    The G = 1 fit is the model with common time effects only (two-way FE
    without unit effects), as the OLS benchmark in GFE_estimates.m.

    Returns a DataFrame with columns G, objective, BIC, BIC2. The selected G's
    and sigma2_hat are stored in DataFrame.attrs (the pandas counterpart of R
    attributes): attrs['G_BIC'], attrs['G_BIC2'], attrs['sigma2'].
    '''
    fits = sorted(fits, key=lambda f: f.G)
    Gs = np.array([f.G for f in fits])
    first = fits[0]
    N, T, K = first.N, first.T, first.K
    NT = first.panel.d.sum()
    Gmax = Gs.max()
    sigma2 = fits[-1].objective / (NT - Gmax * T - N - K)
    Q = np.array([f.objective for f in fits])
    out = pd.DataFrame({
        'G': Gs,
        'objective': Q,
        'BIC': Q / NT + sigma2 * np.log(NT) * (Gs * T + N + K) / NT,
        'BIC2': Q / NT + sigma2 * np.log(NT) * Gs * (N + T + K - Gs) / NT,
    })
    out.attrs['sigma2'] = sigma2
    out.attrs['G_BIC'] = int(out.G[out.BIC.idxmin()])
    out.attrs['G_BIC2'] = int(out.G[out.BIC2.idxmin()])
    return out
