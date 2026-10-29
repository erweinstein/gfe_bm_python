'''04b_bic_tableS8.py -- Table S.VIII (BIC choice of the number of groups), and why
our first attempt did not reproduce it.

The appendix defines (S20)

    BIC(G) = Q(G)/NT + sigma2_hat (GT + N + K)/NT ln(NT),
    sigma2_hat = Q(Gmax)/(NT - Gmax T - N - K),  Gmax = 15  (footnote 12),

and reports, for G0 = 3, G = 3 in 98% of 100 samples and G = 4 in 2%; for G0 = 10,
G = 8, 9, 10, 11 in 10%, 42%, 42%, 6%.

The original code (MonteCarlo/BIC_G3_MC.m, BIC_G10_MC.m) shows one difference from
the text. BIC_G3_MC.m computes sigma2_hat on the G0 = 3 samples with
`BigG = load('assignmentG010G15.txt')`: the G = 15 groupings estimated on the
G0 = 10 samples. Those groupings do not fit the G0 = 3 data, so Q(15) is large,
sigma2_hat is about twice what each sample's own G = 15 fit gives, and the penalty
doubles. BIC_G10_MC.m uses the same file for the G0 = 10 samples, where it is the
samples' own fit. Both scripts average sigma2_hat over the 100 samples and use the
average in every sample; the G0 = 3 script compares G = 1..7, the G0 = 10 script
G = 3..15.

This script runs both readings:
  own   : sigma2_hat from each sample's own G = 15 fit (the text, and BIC_G10_MC.m)
  g0_10 : for G0 = 3, sigma2_hat from a G = 15 grouping estimated on an independent
          G0 = 10 sample and applied to the G0 = 3 sample (BIC_G3_MC.m)
and, for reference, the true error variance.

Groupings are estimated with Algorithm 1 (N_STARTS random starts) for speed; the
authors' assignment files came from their Fortran program (MC1.exe).

Usage: python scripts/04b_bic_tableS8.py [G0] [R] [N_STARTS]
Output: output/tableS8_bic_G{G0}.csv

Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
'''
import collections
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util  # noqa: E402
from gfe import ROOT, load_bm_panel, load_bm_published, gfe  # noqa: E402

_spec = importlib.util.spec_from_file_location('mc', Path(__file__).resolve().parent / '04_montecarlo.py')
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

G0 = int(sys.argv[1]) if len(sys.argv) > 1 else 3
R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_STARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
GMAX = 15
SEED = {3: 20261029, 10: 20261030}.get(G0, 20261029)
PUBLISHED = {3: {3: 98, 4: 2}, 10: {8: 10, 9: 42, 10: 42, 11: 6}}

p = load_bm_panel()
pub = load_bm_published()['groupings']
N, T, K = p.N, p.T, p.K
NT = N * T
truth = gfe(p, G0, groups=pub[f'G{G0}'])
s2_true = truth.objective / (NT - G0 * T - N - K)
t10 = gfe(p, 10, groups=pub['G10'])
s10 = t10.objective / (NT - 10 * T - N - K)
GS = np.arange(1, 8) if G0 == 3 else np.arange(3, GMAX + 1)
tun = dict(algorithm='alg1', n_starts=N_STARTS)

rng = np.random.default_rng(SEED)
Q, S_own, S_g010 = [], [], []
t0 = time.time()
for r in range(R):
    ps = mc.simulate_panel(truth, 'iid', s2_true, rng)
    fits = {G: gfe(ps, G, seed=rng, **tun).objective for G in sorted(set(GS) | {GMAX})}
    Q.append([fits[G] for G in GS])
    S_own.append(fits[GMAX] / (NT - GMAX * T - N - K))
    if G0 != 10:
        p10 = mc.simulate_panel(t10, 'iid', s10, rng)
        g15 = gfe(p10, GMAX, seed=rng, **tun).groups
        S_g010.append(gfe(ps, GMAX, groups=g15).objective / (NT - GMAX * T - N - K))
    if (r + 1) % 20 == 0:
        print(f'{r + 1}/{R}  {time.time() - t0:.0f}s', flush=True)
Q = np.array(Q)

readings = [('own', np.mean(S_own))]
if G0 != 10:
    readings.append(('g0_10', np.mean(S_g010)))
readings.append(('true', s2_true))
rows = []
for lab, s2 in readings:
    bic = Q / NT + s2 * np.log(NT) * (GS * T + N + K) / NT
    cnt = collections.Counter(GS[bic.argmin(axis=1)].tolist())
    rows.append({'sigma2_source': lab, 'sigma2_hat': round(s2, 4), **{f'G{G}': cnt.get(G, 0) for G in GS}})
rows.append({'sigma2_source': 'published', 'sigma2_hat': np.nan,
             **{f'G{G}': round(PUBLISHED[G0].get(G, 0) * R / 100) for G in GS}} if G0 in PUBLISHED else {})
tab = pd.DataFrame([r for r in rows if r])
print(f'\nG0 = {G0}, {R} samples, Algorithm 1 with {N_STARTS} starts:')
print(tab.to_string(index=False))
tab.to_csv(ROOT / 'output' / f'tableS8_bic_G{G0}.csv', index=False)
