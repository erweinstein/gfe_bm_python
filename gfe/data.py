'''gfe/data.py -- Loaders for the data of the empirical application.

Analysis data used in the paper: the 5-year balanced panel of Acemoglu,
Johnson, Robinson, and Yared (2008), N = 90 countries, T = 7 (1970-2000),
exactly as stored in Bonhomme and Manresa's final_data.mat (shipped here as
data/bm_final_data.csv).

Python translation Copyright (c) 2026 Eliot R. Weinstein. Licensed under the
BSD 3-Clause License; see LICENSE and NOTICE.md (the license covers the
translation only, not Bonhomme and Manresa's method or original code).
'''

from pathlib import Path

import pandas as pd

from .core import gfe_panel

# The project root is the folder that contains gfe/ and data/. Using pathlib
# relative to this file (rather than the current working directory) means the
# loaders work no matter where a script or notebook is run from.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def load_bm_panel(path=None):
    '''The estimation sample as a GFEPanel (x = lagged democracy, lagged log
    income). The DataFrame is attached as panel.data.'''
    dat = pd.read_csv(path or DATA / 'bm_final_data.csv')
    dat = dat.sort_values(['unit', 'year']).reset_index(drop=True)
    N, T = dat['unit'].nunique(), dat['year'].nunique()
    first_year = dat['year'].min()
    p = gfe_panel(dat['dem'].to_numpy(), dat[['lag_dem', 'lag_income']].to_numpy(),
                  N=N, T=T,
                  unit_labels=dat.loc[dat['year'] == first_year, 'code'].tolist(),
                  time_labels=sorted(dat['year'].unique()),
                  x_names=['lag_dem', 'lag_income'])
    p.data = dat
    return p


def load_bm_published():
    '''Published groupings and bootstrap standard errors (transcribed from the
    authors' MATLAB files by tools/make_published_groupings.R in gfe_bm_R).

    The CSV stores the authors' 1-based group labels (1..G), exactly as they
    appear in the MATLAB code. This loader converts every grouping column to
    0-based labels (0..G-1), the convention used throughout this package, so
    that `gfe(panel, G, groups=pub['groupings']['G4'])` works directly.
    (Keep the `- 1` in mind when comparing with the original files.)
    '''
    grp = pd.read_csv(DATA / 'bm_published_groupings.csv')
    label_cols = [c for c in grp.columns if c not in ('unit', 'code')]
    grp[label_cols] = grp[label_cols] - 1
    boot = pd.read_csv(DATA / 'bm_published_bootstrap_se.csv')
    return {'groupings': grp, 'bootstrap_se': boot}
