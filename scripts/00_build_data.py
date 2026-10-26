'''scripts/00_build_data.py

Python translation of the replication code for Bonhomme and Manresa (2015),
"Grouped Patterns of Heterogeneity in Panel Data," Econometrica 83(3),
1147-1184.

Builds the estimation sample of the empirical application from the raw
Acemoglu, Johnson, Robinson, and Yared (2008, AER) 5-year panel, following the
"AUTOMATIC CODE" part of GFE_replication_code.do (Replic = 1), and checks that
it reproduces the authors' final_data.mat, which we ship as
data/bm_final_data.csv.

    outcome     y_it : Freedom House democracy index (fhpolrigaug)
    covariates  x_it : lagged democracy, lagged log GDP per capita (lrgdpch),
                       lags of one 5-year period, taken within country
    sample          : balanced panel (samplebalancefe == 1), 90 countries,
                       7 periods (1970, 1975, ..., 2000)

Input: data/ajry_5yearpanel_extract.csv (variables extracted from the authors'
copy of 5yearpanel.dta, restricted to the 90 sample countries and 1965-2000,
the years needed for the lags), or the full .dta via DTA_PATH
(pandas.read_stata reads it directly).

Python translation Copyright (C) 2026 Eliot R. Weinstein. Licensed under the
GNU GPL, version 2 or (at your option) any later version; see LICENSE. The
license covers the translation only, not Bonhomme and Manresa's method or
original code.
'''

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the gfe package importable when this file is run as a script
# (python scripts/00_build_data.py) from any folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gfe import DATA  # noqa: E402

DTA_PATH = None   # e.g. r'C:\...\Bonhomme_Manresa_codes\Application\5yearpanel.dta'


def main():
    if DTA_PATH is not None:
        raw = pd.read_stata(DTA_PATH, convert_categoricals=False)
    else:
        raw = pd.read_csv(DATA / 'ajry_5yearpanel_extract.csv')

    # tsset ID TIME; sort ID TIME
    raw = raw.sort_values(['code_numeric', 'year']).reset_index(drop=True)

    # gen lag_dem = fhpolrigaug[_n-1] if ID[_n] == ID[_n-1]   (and likewise for income)
    # The Stata idiom "previous row, if it belongs to the same unit" is exactly
    # groupby(unit).shift(1) in pandas.
    by_unit = raw.groupby('code_numeric')
    raw['lag_dem'] = by_unit['fhpolrigaug'].shift(1)
    raw['lag_income'] = by_unit['lrgdpch'].shift(1)

    # drop if year < 1960; keep if samplebalancefe == 1
    est = raw[(raw['year'] >= 1960) & (raw['samplebalancefe'] == 1)]
    est = est.sort_values(['code_numeric', 'year']).reset_index(drop=True)
    assert len(est) == 630 and (est['year'].value_counts() == 90).all(), \
        'The balanced sample should have 90 countries x 7 periods.'

    # Compare with the authors' final_data.mat (shipped as a CSV).
    fin = pd.read_csv(DATA / 'bm_final_data.csv')
    assert (est['code'].to_numpy() == fin['code'].to_numpy()).all()
    assert (est['year'].astype(int).to_numpy() == fin['year'].to_numpy()).all()
    diffs = {'dem': np.max(np.abs(est['fhpolrigaug'].to_numpy() - fin['dem'].to_numpy())),
             'lag_dem': np.max(np.abs(est['lag_dem'].to_numpy() - fin['lag_dem'].to_numpy())),
             'lag_income': np.max(np.abs(est['lag_income'].to_numpy() - fin['lag_income'].to_numpy()))}
    print('Max absolute difference with final_data.mat:', diffs)
    print('Differences are at the 1e-7 level (single-precision storage in final_data.mat).\n'
          'All later scripts use data/bm_final_data.csv, i.e. the authors\' exact numbers.')


if __name__ == '__main__':
    main()
