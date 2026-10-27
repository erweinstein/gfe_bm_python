# Notice: scope of the license and attribution

gfe_bm_python: A Python translation of the replication code for Bonhomme and
Manresa (2015), "Grouped Patterns of Heterogeneity in Panel Data"

Copyright (c) 2026 Eliot R. Weinstein  
eliotw@uchicago.edu  
https://erweinstein.com

The Python translation is licensed under the BSD 3-Clause License (see
`LICENSE`), the license used by NumPy, SciPy, pandas, scikit-learn, and
statsmodels.

That license covers only the material written by Eliot R. Weinstein in this
repository: the Python code, the scripts, the tests, and the documentation.
It does not cover, and grants no rights in, the following, which remain the
work and property of their authors and are subject to their own terms:

- The grouped fixed-effects estimator, its theory, the computational
  algorithms, and the original MATLAB, Fortran, and Stata replication code, by
  Stéphane Bonhomme and Elena Manresa, on which this translation is based:
  Bonhomme, S. and E. Manresa (2015), "Grouped Patterns of Heterogeneity in
  Panel Data," *Econometrica* 83(3), 1147–1184. doi:10.3982/ECTA11319

- The data in `data/`, which come from Acemoglu, D., S. Johnson, J. A.
  Robinson, and P. Yared (2008), "Income and Democracy," *American Economic
  Review* 98(3), 808–842, https://www.jstor.org/stable/29730096, as
  distributed in Bonhomme and Manresa's replication files, and the groupings,
  standard errors, and related results transcribed from those files.

The companion R translation, gfe_bm_R (https://github.com/erweinstein/gfe_bm_R),
is licensed separately, under the GNU GPL (version 2 or later), following the
convention of the R community.

If you use this code, please cite Bonhomme and Manresa (2015); see
`CITATION.md`.
