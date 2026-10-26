# How to cite

This repository is a Python translation, by Eliot R. Weinstein, of the replication code for

> Bonhomme, Stéphane and Elena Manresa (2015), "Grouped Patterns of Heterogeneity
> in Panel Data," *Econometrica* 83(3), 1147–1184. doi:10.3982/ECTA11319

The grouped fixed-effects estimator, its theory, the computational algorithms,
the empirical application, and the original MATLAB, Fortran, and Stata code are
the work of Stéphane Bonhomme and Elena Manresa. If you use this code, cite their
paper (and its Supplementary Appendix, which contains the algorithms, the
fixed-T variance, the Monte Carlo design, and the extensions):

```bibtex
@article{BonhommeManresa2015,
  author  = {Bonhomme, St{\'e}phane and Manresa, Elena},
  title   = {Grouped Patterns of Heterogeneity in Panel Data},
  journal = {Econometrica},
  year    = {2015},
  volume  = {83},
  number  = {3},
  pages   = {1147--1184},
  doi     = {10.3982/ECTA11319}
}
```

Please also cite, as relevant:

- the data: Acemoglu, D., S. Johnson, J. A. Robinson, and P. Yared (2008),
  "Income and Democracy," *American Economic Review* 98(3), 808–842,
  https://www.jstor.org/stable/29730096;
- the exact algorithm (`gfe_exact_bb()`): Brusco, M. J. (2006), "A Repetitive
  Branch-and-Bound Procedure for Minimum Within-Cluster Sums of Squares
  Partitioning," *Psychometrika* 71(2), 347–363;
- the fixed-T variance (`gfe_vcov_fixedT()`): Pollard, D. (1982), "A Central
  Limit Theorem for k-Means Clustering," *Annals of Probability* 10(4), 919–926;
- variable neighborhood search (`gfe_vns()`): Hansen, P. and N. Mladenović
  (2001), "Variable Neighborhood Search: Principles and Applications,"
  *European Journal of Operational Research* 130(3), 449–467.

The original replication package (`Bonhomme_Manresa_codes`) is distributed with
the paper's supplemental material by the Econometric Society.

## The Python translation

Python translation by  
Eliot R. Weinstein  
eliotw@uchicago.edu  
[erweinstein.com](https://erweinstein.com)

If you use the Python code itself, please cite it in addition to (never instead of)
Bonhomme and Manresa (2015):

> Weinstein, Eliot R. (2026), "gfe_bm_python: A Python Translation of the
> Replication Code for Bonhomme and Manresa (2015), 'Grouped Patterns of
> Heterogeneity in Panel Data.'" https://github.com/erweinstein/gfe_bm_python

```bibtex
@misc{Weinstein2026gfePython,
  author = {Weinstein, Eliot R.},
  title  = {gfe\_bm\_python: A {P}ython Translation of the Replication Code
            for {B}onhomme and {M}anresa (2015), ``Grouped Patterns of
            Heterogeneity in Panel Data''},
  year   = {2026},
  url    = {https://github.com/erweinstein/gfe_bm_python}
}
```

The Python translation is licensed under the GNU General Public License, version 2
or (at your option) any later version; see `LICENSE`, which also explains what
the license does and does not cover.

The Python translation is not endorsed by, and was not reviewed by, the original
authors. It was translated from the R translation, gfe_bm_R
(https://github.com/erweinstein/gfe_bm_R), which follows the original files line
by line. Differences from the original code and results are documented in README.md;
any errors introduced in translation are the translator's.
