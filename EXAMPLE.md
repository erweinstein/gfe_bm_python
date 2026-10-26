# Example: watching the estimator work

Python translation by  
Eliot R. Weinstein  
eliotw@uchicago.edu  
[erweinstein.com](https://erweinstein.com)

A good first exercise with this code is to change two things, the number of
groups G and the starting values of the search, and watch how the estimated
grouping and the estimates of theta move. It takes a few minutes and shows
quickly why Bonhomme and Manresa (2015) put so much care into computation.

## Why this is worth doing

The GFE objective is minimized over all G^N ways of assigning N units to G
groups, so no search is guaranteed to find the global minimum. Bonhomme and
Manresa's answer is a model of careful applied econometrics. They define the
estimator as a single optimization problem, then check the computation instead
of assuming it works:

- they run the heuristics from many starting values, with an explicit rule that
  different starts should reach the exact same solution;
- they certify the heuristics with an exact branch-and-bound algorithm on
  subsamples;
- they pair the asymptotic theory with Monte Carlo simulations calibrated to
  their own data;
- they organize the robustness checks around the main finding, asking in each
  one whether the grouped patterns survive a specific objection;
- and they release replication files that let a reader rerun every step.

The exercises below let you see, on their data, why each of those steps
matters.

## Setup

From the project folder (in Jupyter, VS Code, Spyder, or a Python console):

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gfe import load_bm_panel, load_bm_published, gfe, match_groups

p = load_bm_panel()                        # 90 countries, 1970-2000
pub = load_bm_published()['groupings']     # the authors' optimal groupings (0-based labels)

def cum(th):                               # cumulative income effect
    return th[1] / (1 - th[0])
```

## 1. Same G, different starting values

Run Algorithm 1 from 200 random starting values with G = 3, one start at a
time, and keep the objective and theta from each run.

```python
rng = np.random.default_rng(2015)
runs = []
for _ in range(200):
    f = gfe(p, G=3, algorithm='alg1', n_starts=1, seed=rng)
    runs.append({'objective': f.objective, 'theta1': f.theta[0],
                 'theta2': f.theta[1], 'cumulative': cum(f.theta)})
runs = pd.DataFrame(runs)

print(runs['objective'].describe())
print((runs['objective'] < 16.5988).sum())   # starts that reach the global minimum, 16.59873

fig, ax = plt.subplots()
ax.scatter(runs['objective'], runs['theta1'], alpha=0.4, color='black')
ax.axvline(16.59873, linestyle='--', color='gray')   # global minimum (published solution)
ax.set_xlabel('objective at convergence (G = 3)')
ax.set_ylabel('estimate of theta1 (lagged democracy)')
plt.show()
```

Each point is a local minimum. With this seed, only 3 of the 200 single starts
reach the global minimum (16.5987, the published solution), and the local
minima give quite different estimates of theta. Taking the first answer the
algorithm returns would be a mistake. Taking the best of many starts, and
checking that the best is reached repeatedly, is what the paper recommends.

Variable neighborhood search (Algorithm 2) adds a local search and random
jumps to escape these local minima. Compare a handful of single-start VNS runs:

```python
rng = np.random.default_rng(2015)
vns_runs = [gfe(p, 3, algorithm='vns', n_starts=1, neighmax=10, itermax=10, seed=rng).objective
            for _ in range(5)]
print(np.round(vns_runs, 5))   # 4 of these 5 reach 16.59873
```

## 2. Same data, different G

Now hold the search fixed and change G. With the authors' groupings, this is
instant:

```python
fits = [gfe(p, G, groups=pub[f'G{G}']) for G in range(2, 7)]
print(pd.DataFrame({'G': range(2, 7),
                    'objective': [f.objective for f in fits],
                    'theta1': [f.theta[0] for f in fits],
                    'theta2': [f.theta[1] for f in fits],
                    'cumulative': [cum(f.theta) for f in fits]}))
```

The objective always falls as G grows; that alone says nothing about the right
G (see `gfe_bic()`). The coefficient on lagged democracy moves a lot between
G = 2 and G = 4, because the group-specific time effects absorb persistence
that a model with fewer groups attributes to lagged democracy. The cumulative
income effect is more stable.

To see how the groups themselves change, cross-tabulate the groupings for
neighboring values of G:

```python
print(pd.crosstab(pub['G3'], pub['G4']))
print(pd.crosstab(pub['G4'], pub['G5']))
```

Most countries stay together: each G = 3 group maps mainly onto one or two of
the G = 4 groups, and only a handful of countries move elsewhere. Group labels
are arbitrary, so focus on the large cells, not on the numbers on the margins.

## 3. Put the two together

Finally, search from scratch for several G, and check each search against the
published solution:

```python
rng = np.random.default_rng(1)
for G in (2, 3, 4):
    v = gfe(p, G, algorithm='vns', n_starts=5, neighmax=10, itermax=5, seed=rng)
    m = match_groups(v.groups, pub[f'G{G}'], G)
    n_reach = np.sum(np.abs(v.objectives_by_start - v.objective) < 1e-8)
    print(f'G = {G}: objective {v.objective:.5f}, {n_reach} of 5 starts reach it, '
          f'{round(m["misclassification"] * p.N)} countries classified differently')
```

With this seed, all three searches recover the published groupings exactly,
but the share of starts that reach the best solution falls as G grows (4, 4,
and 1 of 5 for G = 2, 3, and 4), and each search takes longer. This is why the
paper uses (10; 10; 10) for its main estimates and an exact algorithm to check
the heuristics where that is feasible (`scripts/03_exact_algorithm.py`).

## Going further

- `GFE_walkthrough.qmd` goes through the paper section by section.
- `scripts/04_montecarlo.py` repeats the exercise above many times on
  simulated data, where the true grouping is known, and shows what
  misclassification does to inference.
- To try the estimator on your own panel, build it with
  `gfe_panel_from_df(df, id, time, y, x)` and call `gfe()` on the result.

## Reference

Bonhomme, Stéphane and Elena Manresa (2015), "Grouped Patterns of Heterogeneity
in Panel Data," *Econometrica* 83(3), 1147–1184. doi:10.3982/ECTA11319
