---
primary_layer: layer2
related_layers: []
page_type: concept
---

# CORS-RBF (Constrained Optimization using Response Surfaces)

CORS-RBF is a sequential optimization method for expensive-to-evaluate objective functions that selects the next evaluation point one at a time using a surrogate model built from RBF (Radial Basis Function) interpolation.
CORS-RBF is the implementation of CORS (Constrained Optimization using Response Surfaces), the framework proposed by Regis & Shoemaker (2005), realized with an RBF surrogate model.

## Overview

RBF interpolation only reconstructs a smooth surface passing through the training points, and unlike GP regression, it has no predictive variance.
Because of this, the naive approach of simply minimizing the surrogate model's prediction to choose the next evaluation point ends up repeatedly searching only around points where good values have already been observed, and can converge to a point that isn't even a local minimum of the true function.

CORS avoids this problem by building a **distance constraint** directly into the candidate-point selection itself.
The auxiliary problem solved at each iteration is a constrained optimization that, in addition to minimizing the surrogate model $\hat f_i(x)$, requires the next candidate point to be at least $\beta_i \Delta_i$ away from every existing evaluated point ($\Delta_i$ being the maximum of the minimum distances from the existing point set).
$\beta_i$ is given, iteration by iteration, as a sequence (the **search pattern**) that cycles from values near 1 (favoring global search) down to 0 (favoring local search, i.e. simply minimizing the surrogate model); this distance constraint takes over the role of exploration that a GP's predictive variance would otherwise play.

This distance constraint is not a side effect — it is the core of CORS.
It has been proven that, as long as the search pattern contains at least one nonzero value, convergence to the global minimum of any continuous function is guaranteed, regardless of the type of surrogate model or how the initial evaluation points are chosen.

The source is {cite}`regis2005cors`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} CORS-RBF
:label: alg-rbf-cors

**Inputs** objective function $f$, search domain $\mathcal{D}$, initial evaluated point set $S_1 = \{x_1, \ldots, x_k\}$, periodic distance-parameter sequence (search pattern) $\langle \beta_1, \ldots, \beta_{N+1}=0 \rangle$
**Output** best solution $x^*$

1. Evaluate $S_1$ with the true function $f$ and set $i := 1$
2. Fit an RBF surrogate model $\hat f_i$ to the evaluated data so far, $D_i = \{(x, f(x)) \mid x \in S_i\}$
3. Solve the constrained minimization problem $\min_{x \in \mathcal{D}} \hat f_i(x) \ \mathrm{s.t.} \ \|x - x_j\| \geqslant \beta_i \Delta_i \ (j=1,\ldots,|S_i|)$ to find the candidate point $x_{k+i}$ ($\Delta_i$ being the maximum of the minimum distances from the existing evaluated point set)
4. Evaluate $x_{k+i}$ with the true function and add it to $S_{i+1} := S_i \cup \{x_{k+i}\}$
5. Update $\beta_i$ according to the periodic sequence, set $i := i+1$, and return to step 2 until the termination condition is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>→ true evaluation<br/>(L1)"] --> ASK
    subgraph GEN["One generation (PreSelectionStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidate points"] --> SCORE["SurrogateManager<br/>Fit RBF (L2)<br/>→ Score with CORSDistance<br/>Apply βᵢΔᵢ distance constraint<br/>(L3)"]
        SCORE --> SORT["Select top-1 candidate<br/>by CORSDistance score"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget N<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Best solution x*"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (the specific combination of crossover, mutation, and selection is not part of CORS's definition) | L3 |
| Surrogate model | `RBFSurrogate` (RBF interpolation; this example uses `GaussianKernel()`, whose default `polynomial_degree="auto"` adds a constant term, but `kernel` is a required argument and any `RBFKernel`/`polynomial_degree` can be injected — see Differences from the source) | L2 |
| Acquisition function | `CORSDistance` (applies a $\beta_i\Delta_i$ distance constraint to the predictive mean) | L3 |
| Surrogate management | `GlobalSurrogateManager` (fits the RBF over the entire archive) | L2-3 |
| Evaluation strategy | `PreSelectionStrategy(n_select=1)` (source-faithful: one true evaluation per CORS decision) | L3-4 |

```python
import numpy as np
from saealib import (
    GA,
    GaussianKernel,
    Optimizer,
    Problem,
    PreSelectionStrategy,
    RBFSurrogate,
)
from saealib.acquisition import CORSDistance
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.termination import Termination, max_fe


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


lb = np.asarray([-5.0] * 5)
ub = np.asarray([5.0] * 5)
problem = Problem(sphere, dim=5, lb=lb, ub=ub, n_obj=1, direction=[-1])

algorithm = GA(
    CrossoverBLXAlpha(prob=0.7, alpha=0.4),
    MutationUniform(prob_var=0.3),
    SequentialSelection(),
    TruncationSelection(),
)
surrogate_manager = GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
# delta=None uses the candidate-pool maximin approximation
acquisition = CORSDistance(delta=None, direction=problem.direction)
# n_select=1: source-faithful one-candidate-per-decision cadence
strategy = PreSelectionStrategy(n_candidates=10, n_select=1)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(acquisition)
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

## Differences from the source

The original CORS procedure selects candidate points sequentially by minimizing a constrained prediction, whereas saealib approximates this minimization by selecting the highest-`CORSDistance` candidates from the pool generated by `GA`.
The example above uses `PreSelectionStrategy(n_select=1)` to maintain the source-faithful one-candidate-per-decision cadence.
The crossover, mutation, parent-selection, and environmental-selection combination in the example is an saealib configuration choice, not part of the CORS definition.
The paper's numerical experiments use a thin-plate-spline kernel ($\phi(r) = r^2 \log r$) with a first-order polynomial term $p(x)$.
With the `kernel=GaussianKernel()` used above, `RBFSurrogate`'s default `polynomial_degree="auto"` resolves to a constant term (`GaussianKernel`'s `auto_polynomial_degree` is `0`), so swapping only the kernel does not reproduce the paper's first-order term.
Passing `kernel=ThinPlateSplineKernel()` together with `polynomial_degree=1` reproduces it, since a linear polynomial term is what makes the interpolation system well-posed for a conditionally positive definite kernel such as thin-plate-spline.

## Parameters and variants

**kernel (choice of RBF kernel)**: `RBFSurrogate(kernel=...)` requires an `RBFKernel` instance; there is no default.
The example above passes `GaussianKernel()`.

**delta**: With the default `delta=None`, `CORSDistance` approximates the
paper's iteration-specific $\Delta_i$ with the candidate-pool maximin distance
`max_candidate min_evaluated distance`. A finite positive numeric `delta`
selects a fixed distance scale instead.

## Beta cadence

`CORSDistance.prepare(archive, *, ctx)` selects
`search_pattern[ctx.decision_count % len(search_pattern)]`. `decision_count`
counts the evaluation plans the runtime has confirmed, so the first decision
uses `search_pattern[0]` and the pattern repeats cyclically, one entry per
decision. Preparing the same decision a second time — after an archive change
invalidates a cached reference, for instance — reuses that decision's entry
instead of skipping ahead. Repeated `score()` calls use the prepared beta and
do not advance the pattern.

## Related

- [References](../references.md): Full bibliographic details for the source
- [Surrogate](../concepts/surrogate_modeling/surrogate.md): List of surrogate models including `RBFSurrogate`/`GaussianKernel`
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): List of acquisition functions including distance-constrained `CORSDistance` for CORS
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md): Detailed usage of `GlobalSurrogateManager`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `PreSelectionStrategy`
