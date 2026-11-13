---
primary_layer: layer2
related_layers: []
page_type: concept
---

# NSGA-III

NSGA-III primarily targets many-objective optimization, but its reference-direction behavior can also be observed with three objectives.
It replaces crowding-distance diversity maintenance with a niche-preservation operation against pre-placed reference points, maintaining the distribution of the solution set even as the number of objectives grows.

## Overview

Once the number of objectives grows to four or more, NSGA-II's crowding distance can no longer adequately maintain diversity.
Because the fraction of non-dominated individuals in a randomly generated population grows exponentially with the number of objectives, narrowing down by dominance relations alone is no longer enough to fill the next-generation population, and the relative weight of the role crowding distance plays in diversity maintenance grows accordingly.

NSGA-III replaces this crowding distance with a niche-preservation operation against **reference points** pre-placed in objective space.
In each generation, the hyperplane intercepts are computed from the population's **ideal point** and **extreme points** to normalize the objectives, and each individual is associated with the reference point to which it has the smallest perpendicular distance.
For the last front that cannot be fully accepted, reference points with fewer already-assigned individuals (niche count) are prioritized, and individuals associated with each are chosen in order of smallest perpendicular distance, leaving a roughly equal number of solutions per reference point.

Because NSGA-III already secures diversity through this niche-preservation operation, it does not use dominance-based parent selection like NSGA-II.
Parents are chosen randomly from the current population, and crossover and mutation are applied to generate the offspring population.

Uniform placement of reference points uses the simplex-lattice design proposed by Das and Dennis (Das & Dennis, 1998).

The source is {cite}`deb2014nsga3`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} NSGA-III
:label: alg-nsga3

**Inputs** objective functions, reference point set $Z^r$ (structured points $Z^s$ or user-specified points $Z^a$), population size $N$, initial population $P_0$
**Output** the population $P_{t+1}$ of the final generation

1. Set $t=0$; generate an offspring population $Q_0$ via crossover and mutation from a randomly generated $P_0$
2. Form the combined population $R_t = P_t \cup Q_t$ (size $2N$)
3. Non-dominated-sort $R_t$ to obtain the front sequence $\mathcal{F} = (\mathcal{F}_1, \mathcal{F}_2, \ldots)$
4. Set $S_t = \emptyset$, adding fronts to $S_t$ in order starting from $\mathcal{F}_1$ until $|S_t| \geq N$. Let $\mathcal{F}_l$ be the last front added, $P_{t+1} = \bigcup_{i=1}^{l-1}\mathcal{F}_i$, and $K = N - |P_{t+1}|$ (if $|S_t|=N$, set $P_{t+1}=S_t$ directly and proceed to step 8)
5. Compute the hyperplane intercepts from $S_t$'s ideal point and extreme points to normalize the objectives, and place $Z^r$ in the normalized objective space
6. Associate each individual in $S_t$ with the point having the smallest perpendicular distance to a reference line through the origin and each point of $Z^r$
7. Compute each reference point's niche count $\rho_j$ from the associations on $\mathcal{F}_1, \ldots, \mathcal{F}_{l-1}$, then select $K$ individuals from $\mathcal{F}_l$, prioritizing reference points with the smallest niche count, and add them to $P_{t+1}$
8. Choose parents randomly from $P_{t+1}$, apply crossover and mutation to generate $Q_{t+1}$. Set $t=t+1$ and return to step 2, repeating until the termination condition is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Generate initial population P0<br/>(L1)"] --> GEN
    subgraph GEN["One generation (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Randomly select parents →<br/>SBX crossover →<br/>Polynomial mutation to generate Qt<br/>(L1, L8)"] --> EVAL["True evaluation<br/>(no surrogate involved)"]
        EVAL --> COMB["GA.tell()<br/>Combine Rt = Pt ∪ Qt<br/>(L2)"]
        COMB --> SORT["NSGA3Comparator.sort_population()<br/>Non-dominated sorting → adaptive normalization →<br/>association to reference points →<br/>niche-preservation selection<br/>(L3-7)"]
        SORT --> TRUNC["TruncationSelection<br/>Take top N individuals as Pt+1<br/>(L4-7)"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L8)" --> GEN
    TERM -- "Reached" --> RESULT(["Population of the final generation"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (`ask()` performs crossover and mutation; `tell()` performs combining $R_t=P_t\cup Q_t$ and survivor selection) | L1-2, L8 |
| Parent selection | `TournamentSelection(tournament_size=1)` (with tournament size 1, no comparison is actually performed, corresponding to the paper's description of "choose parents randomly from $P_{t+1}$") | L1, L8 |
| Crossover | `CrossoverSBX(prob=1.0, eta=30.0)` | L1, L8 |
| Mutation | `MutationPolynomial(eta=20.0)` | L1, L8 |
| Reference point generation | `uniform_weight_vectors(n_obj, n_divisions)` (generates the initial value $Z^s$ of $Z^r$ via the Das-Dennis simplex lattice) | L5 |
| Non-dominated sorting + normalization + association + niche preservation | `NSGA3Comparator` (`sort_population` internally calls `_normalize_objectives`/`_associate_to_reference_points`/`_niche_count_select` in order) | L3-7 |
| Survivor selection | `TruncationSelection()` (keeps the top $N$ individuals in the order given by `comparator.sort_population`) | L4-7 |
| Evaluation strategy | `DirectStrategy` (no surrogate involved; every candidate generated by `GA.ask()` is evaluated with the true objective function) | L2 |

```python
from saealib import GA, NSGA3Comparator, Optimizer, uniform_weight_vectors
from saealib.benchmarks import dtlz2
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


problem = dtlz2(n_obj=3)
reference_points = uniform_weight_vectors(n_obj=3, n_divisions=8)

algorithm = GA(
    CrossoverSBX(prob=1.0, eta=30.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=1),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_comparator(NSGA3Comparator(reference_points))
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy())
    .set_termination(Termination(max_fe(3000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

The `set_comparator(NSGA3Comparator(reference_points))` call cannot be omitted.
In NSGA-II, `NSGA2Comparator` is the default for `n_obj > 1`, so the same call could be omitted, but in NSGA-III, as with SPEA2, an explicit comparator is required.

The example uses the three-objective DTLZ2 rather than a two-objective ZDT benchmark because NSGA-III primarily targets many-objective optimization while still exposing reference-direction niche preservation with three objectives.

## Differences from the source

The original NSGA-III chooses parents randomly, while saealib represents this with `TournamentSelection(tournament_size=1)`, which performs no comparison.
Setting `tournament_size` to 2 or more adds selection pressure from a dominance relation that the original method does not specify.
The combination of SBX, Polynomial mutation, and `TruncationSelection` in the example is an saealib configuration choice.
The original method recommends a relationship between population size and the number of reference points, but saealib's default initialization makes population size depend on `dim` and does not tie it to the number of reference points.

## Parameters and variants

### Complexity

The implemented non-dominated sorter has complexity $O(N\log^{M-2}N)$ ($M$ is the number of objectives and $N$ is the population size).
This is asymptotically different from NSGA-II's $O(MN^2)$, but the bound applies only to this sorter, not to NSGA-III as a whole.
The worst-case per-generation complexity combining normalization, association, and niche preservation is the larger of $O(N^2\log^{M-2}N)$ and $O(N^2 M)$ {cite}`deb2014nsga3`.

**$\eta_c$ (SBX distribution index) and crossover probability $p_c$**: The paper reports using $p_c=1$ (`CrossoverSBX(prob=1.0)`) and $\eta_c=30$ (`CrossoverSBX(eta=30.0)`) for NSGA-III.
Both the crossover probability and distribution index are larger than NSGA-II's defaults ($p_c=0.9$, $\eta_c=20$), a setting that generates offspring closer to the parents with higher probability {cite}`deb2014nsga3`.


**Swapping the dominator (dominance predicate)**: `NSGA3Comparator(reference_points, dominator=...)` lets you inject a [Dominator](../concepts/problem_and_ranking/dominance.md) other than the default `ParetoDominator`.
Since this changes the result of non-dominated sorting itself, the population subjected to front splitting and niche preservation also depends on this dominance predicate.

## Related

- [References](../references.md): Full bibliographic details for the source
- [Comparator](../concepts/problem_and_ranking/comparators.md): Detailed specification of `NSGA3Comparator`, the `reference_points` argument, and lazy `rng` generation
- [Crossover](../concepts/search_algorithms/crossover.md): List of crossover operators including `CrossoverSBX`
- [Mutation](../concepts/search_algorithms/mutation.md): List of mutation operators including `MutationPolynomial`
- [ParentSelection](../concepts/search_algorithms/parent_selection.md): Detailed usage of `TournamentSelection`
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md): Detailed usage of `TruncationSelection`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `DirectStrategy`
- [NonDominatedSorting](../concepts/problem_and_ranking/nondominated_sorting.md): Implementation details of non-dominated sorting
- [Dominator](../concepts/problem_and_ranking/dominance.md): List of dominance predicates that can be swapped in via the `dominator` argument
