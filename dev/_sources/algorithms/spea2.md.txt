---
primary_layer: layer2
related_layers: []
page_type: concept
---

# SPEA2 (Strength Pareto Evolutionary Algorithm 2)

SPEA2 is a multi-objective evolutionary algorithm that improves the fitness assignment and archive management of the original SPEA (Strength Pareto Evolutionary Algorithm).
It ranks individuals by combining a dominance-based fitness with $k$-th-nearest-neighbor density, and maintains a fixed-size external archive using a truncation procedure that avoids losing boundary solutions.

## Overview

SPEA showed strong performance as an early multi-objective evolutionary algorithm, but had two weaknesses.

One was that individuals dominated by the same archive individuals ended up with identical fitness, making it impossible to distinguish between them.
The other was that the clustering technique used when the archive exceeded its capacity could lose boundary solutions on the outer edge of the non-dominated solution set.

SPEA2 resolves each of these two weaknesses with an independent mechanism.
Each individual $i$ is assigned a **strength** $S(i)$, the number of individuals it dominates, and the sum of the strengths of the individuals that dominate $i$ becomes its **raw fitness** $R(i)$.
$R(i)=0$ means $i$ is non-dominated, and a larger value indicates it is dominated by more (and stronger) individuals.
To distinguish between individuals sharing the same $R(i)$, the reciprocal of the distance $\sigma_i^k$ to the $k$-th nearest individual in objective space is added as a **density** term, $D(i)=1/(\sigma_i^k+2)$.
The final **fitness** $F(i)=R(i)+D(i)$ is better when smaller, and non-dominated individuals always have $F(i)<1$.

SPEA2 maintains a fixed-size external archive separate from the population.
In each generation, non-dominated individuals ($F(i)<1$) are copied from the combined set of the population and archive into a new archive.
If the copied archive fits exactly within the specified size, it is used as-is; if it falls short, inferior solutions are added in order of increasing $F(i)$ to fill it.
If it exceeds the specified size, a **truncation operator** is applied, removing individuals one at a time — always the one with the smallest nearest-neighbor distance — recomputing distances as it goes.
Because this procedure breaks ties between individuals with equal distance using the distance to their second- and third-nearest neighbors in turn, boundary solutions are unlikely to be removed by mistake.

The source is {cite}`zitzler2001spea2`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} SPEA2
:label: alg-spea2

**Inputs** population size $N$, archive size $\bar N$, maximum generations $T$
**Output** the non-dominated solution set $A$

1. Initialization: generate an initial population $P_0$, prepare an empty archive $\bar P_0 = \emptyset$, and set $t=0$
2. Fitness assignment: compute the fitness $F(i) = R(i) + D(i)$ of every individual in $P_t$ and $\bar P_t$
3. Environmental selection: copy the non-dominated individuals of $P_t \cup \bar P_t$ into $\bar P_{t+1}$. If $|\bar P_{t+1}| > \bar N$, reduce it with the truncation operator; if $|\bar P_{t+1}| < \bar N$, fill it with inferior solutions in order of increasing $F(i)$
4. Termination check: if $t \geq T$ or another termination condition is met, output the non-dominated individuals in $\bar P_{t+1}$ as $A$ and stop
5. Mating selection: perform binary tournament selection (with replacement) on $\bar P_{t+1}$ to fill the mating pool
6. Variation: apply crossover and mutation to the mating pool to generate $P_{t+1}$, set $t=t+1$, and return to step 2
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["LHSInitializer<br/>Sample n_init_archive individuals (≈ literature P0) →<br/>rank_population() ranks them<br/>(fitness assignment + environmental selection) →<br/>take top n_init_population as<br/>initial ctx.population (≈ literature P̄1)<br/>(L1, L2, L3)"] --> GEN
    subgraph GEN["One generation (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Binary tournament selection →<br/>SBX crossover →<br/>Polynomial mutation to generate<br/>N offspring<br/>(L5, L6)"] --> EVAL["True evaluation<br/>(no surrogate involved;<br/>no independent pseudocode step)"]
        EVAL --> COMB["GA.tell()<br/>Combine population (P̄t) and offspring<br/>into a single pool"]
        COMB --> RANK["SPEA2Comparator.rank_population()<br/>prepare_population(): S(i)→R(i)→D(i)→F(i)<br/>sort_population(): F(i)&lt;1 spea2_truncation_order,<br/>F(i)≥1 ascending F(i),<br/>infeasible: ascending cv<br/>(L2, L3)"]
        RANK --> TRUNC["TruncationSelection<br/>Take top N̄ individuals as P̄t+1"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L4)" --> GEN
    TERM -- "Reached" --> RESULT(["Non-dominated solution set A"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Initialization | `LHSInitializer` (samples `n_init_archive` individuals, true-evaluates them, and calls `rank_population()` — folding the paper's initial fitness assignment and first environmental selection into initialization — then hands the top `n_init_population` to the initial population) | L1, L2, L3 |
| Search algorithm | `GA` (`ask()` performs mating selection and variation; `tell()` combines the population and offspring into a single pool and performs environmental selection) | L3, L5, L6 |
| Parent selection | `TournamentSelection(tournament_size=2)` (binary tournament; the winner is decided via `compare_population`) | L5 |
| Crossover | `CrossoverSBX(prob=0.9, eta=20.0)` | L6 |
| Mutation | `MutationPolynomial(eta=20.0)` | L6 |
| Fitness computation & environmental selection | `SPEA2Comparator.rank_population()` (`prepare_population()` computes $S(i)$/$R(i)$/$D(i)$/$F(i)$ via `spea2_fitness` and persists it onto the population; `sort_population()` orders the non-dominated $F(i)<1$ block via `spea2_truncation_order`, the dominated $F(i)\geq1$ block by ascending $F(i)$, and infeasible solutions by ascending `cv`) + `TruncationSelection()` (takes the top $\bar N$) | L2, L3 |
| Evaluation strategy | `DirectStrategy` (no surrogate involved; every candidate generated by `GA.ask()` is evaluated with the true objective function) | No independent step (occurs after L6, before the next L2) |

```python
from saealib import GA, LHSInitializer, SPEA2Comparator, Optimizer
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


N = 30  # offspring per generation, and the initial candidate pool size
N_BAR = 20  # SPEA2 external archive size

problem = zdt1(n_var=10)

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=2),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_comparator(SPEA2Comparator())
    .set_initializer(LHSInitializer(n_init_archive=N, n_init_population=N_BAR))
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy(n_offspring=N))
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

The `set_comparator(SPEA2Comparator())` call cannot be omitted.
In NSGA-II, `NSGA2Comparator` is the default for `n_obj > 1`, so the same line could be omitted, but this is not the case for SPEA2.

`ctx.pareto_archive` is saealib's own cumulative non-dominated archive, tracked across the entire run independently of any specific algorithm; it is not the same state container as SPEA2's external archive, which corresponds to `ctx.population`.
Read `ctx.population` directly if the paper's output $A$ (the non-dominated solutions in the final external archive $\bar P_{t+1}$) is specifically what's needed.

## Differences from the source

With the default `ParetoDominator` on an unconstrained problem, saealib reproduces SPEA2's core selection semantics — the strength/raw-fitness/density fitness assignment and the truncation-based environmental selection — as described in the paper.
Two aspects of the execution structure differ from the source.

**Initialization**: the paper generates an initial population $P_0$ (size $N$) and starts with an empty archive $\bar P_0 = \emptyset$; the first fitness assignment and environmental selection (L2, L3) only happen once the main loop begins at $t=0$.
saealib's `LHSInitializer` folds this initial fitness assignment and environmental selection into construction itself: it samples `n_init_archive` individuals (playing the role of the paper's $P_0$), true-evaluates them, and calls `rank_population()` to rank them before handing the top `n_init_population` to the initial `ctx.population`.
The resulting `ctx.population` at $t=0$ therefore already corresponds to the paper's post-selection archive $\bar P_1$, not to the empty $\bar P_0$.

**Constraint handling**: for constrained problems, `SPEA2Comparator` additionally applies Deb (2000)'s feasibility-first rule {cite}`deb2000feasibility` — feasible individuals always outrank infeasible ones, and infeasible individuals are ordered by ascending constraint violation.
The SPEA2 algorithm as published does not address constraint handling; this is a saealib-side extension.

## Parameters and variants

### Complexity

Fitness computation ($S(i)$/$R(i)$/$D(i)$) is $O(M^2)$ ($M=N+\bar N$), or $O(M^2\log M)$ including the distance sort needed for density estimation.
The truncation operator is $O(M^3)$ in the worst case and $O(M^2\log M)$ on average {cite}`zitzler2001spea2`.

**Sizing $N$ and $\bar N$ independently**: `ctx.population` plays the role of SPEA2's archive $\bar P$ — `GA.tell()` merges it with the offspring, and `TruncationSelection` (using the `SPEA2Comparator` configured with `set_comparator()`) performs environmental selection on the result — so `LHSInitializer(n_init_population=...)` sets $\bar N$, and `DirectStrategy(n_offspring=...)` sets the number of offspring generated (and true-evaluated) per generation, $N$.
`LHSInitializer`'s other size parameter, `n_init_archive`, sets the number of initially evaluated entries in saealib's own cumulative `Archive` (`ctx.archive`) — a different, longer-lived container than SPEA2's external archive, which keeps growing with every subsequent generation's evaluations.
At $t=0$ specifically it also sizes the candidate pool subjected to the first fitness assignment and environmental selection (see *Differences from the source* above), so a literature-faithful initialization sets `n_init_archive=N`.
Built-in initializers require `n_init_population <= n_init_archive`, so an initial state with $\bar N > N$ cannot be constructed directly.

**Swapping the dominator (dominance predicate)**: `SPEA2Comparator(dominator=...)` lets you inject a [Dominator](../concepts/problem_and_ranking/dominance.md) other than the default `ParetoDominator`.
Since the computation of $S(i)$/$R(i)$ depends on this dominance predicate, swapping it changes SPEA2's fitness itself.

## Related

- [References](../references.md): Full bibliographic details for the source
- [Comparator](../concepts/problem_and_ranking/comparators.md): Detailed specification of `SPEA2Comparator`
- [Crossover](../concepts/search_algorithms/crossover.md): List of crossover operators including `CrossoverSBX`
- [Mutation](../concepts/search_algorithms/mutation.md): List of mutation operators including `MutationPolynomial`
- [ParentSelection](../concepts/search_algorithms/parent_selection.md): Detailed usage of `TournamentSelection`
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md): Detailed usage of `TruncationSelection`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `DirectStrategy`
- [Dominator](../concepts/problem_and_ranking/dominance.md): List of dominance predicates that can be swapped in via the `dominator` argument
- [Initializer](../concepts/execution_and_evaluation/initialization.md): How `LHSInitializer`'s `n_init_archive`/`n_init_population` parameters size the initial archive and population
