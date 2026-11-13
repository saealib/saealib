---
primary_layer: layer2
related_layers: []
page_type: concept
---

# SMS-EMOA (S Metric Selection EMOA)

SMS-EMOA is a steady-state multi-objective evolutionary algorithm that incorporates dominated hypervolume (the $\mathcal{S}$ metric) directly into the selection criterion.
After splitting the population into fronts via non-dominated sorting, it culls one individual at a time — the one with the smallest hypervolume contribution within the lowest front — monotonically increasing the population's overall dominated hypervolume with each generation.

## Overview

The hypervolume indicator is widely used as a measure of the quality of a Pareto-front approximation.
Fixing a reference point $\mathbf{y}_{\mathrm{ref}}$ lets you define the Lebesgue measure $\mathcal{S}(B, \mathbf{y}_{\mathrm{ref}})$ of the region dominated by a solution set $B$, and it is known that maximizing $\mathcal{S}$ for a finite Pareto-front approximation is equivalent to finding the true Pareto set.

SMS-EMOA doesn't stop at using this hypervolume indicator for evaluation — it adopts it as the selection operator itself.
Using the same **non-dominated sorting** as NSGA-II, it splits the population into fronts $\mathcal{R}_1, \ldots, \mathcal{R}_v$, and within the lowest front $\mathcal{R}_v$, culls the single individual whose removal causes the smallest decrease in the $\mathcal{S}$ metric.
This decrease is called the **exclusive hypervolume contribution**, $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) := \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$.

Because hypervolume computation is expensive, SMS-EMOA uses **steady-state** generational replacement.
Each generation generates only a single new individual via crossover and mutation, and culls a single existing individual to keep the population size $\mu$ constant.
This avoids having to compare $\binom{\mu+\lambda}{\mu}$ combinations as in $(\mu+\lambda)$ generational replacement, keeping $\mathcal{S}$-metric evaluations within the lowest front to at most $\mu+1$.

The source is {cite}`beume2007smsemoa`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} SMS-EMOA
:label: alg-sms-emoa

**Inputs** objective functions, population size $\mu$, initial population $P_0$
**Output** the population $P_{t+1}$ of the final generation

1. Set $t=0$; generate an initial population $P_0$ of $\mu$ individuals
2. Generate a single new individual $q_{t+1}$ from $P_t$ via crossover and mutation
3. Non-dominated-sort $Q = P_t \cup \{q_{t+1}\}$ (size $\mu+1$) to obtain the front sequence $\mathcal{R}_1, \ldots, \mathcal{R}_v$
4. Identify the lowest front $\mathcal{R}_v$ (if $|\mathcal{R}_v|=1$, that single individual is the one to be culled)
5. If $|\mathcal{R}_v| > 1$, compute the exclusive hypervolume contribution $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) = \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$ for each individual $s$ in $\mathcal{R}_v$, and choose the individual $r$ with the smallest value
6. Set $P_{t+1} = Q \setminus \{r\}$ (which always satisfies $\mathcal{S}(P_t) \leq \mathcal{S}(P_{t+1})$)
7. Set $t=t+1$ and return to step 2, repeating until the termination condition is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Generate initial population P0 of μ individuals<br/>(L1)"] --> GEN
    subgraph GEN["One generation (SteadyStateStrategy.step)"]
        direction TB
        ASK["GA.ask(n_offspring=1)<br/>Randomly select parent →<br/>SBX crossover →<br/>Polynomial mutation to generate one new individual q_t+1<br/>(L2)"] --> EVAL["True evaluation<br/>(no surrogate involved)"]
        EVAL --> COMB["GA.tell()<br/>Combine Q = Pt ∪ {q_t+1}<br/>(L3)"]
        COMB --> SORT["HypervolumeComparator.sort_population()<br/>Non-dominated sorting →<br/>HV contribution within front<br/>(L3-5)"]
        SORT --> TRUNC["TruncationSelection<br/>Cull the last individual<br/>(smallest contribution in the lowest front)<br/>(L4-6)"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L7)" --> GEN
    TERM -- "Reached" --> RESULT(["Population of the final generation"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (`ask(n_offspring=1)` generates only one new individual; `tell()` performs combining $Q=P_t\cup\{q_{t+1}\}$ and survivor selection) | L2-3, L6 |
| Parent selection | `TournamentSelection(tournament_size=1)` (uniform random selection with no comparison; the paper does not specify a parent-selection scheme) | L2 |
| Crossover | `CrossoverSBX(prob=0.9, eta=20.0)` | L2 |
| Mutation | `MutationPolynomial(eta=20.0)` | L2 |
| Non-dominated sorting + in-front HV contribution | `HypervolumeComparator` (`sort_population` internally calls non-dominated sorting and `hypervolume_contributions`) | L3-5 |
| Survivor selection | `TruncationSelection()` (culls the last individual in `comparator.sort_population`'s order — i.e. the one with the smallest contribution in the lowest front) | L4-6 |
| Evaluation strategy | `SteadyStateStrategy` with `SerialEvaluator` (sequential evaluation, equivalent to `max_workers=1`, `max_pending=1`) | L2, L6-7 (in-generation evaluation) |

`SteadyStateStrategy` requests one candidate per step. The faithful configuration below uses sequential evaluation; asynchronous evaluation is described as a separate extension.

```python
from saealib import (
    GA,
    HypervolumeComparator,
    Optimizer,
    SerialEvaluator,
    SteadyStateStrategy,
)
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.termination import Termination, max_fe


problem = zdt1(n_var=10)

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=1),
    TruncationSelection(),
)
evaluator = SerialEvaluator()

opt = (
    Optimizer(problem)
    .set_comparator(HypervolumeComparator())
    .set_algorithm(algorithm)
    .set_strategy(SteadyStateStrategy())
    .set_evaluator(evaluator)
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

The `set_comparator(HypervolumeComparator())` call cannot be omitted.
In NSGA-II, `NSGA2Comparator` is the default for `n_obj > 1`, so the same line could be omitted, but as with SPEA2 and NSGA-III, SMS-EMOA also requires an explicit assignment.

**Asynchronous extension**: When evaluation times vary, combine `SteadyStateStrategy` with `AsyncEvaluator` and `AsyncEvaluationScheduler` (for example, `max_workers=2`, `max_pending=2`). This extension has a different evaluation order from the sequential configuration above.

```python
from saealib import AsyncEvaluator, AsyncEvaluationScheduler

async_evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=2)
async_schedule = AsyncEvaluationScheduler(async_evaluator, max_pending=2)
async_opt = (
    Optimizer(problem)
    .set_comparator(HypervolumeComparator())
    .set_algorithm(algorithm)
    .set_strategy(SteadyStateStrategy())
    .set_evaluator(async_evaluator)
    .set_async_evaluation_scheduler(async_schedule)
)
```

## Differences from the source

The original SMS-EMOA uses a steady-state procedure that evaluates one individual at a time.
This design limits hypervolume calculations for the lowest front to at most $\mu+1$.
The configuration above generates one individual at a time with `SteadyStateStrategy`. If `DirectStrategy` generates several individuals per generation instead, `HypervolumeComparator` ranks hypervolume contributions across every front, so the configuration no longer follows the original procedure that targets only the lowest front.
The original procedure does not specify parent selection, so the example uses `TournamentSelection(tournament_size=1)`, which performs no comparison.
`HypervolumeComparator` also does not implement the SMS-EMOA dp variant described in the paper; it provides only the base hypervolume contribution.

## Parameters and variants

### Complexity

Hypervolume computation itself is polynomial in the number of points but exponential in the number of objectives.
saealib's `hypervolume` (recursive slicing) is $O(n^{m-1} n \log n)$ ($n$ the number of points, $m$ the number of objectives).

The exclusive contribution requires $k$ leave-one-out HV computations for a front of size $k$, so computing one front costs $O(k^{m} \log k)$.
Because the paper's reduction procedure applies this only to the lowest front (size at most $\mu+1$), it stays within $O(\mu^{m} \log \mu)$ per generation {cite}`beume2007smsemoa`.

`HypervolumeComparator` generalizes this by computing contributions across all fronts, but since the sum of front sizes never exceeds $\mu+1$, the asymptotic upper bound remains $O(\mu^{m} \log \mu)$.

**Alternative reduce procedure ("SMS-EMOA dp")**: The paper also proposes a faster variant using the domination count $d(s, P(t))$ instead of hypervolume contribution.

**Handling the reference point**: `HypervolumeComparator(reference_point=...)` accepts a fixed value.
With `None`, it computes a reference point for each front in each generation according to the paper's definition of adding 1.0 to the worst value of each objective.
For two objectives, it follows the paper's treatment of retaining the two boundary solutions without computing a reference point.

**Swapping the dominator (dominance predicate)**: `HypervolumeComparator(reference_point=..., dominator=...)` lets you inject a [Dominator](../concepts/problem_and_ranking/dominance.md) other than the default `ParetoDominator`.
Since this changes the result of non-dominated sorting, the population subjected to front splitting and contribution computation also depends on this dominance predicate.

## Related

- [References](../references.md): Full bibliographic details for the source
- [Comparator](../concepts/problem_and_ranking/comparators.md): Detailed specification of `HypervolumeComparator`, and how population-relative comparators are handled
- [Crossover](../concepts/search_algorithms/crossover.md): List of crossover operators including `CrossoverSBX`
- [Mutation](../concepts/search_algorithms/mutation.md): List of mutation operators including `MutationPolynomial`
- [ParentSelection](../concepts/search_algorithms/parent_selection.md): Detailed usage of `TournamentSelection`
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md): Detailed usage of `TruncationSelection`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): Implementing a custom Strategy, and `AskStage`'s `n_offspring`
- [NonDominatedSorting](../concepts/problem_and_ranking/nondominated_sorting.md): Implementation details of non-dominated sorting
- [Dominator](../concepts/problem_and_ranking/dominance.md): List of dominance predicates that can be swapped in via the `dominator` argument
