# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **History evaluation channel records `g` and `noise`**; checkpoints written without these columns load with them filled with NaN ([#303](https://github.com/saealib/saealib/issues/303))

### Changed

- **The compiler blocks a run only on evidence of a defect**: unregistered vocabulary values are reported as "cannot verify" warnings with near-miss suggestions, and a connection rejected only because of an unknown data kind or cardinality becomes a single `cannot_verify_data_kind` warning ([#298](https://github.com/saealib/saealib/issues/298))
- **Missing information no longer blocks a run**: several compatible candidates, edges to undeclared ports (`cannot_verify_port`), and unbound schema variables are warnings; a missing service, runtime capability, or genome codec is an error only when the real environment is supplied, and `CompileContext.offered_runtime_capabilities` now defaults to `None` ("not supplied") while an explicit empty set still means no capabilities ([#299](https://github.com/saealib/saealib/issues/299))
- **A `Stage` that does not override `contract()` is opaque**: its reads and writes are not enumerated (new `StateContract.writes_enumerable`), it may write any state key at run time, and the compiler reports one `opaque_state_effects` warning; declare a contract to keep strict write checks ([#300](https://github.com/saealib/saealib/issues/300))

### Fixed

- **Compiling a graph that refers to a missing node** reports `invalid_graph_edge`, `invalid_entry_point`, or the new `invalid_state_binding` instead of raising `KeyError` ([#295](https://github.com/saealib/saealib/issues/295))
- **Branches at the start or end of a lowered sequence**: the else path is no longer reported as unreachable and the then path is ordered before what follows ([#296](https://github.com/saealib/saealib/issues/296))
- **Inputs after a branch** supplied by a different component on each path are no longer reported as unresolved ([#297](https://github.com/saealib/saealib/issues/297))
- **`to_spec` round-trips** classes without their own `__init__`, `**kwargs` settings (restored from a same-named mapping attribute; a class that does not keep or cannot represent them now raises `ValidationError` instead of silently dropping them), and the `NSGA3Comparator` seed ([#301](https://github.com/saealib/saealib/issues/301))
- **Comparator sort caches** are keyed by the full comparison criterion, and per instance for NSGA-III, so another comparator, an in-place `eps_cv` change or built-in dominator setting, or an independent NSGA-III random stream no longer returns a stale order ([#302](https://github.com/saealib/saealib/issues/302))
- **Components replaced during `run()` / `iterate()`** take effect at the next commit point in sync and async runs: replacing the algorithm (with `set_algorithm()` or by assignment) or another component switches ask and tell together, so they never use different algorithms ([#286](https://github.com/saealib/saealib/issues/286))

## [0.1.0] - 2026-09-03

Framework redesign, experiment execution, and result analysis.

### Added

- **Modular framework**: component contracts, graph/plan compilation, and an explicit runtime boundary, replacing the implicit wiring between algorithm, strategy, and surrogate ([#226](https://github.com/saealib/saealib/issues/226)) ([#234](https://github.com/saealib/saealib/issues/234))
- **Execution history**: per-generation recording of summary, front, population, surrogate accuracy, decision, and evaluation channels ([#228](https://github.com/saealib/saealib/issues/228))
- **Performance indicators**: IGD, IGD+, GD, GD+, spacing, and generalized spread ([#229](https://github.com/saealib/saealib/issues/229))
- **Result analysis and visualization**: `Result.history_series()` with a built-in value registry, and `saealib.viz` (`plot_result` / `plot_history`) behind the `viz` extra ([#231](https://github.com/saealib/saealib/issues/231))
- **Experiment execution**: `saealib.experiment` for multi-trial sweeps over problems, algorithms, and seeds, with YAML configuration, parallel execution, per-trial checkpointing and resume, progress reporting, summary and aggregate output, and a `saealib` command-line entry point ([#230](https://github.com/saealib/saealib/issues/230))
- **pymoo adapters**: `PymooCrossover`, `PymooMutation`, `PymooAlgorithm`, and `PymooProblem` ([#222](https://github.com/saealib/saealib/issues/222))
- **DEAP adapters**: `DeapCrossover`, `DeapMutation`, and `DeapGenerateUpdateAlgorithm`, bridging saealib's generator to DEAP's global random state ([#262](https://github.com/saealib/saealib/issues/262))
- **Asynchronous evaluation**: a resumable scheduler for out-of-order and partial evaluation results ([#226](https://github.com/saealib/saealib/issues/226))
- **`spacing(squared=...)`**: Schott's variance form alongside the standard-deviation form the surrounding ecosystem reports ([#263](https://github.com/saealib/saealib/issues/263))
- **`HypervolumeComparator(last_front_only=...)`**: restricts hypervolume contributions to the last front, as in the original SMS-EMOA ([#263](https://github.com/saealib/saealib/issues/263))
- **Optional extras**: `viz`, `tqdm`, `rich`, `hdf5`, `deap`

### Changed

- **`AcquisitionFunction` separated from `SurrogateManager`** (breaking): scoring and fit/predict coordination are now distinct responsibilities ([#226](https://github.com/saealib/saealib/issues/226))
- **Top-level `Registry` surface reduced to `register`** (breaking) ([#203](https://github.com/saealib/saealib/issues/203))
- **APIs deprecated for removal in 0.1.0 were removed** (breaking) ([#205](https://github.com/saealib/saealib/issues/205))
- **Checkpoint state is versioned with per-entry migration**, so an older checkpoint stays readable as the schema moves ([#226](https://github.com/saealib/saealib/issues/226))
- **Portable checkpointing is refused for components holding non-exportable state**, naming the component, instead of silently resuming from an incomplete state ([#264](https://github.com/saealib/saealib/issues/264))

### Fixed

- **`RBFSurrogate` polynomial term**, without which the CORS-RBF paper configuration could not be reproduced ([#235](https://github.com/saealib/saealib/issues/235))
- **`LowerConfidenceBound` beta_t schedule**, restoring the GP-UCB theoretical guarantee ([#236](https://github.com/saealib/saealib/issues/236))
- **SPEA2 fitness persistence** across environmental and mating selection ([#237](https://github.com/saealib/saealib/issues/237))
- **CORS beta cadence**, which advanced per generation rather than per evaluation ([#238](https://github.com/saealib/saealib/issues/238))
- **NSGA-III default population size**, which did not scale with the reference-point count ([#239](https://github.com/saealib/saealib/issues/239))
- **Comparator fidelity to their source papers** across SPEA2, R-NSGA-II, NSGA-III, MaxUncertainty, CORS, and the hypervolume comparator ([#209](https://github.com/saealib/saealib/issues/209)) ([#210](https://github.com/saealib/saealib/issues/210)) ([#211](https://github.com/saealib/saealib/issues/211)) ([#212](https://github.com/saealib/saealib/issues/212)) ([#213](https://github.com/saealib/saealib/issues/213)) ([#214](https://github.com/saealib/saealib/issues/214))
- **npz checkpointing**, which aborted the run when the surrogate predicted a non-finite value ([#265](https://github.com/saealib/saealib/issues/265))
- **Eight example scripts** that had failed on import since the `weight` to `direction` unification, and CI now runs every example ([#253](https://github.com/saealib/saealib/issues/253))

## [0.1.0b4] - 2026-06-15

Multi-objective optimization and constraint surrogate — full MOO algorithm stack, decomposition strategies, and surrogate-assisted constraint handling.

### Added

- **Pareto Archive**: non-dominated solution archive maintaining the current Pareto front across generations ([#76](https://github.com/saealib/saealib/issues/76))
- **Multi-objective comparators**: SPEA2 fitness, hypervolume contribution, and ε-dominance comparators for population selection ([#74](https://github.com/saealib/saealib/issues/74))
- **MOO acquisition functions**: ParEGO (scalarized EI), SMS-EGO (hypervolume-based), and HV-based acquisition for multi-objective surrogate-assisted search ([#75](https://github.com/saealib/saealib/issues/75))
- **MOEA/D decomposition strategies**: Tchebycheff, PBI, and WeightedSum decomposition for the MOEA/D framework ([#88](https://github.com/saealib/saealib/issues/88))
- **Reference-point-based MOO**: NSGA-III and R-NSGA-II algorithms for many-objective optimization ([#77](https://github.com/saealib/saealib/issues/77))
- **`EpsilonConstraintHandler`**: dynamic ε-constraint method that progressively tightens the feasibility threshold over generations ([#109](https://github.com/saealib/saealib/issues/109))
- **`GradientRepairHandler`**: gradient-based constraint repair that projects infeasible candidates onto the feasible boundary ([#110](https://github.com/saealib/saealib/issues/110))
- **Constraint surrogate path**: `ConstraintObjectiveSet` and `KNNConstraintObjectiveSet` training sets for `g`-value surrogate training; `ProductOfFeasibility` acquisition function combining per-constraint PoF scores into a joint feasibility score ([#86](https://github.com/saealib/saealib/issues/86))

### Changed

- **`CompositeSurrogateManager` replaces `EnsembleSurrogateManager`** (breaking): composable manager that combines independent surrogate managers (e.g., f-surrogate × PoF-manager) with explicit score aggregation
- **`direction` replaces `weights` in Pareto-based comparators** (breaking): comparators now accept `direction` (±1 per objective) to declare minimization/maximization per dimension, decoupling sign convention from scalarization magnitude ([#118](https://github.com/saealib/saealib/issues/118))

### Performance

- Efficient Non-Dominated Sort: eliminated O(N²) Python loop, replacing with a vectorized dominance check ([#89](https://github.com/saealib/saealib/issues/89))

## [0.1.0b3] - 2026-06-08

Evaluation and constraint refactors — modular evaluation, constraint handling, and composable termination.

### Added

- **`Evaluator` abstraction**: pluggable evaluation interface with a batch evaluation API, decoupling true evaluation from the optimization loop ([#85](https://github.com/saealib/saealib/issues/85))
- **`EqualityConstraint`**: equality constraint support alongside inequality constraints ([#87](https://github.com/saealib/saealib/issues/87))
- **Composable `Termination`**: logical composition of termination conditions via `&` / `|` / `~` operators (and `all_of` / `any_of` / `not_` helpers); `max_fe` / `max_gen` now return a `TerminationCondition`, plus `f_target` and `stalled` built-in conditions ([#94](https://github.com/saealib/saealib/issues/94))

### Changed

- **`ConstraintHandler` abstraction** (breaking): constraint processing extracted into a modular handler, replacing inline constraint logic ([#108](https://github.com/saealib/saealib/issues/108))
- **Pareto comparator sign convention unified** (breaking): consistent sign handling so the Pareto comparator supports maximization ([#93](https://github.com/saealib/saealib/issues/93))
- **`Problem.eps` split into `eps_cv` and `eps_obj`** (breaking): separate tolerances for constraint violation and objective value ([#92](https://github.com/saealib/saealib/issues/92))

### Performance

- Eliminated the `np.vstack` loop in `GA.ask`, removing O(N²) reallocation ([#91](https://github.com/saealib/saealib/issues/91))

## [0.1.0b2] - 2026-05-21

Surrogate API flexibility — concentrated breaking changes to decouple surrogate output from objective semantics.

### Added

- **`GPSurrogate`**: Gaussian Process surrogate providing mean and uncertainty estimates; `SurrogatePrediction.std` is populated and usable with EI, LCB, PoF, and MaxUncertainty acquisition functions ([#78](https://github.com/saealib/saealib/issues/78))
- **Sklearn / torch adapter**: `SklearnSurrogate` wraps any scikit-learn estimator; `SVMSurrogate`, `NNSurrogate` (MLP), and `DTSurrogate` (Random Forest) added ([#72](https://github.com/saealib/saealib/issues/72))
- **`ArchiveBasedManager`**: abstract base for archive-direct scoring without a surrogate model; concrete implementations: `NoveltyManager` (nearest-neighbor distance), `DensityManager` (inverse ε-NN density), `NichingManager` (inter-candidate distance) ([#84](https://github.com/saealib/saealib/issues/84))
- **`Optimizer.validate()`**: pre-run configuration consistency check — validates operator presence, strategy/surrogate/algorithm compatibility, and reports misconfigurations with actionable error messages ([#73](https://github.com/saealib/saealib/issues/73))
- **Training data abstraction**: `TrainingData` / `TrainingSet` builder with `PairwiseComparisonSet`, `TopKBipartitionSet`, `LevelBasedSet`, and `FeasibilityClassificationSet` labelling strategies for flexible surrogate training data construction ([#67](https://github.com/saealib/saealib/issues/67))

### Fixed

- Intercept `NaN` surrogate predictions at `SurrogateManager` boundary and `OptimizationStrategy` to prevent silent downstream propagation

### Changed

- **`SurrogatePrediction.mean` renamed to `SurrogatePrediction.value`** (breaking): decouples raw surrogate output from the objective value written to offspring, preventing pbest corruption when non-regression surrogates (e.g., novelty scores, classification probabilities) are used ([#69](https://github.com/saealib/saealib/issues/69))
- **`SurrogatePrediction.tell_f` added** (breaking): strategies now assign objective function values via `tell_f` rather than `mean`, separating surrogate output semantics from tell semantics ([#69](https://github.com/saealib/saealib/issues/69))

## [0.1.0b1] - 2026-05-03

Initial beta release of saealib.

### Added

- **High-level API**: `minimize()` and `maximize()` functions for quick setup with sensible defaults
- **Low-level API**: `Optimizer` builder with swappable components via `set_algorithm()`, `set_surrogate_manager()`, and `set_strategy()`
- **Algorithms**: Genetic Algorithm (`GA`) with crossover, mutation, and selection operators
- **Surrogate models**: RBF (Radial Basis Function) surrogate with configurable kernels
- **Acquisition functions**: pluggable acquisition function interface with built-in implementations
- **Optimization strategies**: `IndividualBasedStrategy`, `GenerationBasedStrategy`, `PreSelectionStrategy`
- **Multi-objective support**: Pareto-based archive and multi-objective problem interface
- **Constraint handling**: feasibility-aware selection and result reporting
- **Callbacks**: extensible callback interface for monitoring and early stopping
- **Archive**: solution archive with sorting and feasibility tracking
- **Problem definition**: `Problem` class supporting minimization (`weight=-1`) and maximization (`weight=+1`)
- **Termination criteria**: configurable stopping conditions
- **Type hints**: fully typed public API, PEP 561 compliant (`py.typed` marker included)

[Unreleased]: https://github.com/saealib/saealib/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/saealib/saealib/compare/v0.1.0b4...v0.1.0
[0.1.0b4]: https://github.com/saealib/saealib/compare/v0.1.0b3...v0.1.0b4
[0.1.0b3]: https://github.com/saealib/saealib/compare/v0.1.0b2...v0.1.0b3
[0.1.0b2]: https://github.com/saealib/saealib/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/saealib/saealib/releases/tag/v0.1.0b1
