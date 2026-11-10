# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Stage flow declarations** expose component connections and preserve named data transformations on compiler graph edges ([#319](https://github.com/saealib/saealib/issues/319))
- **History evaluation channel records `g` and `noise`**; checkpoints written without these columns load with them filled with NaN ([#303](https://github.com/saealib/saealib/issues/303))
- **`ComponentRef` and `TemplateStrategy`**: declared Stages reference components by `(slot, role)`, and each plan resolves its own copy of the declaration without modifying it ([#308](https://github.com/saealib/saealib/issues/308))
- **`objective_source` population column** records whether each row's objective values come from a true evaluation (`0`), a surrogate prediction (`1`), or an imputed fallback (`2`); `255` marks an unknown source. Populations and archives built by the library initializers always carry it, and `PSO` and `PymooAlgorithm` keep it when they rebuild the population ([#311](https://github.com/saealib/saealib/issues/311))
- **`Stage.execution_boundary`**: `"transaction"` (default) runs a Stage on its declared reads, `"state"` gives the Stage and its callbacks the whole run state; any other value raises `ValidationError` when the plan is prepared ([#309](https://github.com/saealib/saealib/issues/309))

### Deprecated

- **Direct `execute_strategy_step()` calls with asynchronous evaluation** emit a `DeprecationWarning`; use `Optimizer.iterate()` or `Optimizer.run()` ([#329](https://github.com/saealib/saealib/issues/329))
- **`Population.dense_view`, `Archive.dense_view`, and `ParetoArchive.dense_view`** emit a `DeprecationWarning`; use `dense_numeric_view` ([#329](https://github.com/saealib/saealib/issues/329))

### Removed

- **Unused execution protocol aliases, a private defaults forwarding method, and compiler rule re-exports outside `__all__`** were removed; use the canonical execution protocols, `Optimizer.resolve_defaults()`, or the implementation modules ([#329](https://github.com/saealib/saealib/issues/329))

### Changed

- **`FeatureEncoder` defaults** are selected during execution preparation when the prepared strategy consumes the encoder. `VectorSpace` uses identity encoding and reports INFO with its source and reason; `PermutationSpace` and `SequenceSpace` retain their built-in encodings and report a non-blocking WARNING recommending explicit selection because encoding affects surrogate learning. Explicit encoders supplied through `SearchSpace.services` take priority, and `minimize()`, `maximize()`, and existing examples continue to run without encoder arguments. Direct `encode_features()` calls use the same supported defaults and emit a Python `UserWarning` when no encoder is registered. Standalone `ArchiveObjectiveSet` builds against a context without a prepared encoder follow the same defaults and warning, while a prepared or explicitly registered encoder keeps priority without the default warning ([#334](https://github.com/saealib/saealib/issues/334))
- **Component defaults** are selected from ordered configuration sources during execution preparation, and compiler diagnostics report the selected source and reason; comparators expose facts used by composition rules, including `aligns_population_to_reference_points` for comparators whose default population size follows their reference points. The strength-ranked default-hint mechanism (`DefaultResolver`, `DefaultHintProvider`, `DefaultContext`, `BuiltinDefaultProvider`, and component `default_hints()`) was removed, and custom `default_hints()` implementations are no longer consulted; set components through setters or direct assignment, user or bundled presets, strategy defaults, or Stage-local components, and set sizes through `pop_size`/`max_fe`, an explicit initializer, or termination. An acquisition paired with a surrogate manager follows the manager that is actually selected, and an explicitly supplied acquisition keeps priority ([#255](https://github.com/saealib/saealib/issues/255))
- **Optimizer configuration checks** use structured diagnostics from the prepared component bindings for missing components, planner and feedback requirements, surrogate and acquisition needs, objective directions, and evaluator lifecycle consistency. Compiler flow diagnostics report ERROR for unregistered transforms, provable transform-chain contradictions, and known transform outputs incompatible with their destinations even when the source kind is unknown. `Optimizer.validate()` resolves defaults non-destructively, so an optimizer with unset components returns no validation errors, and it remains a `list[str]` of ERROR findings whose messages carry diagnostic codes; WARNING reports requirements that cannot be verified, and INFO and WARNING remain available through diagnostic access without blocking execution ([#333](https://github.com/saealib/saealib/issues/333))
- **Execution plans** share preparation logic: adapters, services, and component bindings are resolved against the problem used by the runtime, and runtime capabilities are read from its provider across startup, resume, replacement, recompilation, and direct strategy steps; direct steps reject verification ERROR diagnostics ([#317](https://github.com/saealib/saealib/issues/317))
- **Stage flow verification graphs** use occurrence-qualified call IDs such
  as `ask__proposer`; repeated region names receive occurrence suffixes. They
  include verification call and external state nodes in `build_component_graph`
  and preserve declared transforms and schema bindings. Unused operators are
  no longer represented as nodes, automatic `dense_numeric_view` adapters are
  no longer inserted, and schema variables keep their declared names. State
  edges connect each reader to the active producer across branch and loop
  joins. Transform declarations control schema binding carry-through, so
  `encode_features` does not preserve `representation`. The sequential runtime
  executes framework components with roles and skips nodes used only for graph
  verification.
  Stages with unknown flow report one `cannot_verify_stage_flow` warning;
  custom strategies should call `decompose_structured_graph` after lowering to
  include flow edges ([#319](https://github.com/saealib/saealib/issues/319))
- **Planned evaluation flow** now verifies evaluator observations through
  `EVALUATION_UPDATES` and `EVALUATED_OFFSPRING` using the declared transforms
  ([#319](https://github.com/saealib/saealib/issues/319))
- **Stage state writes** replace inferred producers with state sources, while
  unknown Stage flow clears producers that can no longer be verified
  ([#319](https://github.com/saealib/saealib/issues/319))
- **Surrogate state exports** are bound to declared component calls; manager
  internals no longer produce separate bindings unless they are exposed as
  separate calls ([#319](https://github.com/saealib/saealib/issues/319))
- **Checkpoints without the `objective_source` column** still load and resume; the reader adds it, marks rows true when `f` matches the candidate's last result-bearing true evaluation in the History `evaluation` channel, and marks other rows unknown ([#311](https://github.com/saealib/saealib/issues/311))
- **The compiler blocks a run only on evidence of a defect**: unregistered vocabulary values are reported as "cannot verify" warnings with near-miss suggestions, and a connection rejected only because of an unknown data kind or cardinality becomes a single `cannot_verify_data_kind` warning ([#298](https://github.com/saealib/saealib/issues/298))
- **Missing information no longer blocks a run**: several compatible candidates, edges to undeclared ports (`cannot_verify_port`), and unbound schema variables are warnings; a missing service, runtime capability, or genome codec is an error only when the real environment is supplied, and `CompileContext.offered_runtime_capabilities` now defaults to `None` ("not supplied") while an explicit empty set still means no capabilities ([#299](https://github.com/saealib/saealib/issues/299))
- **A `Stage` that does not override `contract()` exposes non-enumerable read and write effects** (through `StateContract.writes_enumerable`); when its flow is unknown, it may write any state key at run time, so a later read is not rejected solely because its write cannot be enumerated. The compiler reports `cannot_verify_stage_flow` for the Stage ([#300](https://github.com/saealib/saealib/issues/300), [#319](https://github.com/saealib/saealib/issues/319))
- **`GenerationBasedStrategy` Stage callbacks receive the whole run state as `event.ctx`**; values that were readable before are unchanged ([#309](https://github.com/saealib/saealib/issues/309))
- **`strategy.pipeline` holds the most recently prepared Pipeline** ([#309](https://github.com/saealib/saealib/issues/309))
- **`GenerationBasedStrategy` rejects a non-checkpointable algorithm before the run** when npz checkpointing is requested ([#309](https://github.com/saealib/saealib/issues/309))
- **`DirectStrategy.build_graph()` returns a `StructuredGraph`**; GenerationBased and template graphs get component part nodes, the adapters the compiler chooses, and contract diagnostics, so an algorithm requiring an unoffered runtime capability now stops a GenerationBased run before it starts ([#309](https://github.com/saealib/saealib/issues/309))
- **`IndividualBasedStrategy.build_graph()` and `PreSelectionStrategy.build_graph()` return a `StructuredGraph`**; node IDs are unchanged ([#314](https://github.com/saealib/saealib/issues/314))
- **`SteadyStateStrategy.build_graph()` returns a `StructuredGraph`**; node IDs are unchanged ([#315](https://github.com/saealib/saealib/issues/315))
- **`SteadyStateStrategy.pipeline` holds the most recently prepared `Pipeline`** ([#315](https://github.com/saealib/saealib/issues/315))
- **Asynchronous structured flows without regions refill available evaluation slots whenever their strategy opts into `supports_async_refill`**, including advances with no newly delivered result ([#315](https://github.com/saealib/saealib/issues/315))
- **`IndividualBasedStrategy` and `PreSelectionStrategy` honor subclass `build_pipeline()` overrides when running**; `strategy.pipeline` holds the most recently prepared `Pipeline` ([#314](https://github.com/saealib/saealib/issues/314))

### Fixed

- **Built-in strategy pipeline replacements** affect prepared runs at the next safe commit point, including changes made in nested flows; asynchronous runs wait for outstanding requests to finish, and replaced ask stages keep their own count ([#328](https://github.com/saealib/saealib/issues/328))
- **Checkpoint save points**: the initial `iterate()` state and resumable boundaries can be saved, asynchronous wait points are accepted only when resume can reconstruct them, pending requests are checked for reattachment and feedback consistency, a `GenerationBasedStrategy` run-end callback with an open region is accepted when its evaluation plan has no deferred requests and rejected otherwise, schema-1 acquisition results restored from pickle can be saved in portable format, and portable checkpoints do not require a serializable evaluator; asynchronous `GenerationBasedStrategy` runs that terminate inside a generation while true evaluations remain in flight may not resume to the same result from the state returned by `run()`, so save at `GenerationEndEvent` or from a `RunEndEvent` callback ([#325](https://github.com/saealib/saealib/issues/325))
- **Compiler diagnostics** report acquisition channels whose surrogate output lacks required uncertainty, is unavailable, or cannot be verified; `Optimizer.validate()` uses the same check before compilation ([#320](https://github.com/saealib/saealib/issues/320))
- **Direct asynchronous strategy steps** block until in-flight requests finish and return without running the step; the next call uses the current components. An explicit recompile requested while requests are in flight is retained until they finish ([#317](https://github.com/saealib/saealib/issues/317))
- **Strategy and Pipeline specs** preserve nested declarations, controls, component references, and explicit stage components; nested values build recursively, and unsupported values report their location. `to_spec` rejects non-importable local classes and shared instances with a location; `Optimizer.save_preset` uses the same validation ([#318](https://github.com/saealib/saealib/issues/318)).
- **Pickle checkpoints for asynchronous evaluation**: `EvaluationPlan` serializes its immutable artifacts mapping so runs can resume from pickle checkpoints ([#313](https://github.com/saealib/saealib/issues/313))
- **Compiling a graph that refers to a missing node** reports `invalid_graph_edge`, `invalid_entry_point`, or the new `invalid_state_binding` instead of raising `KeyError` ([#295](https://github.com/saealib/saealib/issues/295))
- **Branches at the start or end of a lowered sequence**: the else path is no longer reported as unreachable and the then path is ordered before what follows ([#296](https://github.com/saealib/saealib/issues/296))
- **Inputs after a branch** supplied by a different component on each path are no longer reported as unresolved ([#297](https://github.com/saealib/saealib/issues/297))
- **`to_spec` round-trips** classes without their own `__init__`, `**kwargs` settings (restored from a same-named mapping attribute; a class that does not keep or cannot represent them now raises `ValidationError` instead of silently dropping them), and the `NSGA3Comparator` seed ([#301](https://github.com/saealib/saealib/issues/301))
- **Comparator sort caches** are keyed by the full comparison criterion, and per instance for NSGA-III, so another comparator, an in-place `eps_cv` change or built-in dominator setting, or an independent NSGA-III random stream no longer returns a stale order ([#302](https://github.com/saealib/saealib/issues/302))
- **Components replaced during `run()` / `iterate()`** take effect at the next commit point in sync and async runs: replacing the algorithm (with `set_algorithm()` or by assignment) or another component switches ask and tell together, so they never use different algorithms ([#286](https://github.com/saealib/saealib/issues/286))
- **Replacing `DirectStrategy` with `GenerationBasedStrategy`, or the reverse, during a run** works ([#309](https://github.com/saealib/saealib/issues/309))
- **Replacing `IndividualBasedStrategy` or `PreSelectionStrategy` with `DirectStrategy` or `GenerationBasedStrategy`, or the reverse, during a run** no longer stops with a `ValidationError`, in sync and async runs ([#314](https://github.com/saealib/saealib/issues/314))
- **Replacing `SteadyStateStrategy` with another built-in strategy, or the reverse, during a run** no longer stops with a `ValidationError`, in sync and async runs ([#315](https://github.com/saealib/saealib/issues/315))
- **`SteadyStateStrategy` resumes npz checkpoints loaded with a different `Problem` instance** without a `BoundsService` conflict ([#315](https://github.com/saealib/saealib/issues/315))
- **The GenerationBased true-evaluation generation uses the Optimizer's feedback builder only when it is set explicitly**, in sync and async runs alike; async runs used an attribute-assigned builder ([#309](https://github.com/saealib/saealib/issues/309))
- **GenerationBased async runs respect `AsyncEvaluationScheduler.max_pending`** instead of submitting a whole generation at once; results change where the limit binds ([#309](https://github.com/saealib/saealib/issues/309))
- **Async runs of a `TemplateStrategy` without regions** continue until the termination condition ([#309](https://github.com/saealib/saealib/issues/309))
- **Direct async runs resume from an npz checkpoint**; restoring one failed with a conflicting `BoundsService` binding ([#309](https://github.com/saealib/saealib/issues/309))
- **`IndividualBasedStrategy` and `PreSelectionStrategy` resume from npz checkpoints attached to a different `Problem` instance** without a `BoundsService` conflict ([#314](https://github.com/saealib/saealib/issues/314))

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
