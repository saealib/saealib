"""End-to-end tests for constraint surrogate integration."""

from __future__ import annotations

import numpy as np
import pytest

from saealib import (
    GA,
    CompositeAcquisition,
    CompositeSurrogateManager,
    CrossoverBLXAlpha,
    ExpectedImprovement,
    GlobalSurrogateManager,
    InequalityConstraint,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    PerObjectiveSurrogate,
    PreSelectionStrategy,
    Problem,
    ProductOfFeasibility,
    SequentialSelection,
    StaticToleranceHandler,
    Termination,
    TruncationSelection,
    max_fe,
    product_combine,
)
from saealib.surrogate import SklearnGPRSurrogate
from saealib.surrogate.training_set import ArchiveObjectiveSet, ConstraintObjectiveSet

# ---------------------------------------------------------------------------
# G01 benchmark definition
# ---------------------------------------------------------------------------

_DIM = 13
_N_CONSTRAINTS = 9


def _g01_obj(x: np.ndarray) -> np.ndarray:
    return np.array([5.0 * np.sum(x[:4]) - 5.0 * np.sum(x[:4] ** 2) - np.sum(x[4:13])])


_G01_CONSTRAINTS = [
    InequalityConstraint(lambda x: 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10.0),
    InequalityConstraint(lambda x: 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10.0),
    InequalityConstraint(lambda x: 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10.0),
    InequalityConstraint(lambda x: -8.0 * x[0] + x[9]),
    InequalityConstraint(lambda x: -8.0 * x[1] + x[10]),
    InequalityConstraint(lambda x: -8.0 * x[2] + x[11]),
    InequalityConstraint(lambda x: -2.0 * x[3] - x[4] + x[9]),
    InequalityConstraint(lambda x: -2.0 * x[5] - x[6] + x[10]),
    InequalityConstraint(lambda x: -2.0 * x[7] - x[8] + x[11]),
]


def _make_g01_problem() -> Problem:
    return Problem(
        func=_g01_obj,
        dim=_DIM,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0] * _DIM,
        ub=[1.0] * _DIM,
        constraints=_G01_CONSTRAINTS,
        handler=StaticToleranceHandler(eps_cv=1e-6),
    )


def _make_ga() -> GA:
    return GA(
        crossover=CrossoverBLXAlpha(prob=0.9, alpha=0.4),
        mutation=MutationUniform(prob_var=0.1),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )


# ---------------------------------------------------------------------------
# G01 end-to-end test
# ---------------------------------------------------------------------------


class TestConstraintSurrogateG01:
    """End-to-end test: EI x ProductOfFeasibility on G01."""

    def test_optimizer_runs_without_error(self) -> None:
        """CompositeSurrogateManager completes 5 generations on G01."""
        problem = _make_g01_problem()

        ei_mgr = GlobalSurrogateManager(
            SklearnGPRSurrogate(),
            training_set=ArchiveObjectiveSet(),
        )
        ei_acq = ExpectedImprovement()
        pof_mgr = GlobalSurrogateManager(
            PerObjectiveSurrogate(
                [SklearnGPRSurrogate() for _ in range(_N_CONSTRAINTS)]
            ),
            training_set=ConstraintObjectiveSet(),
        )
        pof_acq = ProductOfFeasibility()
        composite_acq = CompositeAcquisition(
            {"objective": ei_acq, "feasibility": pof_acq},
            combine_fn=product_combine,
        )
        surrogate_manager = CompositeSurrogateManager(
            {"objective": ei_mgr, "feasibility": pof_mgr},
        )

        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=20, n_init_population=10, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=20, n_select=3))
            .set_surrogate_manager(surrogate_manager)
            .set_acquisition(composite_acq)
            .set_termination(Termination(max_fe(40)))
        )
        ctx = optimizer.run()
        assert ctx is not None

    def test_constraint_surrogate_trains_on_g(self) -> None:
        """ConstraintObjectiveSet produces train_y with n_constraints columns."""
        problem = _make_g01_problem()
        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=15, n_init_population=10, seed=1)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=20, n_select=3))
            .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
            .set_acquisition(ExpectedImprovement())
            .set_termination(Termination(max_fe(15)))
        )
        # Run initialisation only (max_fe equals initial archive sample size)
        ctx = optimizer.run()
        archive = ctx.archive
        ts = ConstraintObjectiveSet()
        data = ts.build(archive, None, ctx=None)
        assert data.train_y.shape == (len(archive), _N_CONSTRAINTS)

    def test_feasible_solution_found(self) -> None:
        """With enough budget, at least one feasible solution is archived.

        Note: the continuous [0, 1]^13 variant of G01 has feasibility rate ~64.6 %
        (g1-g3 are always satisfied; the original binary x[10..12] reduce it to
        0.011 %). This test is therefore a smoke test for the EI x PoF pipeline,
        not a surrogate-guidance validation. See TestConstraintSurrogate2D for a
        direct test of PoF scoring quality.
        """
        problem = _make_g01_problem()

        ei_mgr = GlobalSurrogateManager(
            SklearnGPRSurrogate(),
            training_set=ArchiveObjectiveSet(),
        )
        ei_acq = ExpectedImprovement()
        pof_mgr = GlobalSurrogateManager(
            PerObjectiveSurrogate(
                [SklearnGPRSurrogate() for _ in range(_N_CONSTRAINTS)]
            ),
            training_set=ConstraintObjectiveSet(),
        )
        pof_acq = ProductOfFeasibility()
        composite_acq = CompositeAcquisition(
            {"objective": ei_acq, "feasibility": pof_acq},
            combine_fn=product_combine,
        )

        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=30, n_init_population=10, seed=42)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=30, n_select=5))
            .set_surrogate_manager(
                CompositeSurrogateManager(
                    {"objective": ei_mgr, "feasibility": pof_mgr},
                )
            )
            .set_acquisition(composite_acq)
            .set_termination(Termination(max_fe(60)))
        )
        ctx = optimizer.run()
        cv_arr = ctx.archive.get_array("cv")
        assert np.any(cv_arr <= 1e-6), (
            "Expected at least one feasible solution in archive"
        )


# ---------------------------------------------------------------------------
# 2-D circle-constraint problem (continuous, feasibility ~12.6 %)
# ---------------------------------------------------------------------------

# g(x) = (x0 - 0.5)^2 + (x1 - 0.5)^2 - 0.04 ≤ 0
# Feasibility rate: π·r² = π·0.04 ≈ 12.6 % of [0, 1]²
_CIRCLE_R2 = 0.04


def _circle_obj(x: np.ndarray) -> np.ndarray:
    return np.array([-(x[0] + x[1])])  # minimize → maximise x0 + x1


def _make_circle_problem() -> Problem:
    return Problem(
        func=_circle_obj,
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[0.0, 0.0],
        ub=[1.0, 1.0],
        constraints=[
            InequalityConstraint(
                lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - _CIRCLE_R2
            )
        ],
        handler=StaticToleranceHandler(eps_cv=1e-6),
    )


class TestConstraintSurrogate2D:
    """Functional test: PoF surrogate correctly ranks feasible vs. infeasible."""

    def test_pof_scores_distinguish_feasibility(self) -> None:
        """PoF surrogate assigns higher scores to feasible candidates than infeasible.

        Uses a 2-D circle-constraint problem (feasibility ~12.6 %). After seeding the
        archive with 50 LHS points (~6 expected feasible), the constraint surrogate
        should rank inside-circle candidates above outside-circle candidates.
        """
        problem = _make_circle_problem()
        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=50, n_init_population=10, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=20, n_select=3))
            .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
            .set_acquisition(ExpectedImprovement())
            .set_termination(Termination(max_fe(50)))
        )
        ctx = optimizer.run()
        archive = ctx.archive

        pof_mgr = GlobalSurrogateManager(
            PerObjectiveSurrogate([SklearnGPRSurrogate()]),
            training_set=ConstraintObjectiveSet(),
        )
        pof_acq = ProductOfFeasibility()

        # Points clearly inside circle (g < 0)
        inside = np.array(
            [
                [0.50, 0.50],  # centre: g = -0.04
                [0.55, 0.50],  # g = -0.0375
                [0.50, 0.55],  # g = -0.0375
                [0.45, 0.50],  # g = -0.0375
                [0.50, 0.45],  # g = -0.0375
            ]
        )
        # Points clearly outside circle (g >> 0)
        outside = np.array(
            [
                [0.0, 0.0],  # g = 0.46
                [1.0, 0.0],  # g = 0.46
                [0.0, 1.0],  # g = 0.46
                [1.0, 1.0],  # g = 0.46
                [0.5, 1.0],  # g = 0.21
            ]
        )

        candidates = np.vstack([inside, outside])
        prediction = pof_mgr.predict(candidates, archive, ctx=None)
        scores = pof_acq.evaluate(candidates, prediction, archive, ctx=None).scores
        assert scores is not None

        pof_inside = scores[: len(inside)]
        pof_outside = scores[len(inside) :]

        assert np.mean(pof_inside) > np.mean(pof_outside), (
            f"PoF should be higher for feasible candidates: "
            f"inside={np.mean(pof_inside):.3f}, outside={np.mean(pof_outside):.3f}"
        )


# ---------------------------------------------------------------------------
# Zero-constraint problems
# ---------------------------------------------------------------------------


class TestConstraintSurrogateBackwardCompat:
    """Zero-constraint problems work as before; ConstraintObjectiveSet raises."""

    def test_unconstrained_optimizer_unchanged(self) -> None:
        """Unconstrained sphere problem runs identically with new imports loaded."""
        problem = Problem(
            func=lambda x: np.array([np.sum(x**2)]),
            dim=3,
            n_obj=1,
            direction=np.array([-1.0]),
            lb=[-5.0] * 3,
            ub=[5.0] * 3,
        )
        optimizer = (
            Optimizer(problem)
            .set_initializer(
                LHSInitializer(n_init_archive=10, n_init_population=5, seed=0)
            )
            .set_algorithm(_make_ga())
            .set_strategy(PreSelectionStrategy(n_candidates=10, n_select=2))
            .set_surrogate_manager(GlobalSurrogateManager(SklearnGPRSurrogate()))
            .set_acquisition(ExpectedImprovement())
            .set_termination(Termination(max_fe(20)))
        )
        ctx = optimizer.run()
        assert ctx is not None

    def test_constraint_objective_set_raises_on_zero_constraints(self) -> None:
        """ConstraintObjectiveSet raises ValueError when archive has no g columns."""
        from saealib.population import Archive, PopulationAttribute

        attrs = [
            PopulationAttribute("x", np.float64, (3,)),
            PopulationAttribute("f", np.float64, (1,)),
            PopulationAttribute("g", np.float64, (0,)),
            PopulationAttribute("cv", np.float64, ()),
        ]
        arc = Archive(attrs, init_capacity=10)
        rng = np.random.default_rng(0)
        for _ in range(5):
            arc.add(x=rng.uniform(size=3), f=np.array([1.0]), cv=0.0)

        ts = ConstraintObjectiveSet()
        with pytest.raises(ValueError, match="0 columns"):
            ts.build(arc, None, ctx=None)
