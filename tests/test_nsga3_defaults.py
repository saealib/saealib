"""Tests for NSGA-III composition-aware default resolution.

Verifies that:
1. NSGA-III population default scales with reference points.
2. No isinstance NSGA3Comparator checks in resolver.
3. NSGA3Comparator doesn't know about concrete initializers.
4. Explicit initializer unchanged.
5. Generic population fallback is backward compatible.
"""

import math

import numpy as np

from saealib.comparators.comparators import NSGA3Comparator
from saealib.defaults.builtin import BUILTIN_DEFAULT_PROVIDER
from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import (
    INITIAL_ARCHIVE_SIZE,
    POPULATION_SIZE,
)
from saealib.defaults.resolver import DEFAULT_RESOLVER, DefaultHintProvider
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem


def _problem_with_nsga3(
    n_obj: int = 3, dim: int = 5, n_ref: int = 10
) -> tuple[Problem, NSGA3Comparator]:
    """Create a problem with NSGA-III comparator and reference points."""
    # Create reference points with exact count using manual array
    # For testing, we create simple reference points on the unit simplex
    reference_points = np.eye(n_obj, dtype=float)
    if n_ref > n_obj:
        # Add more points by interpolating
        extra = n_ref - n_obj
        for i in range(extra):
            # Create points that sum to 1
            point = np.ones(n_obj, dtype=float) / n_obj
            reference_points = np.vstack([reference_points, point])
    reference_points = reference_points[:n_ref]
    # Normalize to unit simplex
    reference_points = reference_points / reference_points.sum(axis=1, keepdims=True)

    comparator = NSGA3Comparator(reference_points=reference_points)
    problem = Problem(
        func=lambda x: np.sum(x) * np.ones(n_obj),
        dim=dim,
        n_obj=n_obj,
        direction=np.array([-1.0] * n_obj),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
        comparator=comparator,
    )
    return problem, comparator


def _problem_single_obj(dim: int = 4) -> Problem:
    """Create a single-objective problem (no NSGA-III)."""
    return Problem(
        func=lambda x: np.sum(x),
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0] * dim,
        ub=[5.0] * dim,
    )


class TestNSGA3PopulationScaling:
    """Test that NSGA-III population size scales with reference points."""

    def test_population_scales_with_reference_points(self):
        """Population size should be 4 * ceil(n_ref / 4)."""
        problem, _ = _problem_with_nsga3(n_obj=3, n_ref=10)
        opt = Optimizer(problem)
        opt.resolve_defaults()

        assert isinstance(opt.initializer, LHSInitializer)
        # 4 * ceil(10 / 4) = 4 * 3 = 12
        assert opt.initializer.n_init_population == 12

    def test_population_scales_with_different_ref_counts(self):
        """Test various reference point counts."""
        for n_ref in [5, 9, 10, 15, 20, 21]:
            problem, _ = _problem_with_nsga3(n_obj=3, n_ref=n_ref)
            opt = Optimizer(problem)
            opt.resolve_defaults()

            assert isinstance(opt.initializer, LHSInitializer)
            expected = 4 * math.ceil(n_ref / 4)
            assert opt.initializer.n_init_population == expected, (
                f"n_ref={n_ref}: expected {expected}, "
                f"got {opt.initializer.n_init_population}"
            )

    def test_archive_scales_with_reference_points(self):
        """Archive size should also scale with reference points."""
        problem, _ = _problem_with_nsga3(n_obj=3, n_ref=10)
        opt = Optimizer(problem)
        opt.resolve_defaults()

        assert isinstance(opt.initializer, LHSInitializer)
        # Archive should be at least as large as population
        assert opt.initializer.n_init_archive >= opt.initializer.n_init_population

    def test_single_obj_uses_generic_fallback(self):
        """Single-objective problems should use generic 4*dim fallback."""
        problem = _problem_single_obj(dim=4)
        opt = Optimizer(problem)
        opt.resolve_defaults()

        assert isinstance(opt.initializer, LHSInitializer)
        assert opt.initializer.n_init_population == 4 * 4  # 16
        assert opt.initializer.n_init_archive == 5 * 4  # 20


class TestNoIsinstanceChecks:
    """Verify no isinstance(..., NSGA3Comparator) in resolver."""

    def test_resolver_has_no_nsga3_imports(self):
        """Resolver should not import or check for NSGA3Comparator."""
        import inspect

        from saealib.defaults import resolver

        source = inspect.getsource(resolver)
        assert "NSGA3Comparator" not in source
        assert "NSGA3" not in source

    def test_builtin_provider_has_no_nsga3_imports(self):
        """BuiltinDefaultProvider should not import or check for NSGA3Comparator."""
        import inspect

        from saealib.defaults import builtin

        source = inspect.getsource(builtin)
        assert "NSGA3Comparator" not in source
        assert "NSGA3" not in source


class TestNSGA3ComparatorProtocol:
    """Verify NSGA3Comparator implements DefaultHintProvider protocol."""

    def test_nsga3_comparator_is_hint_provider(self):
        """NSGA3Comparator should be an instance of DefaultHintProvider."""
        from saealib.utils.weight_vectors import uniform_weight_vectors

        ref_points = uniform_weight_vectors(3, 10)
        comparator = NSGA3Comparator(reference_points=ref_points)
        assert isinstance(comparator, DefaultHintProvider)

    def test_nsga3_comparator_has_default_hints_method(self):
        """NSGA3Comparator should have default_hints method."""
        from saealib.utils.weight_vectors import uniform_weight_vectors

        ref_points = uniform_weight_vectors(3, 10)
        comparator = NSGA3Comparator(reference_points=ref_points)
        assert hasattr(comparator, "default_hints")
        assert callable(comparator.default_hints)

    def test_nsga3_comparator_hints_contain_population_size(self):
        """default_hints should return a hint for POPULATION_SIZE."""
        from saealib.utils.weight_vectors import uniform_weight_vectors

        ref_points = uniform_weight_vectors(3, 10)
        comparator = NSGA3Comparator(reference_points=ref_points)
        ctx = DefaultContext(problem=_problem_single_obj())
        hints = comparator.default_hints(ctx)

        assert len(hints) > 0
        assert any(h.key == POPULATION_SIZE for h in hints)


class TestExplicitInitializerUnchanged:
    """Verify explicit initializer is not overwritten."""

    def test_explicit_initializer_preserved(self):
        """Explicitly set initializer should not be overwritten."""
        problem, _ = _problem_with_nsga3(n_obj=3, n_ref=10)
        opt = Optimizer(problem)

        # Set explicit initializer
        explicit_init = LHSInitializer(
            n_init_archive=100, n_init_population=50, seed=42
        )
        opt.set_initializer(explicit_init)
        opt.resolve_defaults()

        assert opt.initializer is explicit_init
        assert opt.initializer.n_init_population == 50
        assert opt.initializer.n_init_archive == 100


class TestGenericPopulationFallback:
    """Verify generic population fallback is backward compatible."""

    def test_generic_fallback_returns_dim_based_values(self):
        """BuiltinDefaultProvider should return 4*dim for population."""
        problem = _problem_single_obj(dim=5)
        ctx = DefaultContext(problem=problem)
        hints = BUILTIN_DEFAULT_PROVIDER.default_hints(ctx)

        pop_hint = next(h for h in hints if h.key == POPULATION_SIZE)
        assert pop_hint.value == 4 * 5

    def test_generic_fallback_returns_dim_based_archive(self):
        """BuiltinDefaultProvider should return 5*dim for archive."""
        problem = _problem_single_obj(dim=5)
        ctx = DefaultContext(problem=problem)
        hints = BUILTIN_DEFAULT_PROVIDER.default_hints(ctx)

        archive_hint = next(h for h in hints if h.key == INITIAL_ARCHIVE_SIZE)
        assert archive_hint.value == 5 * 5


class TestResolverIntegration:
    """Test the resolver with multiple providers."""

    def test_nsga3_hint_overrides_fallback(self):
        """NSGA-III RECOMMENDED hint should override FALLBACK."""

        problem, comparator = _problem_with_nsga3(n_obj=3, n_ref=10)
        ctx = DefaultContext(problem=problem)

        providers = [BUILTIN_DEFAULT_PROVIDER, comparator]
        resolution = DEFAULT_RESOLVER.resolve(ctx, providers)

        # NSGA-III hint (RECOMMENDED) should win over builtin (FALLBACK)
        assert resolution.get(POPULATION_SIZE) == 12  # 4 * ceil(10/4)

    def test_explicit_overrides_all_hints(self):
        """Explicit values should override all hints."""

        problem, comparator = _problem_with_nsga3(n_obj=3, n_ref=10)
        ctx = DefaultContext(problem=problem)

        providers = [BUILTIN_DEFAULT_PROVIDER, comparator]
        explicit = {POPULATION_SIZE: 100}
        resolution = DEFAULT_RESOLVER.resolve(ctx, providers, explicit=explicit)

        assert resolution.get(POPULATION_SIZE) == 100

    def test_deterministic_resolution(self):
        """Resolution should be deterministic."""

        problem, comparator = _problem_with_nsga3(n_obj=3, n_ref=10)
        ctx = DefaultContext(problem=problem)

        providers = [BUILTIN_DEFAULT_PROVIDER, comparator]

        # Run resolution multiple times
        results = []
        for _ in range(5):
            resolution = DEFAULT_RESOLVER.resolve(ctx, providers)
            results.append(resolution.get(POPULATION_SIZE))

        # All results should be the same
        assert len(set(results)) == 1
