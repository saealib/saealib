"""Tests for Optimizer.from_problem_file(): building an Optimizer from a
Python file that defines a top-level ``problem`` variable (and optionally
``algorithm``/``strategy``/``surrogate_manager``/``termination``/``seed``
variables), picked up via implicit top-level-variable assignment.
"""

import textwrap

import pytest

from saealib.exceptions import ValidationError
from saealib.optimizer import Optimizer
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy

_PROBLEM_ONLY = """
    import numpy as np
    from saealib import Problem

    problem = Problem(
        func=lambda x: np.sum(x**2) * np.ones(1),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.0, -5.0],
        ub=[5.0, 5.0],
    )
"""

_PRESET_YAML = """
    schema_version: 1
    algorithm:
      type: GA
      params:
        crossover: {type: CrossoverBLXAlpha, params: {prob: 0.7, alpha: 0.4}}
        mutation: {type: MutationUniform, params: {prob_var: 0.3}}
        parent_selection: {type: SequentialSelection}
        survivor_selection: {type: TruncationSelection}
    surrogate_manager:
      type: LocalSurrogateManager
      params:
        training_set: {type: KNNObjectiveSet, params: {n_neighbors: 20}}
    strategy:
      type: IndividualBasedStrategy
      params: {evaluation_ratio: 0.2}
    termination:
      type: Termination
      params: [{type: max_fe, params: {value: 60}}]
"""


def _write_problem_file(tmp_path, extra: str = "", body: str = _PROBLEM_ONLY) -> str:
    path = tmp_path / "problem_def.py"
    path.write_text(textwrap.dedent(body) + textwrap.dedent(extra))
    return str(path)


def _write_preset_file(tmp_path) -> str:
    path = tmp_path / "preset.yaml"
    path.write_text(textwrap.dedent(_PRESET_YAML))
    return str(path)


class TestProblemOnlyNoPreset:
    def test_run_completes_with_bundled_fallback(self, tmp_path):
        path = _write_problem_file(tmp_path)

        opt = Optimizer.from_problem_file(path)
        ctx = opt.run()

        assert ctx.fe > 0


class TestProblemOnlyWithPreset:
    def test_run_uses_preset_components(self, tmp_path):
        problem_path = _write_problem_file(tmp_path)
        preset_path = _write_preset_file(tmp_path)

        opt = Optimizer.from_problem_file(problem_path, preset=preset_path)
        ctx = opt.run()

        assert isinstance(opt.strategy, IndividualBasedStrategy)
        assert opt.strategy.evaluation_ratio == 0.2
        assert ctx.fe >= 60


class TestPythonRedefinitionWinsOverPreset:
    def test_strategy_from_file_beats_preset_strategy(self, tmp_path):
        preset_path = _write_preset_file(tmp_path)
        problem_path = _write_problem_file(
            tmp_path,
            body="""
                import numpy as np
                from saealib import Problem
                from saealib.strategies.gb import GenerationBasedStrategy

                problem = Problem(
                    func=lambda x: np.sum(x**2) * np.ones(1),
                    dim=2,
                    n_obj=1,
                    direction=np.array([-1.0]),
                    lb=[-5.0, -5.0],
                    ub=[5.0, 5.0],
                )
                strategy = GenerationBasedStrategy(gen_ctrl=7)
            """,
        )

        opt = Optimizer.from_problem_file(problem_path, preset=preset_path)
        opt.resolve_defaults()

        assert isinstance(opt.strategy, GenerationBasedStrategy)
        assert opt.strategy.gen_ctrl == 7


class TestSeedVariable:
    def test_seed_is_forwarded(self, tmp_path):
        path = _write_problem_file(tmp_path, extra="\nseed = 42\n")

        opt = Optimizer.from_problem_file(path)

        assert opt.seed == 42


class TestMissingProblemVariable:
    def test_no_problem_variable_raises(self, tmp_path):
        path = tmp_path / "problem_def.py"
        path.write_text("x = 1\n")

        with pytest.raises(ValidationError):
            Optimizer.from_problem_file(str(path))


class TestProblemVariableWrongType:
    def test_non_problem_instance_raises(self, tmp_path):
        path = tmp_path / "problem_def.py"
        path.write_text("problem = 'not a problem'\n")

        with pytest.raises(ValidationError):
            Optimizer.from_problem_file(str(path))


class TestAlgorithmVariableWrongType:
    def test_non_algorithm_instance_raises(self, tmp_path):
        path = _write_problem_file(tmp_path, extra="\nalgorithm = 'GA'\n")

        with pytest.raises(ValidationError, match="algorithm"):
            Optimizer.from_problem_file(path)


class TestStrategyVariableWrongType:
    def test_non_strategy_instance_raises(self, tmp_path):
        path = _write_problem_file(tmp_path, extra="\nstrategy = 123\n")

        with pytest.raises(ValidationError, match="strategy"):
            Optimizer.from_problem_file(path)


class TestSurrogateManagerVariableWrongType:
    def test_non_surrogate_manager_instance_raises(self, tmp_path):
        path = _write_problem_file(tmp_path, extra="\nsurrogate_manager = object()\n")

        with pytest.raises(ValidationError, match="surrogate_manager"):
            Optimizer.from_problem_file(path)


class TestTerminationVariableWrongType:
    def test_non_termination_instance_raises(self, tmp_path):
        path = _write_problem_file(
            tmp_path, extra="\ntermination = 'not a termination'\n"
        )

        with pytest.raises(ValidationError, match="termination"):
            Optimizer.from_problem_file(path)


class TestSeedVariableWrongType:
    def test_non_int_seed_raises(self, tmp_path):
        path = _write_problem_file(tmp_path, extra="\nseed = '42'\n")

        with pytest.raises(ValidationError, match="seed"):
            Optimizer.from_problem_file(path)


class TestValidComponentTypesStillWork:
    def test_correctly_typed_components_are_accepted(self, tmp_path):
        problem_path = _write_problem_file(
            tmp_path,
            body="""
                import numpy as np
                from saealib import Problem
                from saealib.strategies.gb import GenerationBasedStrategy
                from saealib.termination import Termination, max_fe

                problem = Problem(
                    func=lambda x: np.sum(x**2) * np.ones(1),
                    dim=2,
                    n_obj=1,
                    direction=np.array([-1.0]),
                    lb=[-5.0, -5.0],
                    ub=[5.0, 5.0],
                )
                strategy = GenerationBasedStrategy(gen_ctrl=3)
                termination = Termination(max_fe(30))
                seed = 7
            """,
        )

        opt = Optimizer.from_problem_file(problem_path)

        assert isinstance(opt.strategy, GenerationBasedStrategy)
        assert opt.seed == 7


class TestNonExistentPath:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Optimizer.from_problem_file(str(tmp_path / "does_not_exist.py"))


class TestEndToEndRun:
    def test_run_populates_archive_and_respects_termination(self, tmp_path):
        problem_path = _write_problem_file(tmp_path)
        preset_path = _write_preset_file(tmp_path)

        opt = Optimizer.from_problem_file(problem_path, preset=preset_path)
        ctx = opt.run()

        assert ctx.archive is not None
        assert len(ctx.archive) > 0
        assert ctx.fe >= 60
