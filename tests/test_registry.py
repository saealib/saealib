"""Unit tests for saealib.registry: register/get/build/to_spec resolution."""

from typing import Any, cast

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.pipeline import (
    Branch,
    ComponentRef,
    Condition,
    Loop,
    Pipeline,
    Repeat,
    Stage,
)
from saealib.policies.evaluation import RatioEvaluation
from saealib.registry import (
    _inject_params,
    _strip_params,
    build,
    get,
    register,
    to_spec,
)
from saealib.stages import AskStage, CountGenerationStage, EvaluationPlanStage


class Engine:
    def __init__(self, power: int = 100):
        self.power = power


class Car:
    def __init__(self, engine=None, wheels: int = 4):
        self.engine = engine
        self.wheels = wheels


class Convoy:
    """Constructor takes *args — exercises the vararg spec path."""

    def __init__(self, *vehicles):
        self.vehicles = vehicles


class OptionalOnly:
    def __init__(self, note: str = "default"):
        self.note = note


class Opaque:
    """Opts out of generic reflection via ``_registry_spec``."""

    def __init__(self, spec=None):
        self._registry_spec = spec


class Powered:
    """Takes a callable param — exercises the callable-ref spec path."""

    def __init__(self, kernel=None):
        self.kernel = kernel


class KwargsPlugin:
    """Constructor takes **kwargs, stored under the same-named attribute."""

    def __init__(self, **settings):
        self.settings = settings


class KwargsWithNamed:
    """Constructor takes a named param plus **kwargs."""

    def __init__(self, name="base", **settings):
        self.name = name
        self.settings = settings


class ModuleLevelFallback:
    def __init__(self, note: str = "default"):
        self.note = note


class ComponentWithoutSerializableState:
    def __init__(self, required):
        self._required = required


class FreshEngine:
    def __init__(self, power: int):
        self.power = power


class FreshBox:
    def __init__(self, engine: FreshEngine):
        self.power = engine.power

    @property
    def engine(self):
        return FreshEngine(self.power)


class InheritedPlanStage(EvaluationPlanStage):
    pass


class PrivateAmountStage(Stage):
    def __init__(self, amount: int = 1):
        super().__init__()
        self._amount = amount

    def contract(self):
        from saealib.core.contracts import ComponentContract

        return ComponentContract()

    def execute(self, state):
        return state


class VariadicMetadataStage(Stage):
    def __init__(self, *values):
        super().__init__()
        self.values = values

    def contract(self):
        from saealib.core.contracts import ComponentContract

        return ComponentContract()

    def execute(self, state):
        return state


class UnreadableOptionalStage(Stage):
    def __init__(self, amount: int = 1):
        super().__init__()

    def contract(self):
        from saealib.core.contracts import ComponentContract

        return ComponentContract()

    def execute(self, state):
        return state


class AlwaysCondition:
    def contract(self):
        from saealib.core.contracts import StateContract

        return StateContract()

    def evaluate(self, view):
        return True


class MarkedPipeline(Pipeline):
    def __init__(
        self,
        stages=None,
        name="",
        label="",
        notation="",
        *,
        marker="default",
    ):
        self.marker = marker
        super().__init__(stages, name=name, label=label, notation=notation)


class MarkedRepeat(Repeat):
    def __init__(self, body, count, *, marker="default", name=""):
        self.marker = marker
        super().__init__(body, count, name=name)


class MarkedLoop(Loop):
    def __init__(self, body, *, until, marker="default", name=""):
        self.marker = marker
        super().__init__(body, until=until, name=name)


class MarkedBranch(Branch):
    def __init__(self, condition, *, then, else_=None, marker="default", name=""):
        self.marker = marker
        super().__init__(condition, then=then, else_=else_, name=name)


class TestRegisterAndGet:
    def test_register_default_name(self):
        register()(Engine)
        assert get("Engine") is Engine

    def test_register_explicit_name(self):
        register("MyEngine")(Engine)
        assert get("MyEngine") is Engine

    def test_get_unknown_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("NoSuchThing")

    def test_get_dotted_path_resolves_without_registration(self):
        from collections import OrderedDict

        resolved = get("collections.OrderedDict")
        assert resolved is OrderedDict

    def test_get_dotted_path_bad_module_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("no.such.module.Thing")

    def test_get_dotted_path_bad_attr_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("collections.NoSuchClass")


class TestBuild:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)

    def test_build_passthrough_live_instance(self):
        instance = Engine(power=200)
        assert build(instance) is instance

    def test_build_from_bare_string(self):
        obj = build("Engine")
        assert isinstance(obj, Engine)
        assert obj.power == 100

    def test_build_from_spec_with_params(self):
        obj = build({"type": "Engine", "params": {"power": 250}})
        assert isinstance(obj, Engine)
        assert obj.power == 250

    def test_build_recurses_into_nested_spec(self):
        obj = build(
            {
                "type": "Car",
                "params": {"engine": {"type": "Engine", "params": {"power": 300}}},
            }
        )
        assert isinstance(obj, Car)
        assert isinstance(obj.engine, Engine)
        assert obj.engine.power == 300

    def test_build_recurses_through_top_level_lists_and_mappings(self):
        values = build(to_spec([RatioEvaluation(0.25)]))
        mapping = build(to_spec({"planner": RatioEvaluation(0.5)}))
        plain_mapping = build(
            {"planner": {"type": "RatioEvaluation", "params": {"ratio": 0.75}}}
        )

        assert isinstance(values[0], RatioEvaluation)
        assert values[0].ratio == 0.25
        assert isinstance(mapping["planner"], RatioEvaluation)
        assert mapping["planner"].ratio == 0.5
        assert isinstance(plain_mapping, dict)
        assert isinstance(plain_mapping["planner"], RatioEvaluation)
        assert isinstance(build("Engine"), Engine)

    def test_build_leaves_non_spec_params_untouched(self):
        obj = build({"type": "Car", "params": {"wheels": 3}})
        assert obj.wheels == 3
        assert obj.engine is None

    def test_build_unknown_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            build({"type": "NoSuchThing", "params": {}})

    def test_build_unknown_param_raises_validation_error(self):
        with pytest.raises(ValidationError):
            build({"type": "Engine", "params": {"not_a_param": 1}})

    def test_build_missing_required_param_raises_validation_error(self):
        class Needy:
            def __init__(self, required_arg):
                self.required_arg = required_arg

        register("Needy")(Needy)
        with pytest.raises(ValidationError):
            build({"type": "Needy", "params": {}})


class TestBuildVarargs:
    def setup_method(self):
        register("Engine")(Engine)
        register("Convoy")(Convoy)

    def test_list_params_call_positionally(self):
        obj = build(
            {
                "type": "Convoy",
                "params": [
                    {"type": "Engine", "params": {"power": 1}},
                    {"type": "Engine", "params": {"power": 2}},
                ],
            }
        )
        assert isinstance(obj, Convoy)
        assert [v.power for v in obj.vehicles] == [1, 2]

    def test_empty_list_params(self):
        obj = build({"type": "Convoy", "params": []})
        assert obj.vehicles == ()


class TestBuildCallableRef:
    def setup_method(self):
        register("Powered")(Powered)

    def test_build_resolves_callable_ref_param(self):
        import math

        obj = build(
            {"type": "Powered", "params": {"kernel": {"callable": "math.isnan"}}}
        )
        assert obj.kernel is math.isnan

    def test_build_top_level_callable_ref(self):
        import math

        assert build({"callable": "math.isnan"}) is math.isnan


class TestToSpec:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)
        register("Convoy")(Convoy)
        register("OptionalOnly")(OptionalOnly)
        register("Opaque")(Opaque)
        register("Powered")(Powered)
        register("KwargsPlugin")(KwargsPlugin)
        register("KwargsWithNamed")(KwargsWithNamed)

    @pytest.mark.parametrize(
        "value", [None, True, 1, 1.5, "s", [1, 2], (1, 2), {"a": 1}]
    )
    def test_primitives_and_containers_pass_through_structurally(self, value):
        assert to_spec(value) == (list(value) if isinstance(value, tuple) else value)

    def test_numpy_array_becomes_list(self):
        assert to_spec(np.array([1.0, 2.0])) == [1.0, 2.0]

    def test_function_becomes_callable_ref(self):
        import math

        assert to_spec(math.isnan) == {"callable": "math.isnan"}

    def test_registered_instance_reflects_constructor_params(self):
        obj = Engine(power=42)
        assert to_spec(obj) == {"type": "Engine", "params": {"power": 42}}

    def test_nested_registered_instance_recurses(self):
        car = Car(engine=Engine(power=7), wheels=3)
        assert to_spec(car) == {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 7}},
                "wheels": 3,
            },
        }

    def test_unregistered_class_falls_back_to_dotted_path_type(self):
        spec = to_spec(ModuleLevelFallback(note="x"))
        assert spec == {
            "type": f"{__name__}.{ModuleLevelFallback.__qualname__}",
            "params": {"note": "x"},
        }

    def test_local_reflected_type_is_rejected_with_location(self):
        class NotRegistered:
            def __init__(self, note: str = "default"):
                self.note = note

        with pytest.raises(ValidationError, match="value"):
            to_spec(NotRegistered(note="x"))

    def test_local_component_type_requires_importable_name(self):
        class LocalComponent:
            def contract(self):
                return None

        with pytest.raises(ValidationError, match=r"value.*not importable"):
            to_spec(LocalComponent())

    def test_missing_optional_attr_is_omitted(self):
        obj = OptionalOnly()
        del obj.note
        assert to_spec(obj) == {"type": "OptionalOnly", "params": {}}

    def test_missing_required_attr_raises_validation_error(self):
        class Needy:
            def __init__(self, required_arg):
                pass  # does not store self.required_arg

        register("Needy")(Needy)
        with pytest.raises(ValidationError):
            to_spec(Needy(1))

    def test_var_positional_becomes_list_params(self):
        convoy = Convoy(Engine(power=1), Engine(power=2))
        assert to_spec(convoy) == {
            "type": "Convoy",
            "params": [
                {"type": "Engine", "params": {"power": 1}},
                {"type": "Engine", "params": {"power": 2}},
            ],
        }

    def test_var_positional_missing_attr_raises_validation_error(self):
        class BadConvoy:
            def __init__(self, *vehicles):
                pass  # does not store self.vehicles

        register("BadConvoy")(BadConvoy)
        with pytest.raises(ValidationError):
            to_spec(BadConvoy())

    def test_var_positional_with_keyword_settings_raises_validation_error(self):
        class Both:
            def __init__(self, *args, **settings):
                self.args = args
                self.settings = settings

        register("BothVarargsKwargs")(Both)
        with pytest.raises(ValidationError, match=r"BothVarargsKwargs.*settings"):
            to_spec(Both(1, 2, alpha=3))

    def test_var_positional_without_keyword_settings_round_trips(self):
        class PositionalOnly:
            def __init__(self, *args, **settings):
                self.args = args
                self.settings = settings

        register("PositionalOnly")(PositionalOnly)
        obj = PositionalOnly(1, 2)
        assert to_spec(obj) == {"type": "PositionalOnly", "params": [1, 2]}
        rebuilt = build(to_spec(obj))
        assert rebuilt.args == (1, 2)
        assert rebuilt.settings == {}

    def test_var_positional_with_empty_settings_attribute_round_trips(self):
        class EmptySettings:
            def __init__(self, *args, **settings):
                self.args = args
                self.settings = settings

        register("EmptySettings")(EmptySettings)
        obj = EmptySettings(1, 2)
        obj.settings = {}
        rebuilt = build(to_spec(obj))
        assert rebuilt.args == (1, 2)
        assert rebuilt.settings == {}

    def test_var_positional_without_settings_attribute_round_trips(self):
        class UnstoredSettings:
            def __init__(self, *args, **settings):
                self.args = args

        register("UnstoredSettings")(UnstoredSettings)
        rebuilt = build(to_spec(UnstoredSettings(1, 2)))
        assert rebuilt.args == (1, 2)

    def test_registry_spec_hook_used_instead_of_reflection(self):
        obj = Opaque(spec={"type": "whatever", "params": {"x": 1}})
        assert to_spec(obj) == {"type": "whatever", "params": {"x": 1}}

    def test_registry_spec_hook_none_raises_validation_error(self):
        obj = Opaque(spec=None)
        from saealib.stages import EvaluationPlanStage

        with pytest.raises(ValidationError, match=r"value.*planner"):
            to_spec(Pipeline([EvaluationPlanStage(cast(Any, obj))]))

    def test_round_trip_build_to_spec_build(self):
        original = Car(engine=Engine(power=9), wheels=6)
        rebuilt = build(to_spec(original))
        assert isinstance(rebuilt, Car)
        assert rebuilt.wheels == 6
        assert rebuilt.engine.power == 9

    def test_round_trip_varargs(self):
        original = Convoy(Engine(power=1), Engine(power=2))
        rebuilt = build(to_spec(original))
        assert isinstance(rebuilt, Convoy)
        assert [v.power for v in rebuilt.vehicles] == [1, 2]

    def test_round_trip_function_param(self):
        import math

        original = Powered(kernel=math.isnan)
        spec = to_spec(original)
        assert spec == {
            "type": "Powered",
            "params": {"kernel": {"callable": "math.isnan"}},
        }
        rebuilt = build(spec)
        assert rebuilt.kernel is math.isnan

    def test_build_recurses_through_lists_and_plain_mappings(self):
        spec = {
            "type": "Car",
            "params": {
                "engine": [
                    {"type": "Engine", "params": {"power": 3}},
                    {
                        "metadata": {
                            "engine": {"type": "Engine", "params": {"power": 4}}
                        }
                    },
                ]
            },
        }
        rebuilt = build(spec)
        assert isinstance(rebuilt.engine[0], Engine)
        assert isinstance(rebuilt.engine[1]["metadata"]["engine"], Engine)
        assert rebuilt.engine[1]["metadata"]["engine"].power == 4

    def test_repeated_container_metadata_is_serializable(self):
        metadata = [1, {"label": "same"}]
        assert to_spec({"left": metadata, "right": metadata}) == {
            "left": [1, {"label": "same"}],
            "right": [1, {"label": "same"}],
        }

    def test_pipeline_spec_uses_stages_and_round_trips_nested_entries(self):
        from saealib.stages import CountGenerationStage

        original = Pipeline([Pipeline([CountGenerationStage()])])
        spec = to_spec(original)
        assert set(spec["params"]) == {"stages", "name", "label", "notation"}
        rebuilt = build(spec)
        assert isinstance(rebuilt, Pipeline)
        assert isinstance(rebuilt.stages[0].stages[0], CountGenerationStage)

    def test_surrogate_loop_rebuilds_its_inner_stages_from_components(self):
        from saealib.stages import SurrogateOnlyLoopStage

        original = SurrogateOnlyLoopStage(
            cast(Any, Engine(power=1)),
            cast(Any, Engine(power=2)),
            2,
            acquisition=cast(Any, Engine(power=3)),
        )
        spec = to_spec(original)
        assert "stages" not in spec["params"]
        rebuilt = build(spec)
        assert len(rebuilt.stages) == 6
        assert to_spec(rebuilt) == spec

    def test_component_reference_can_repeat_in_a_declaration(self):
        from saealib.stages import AskStage

        reference = ComponentRef("proposer", "algorithm")
        spec = to_spec(Pipeline([AskStage(reference), AskStage(reference)]))
        assert spec["params"]["stages"][0] == spec["params"]["stages"][1]

    def test_equal_distinct_components_remain_serializable(self):
        from saealib.stages import AskStage

        spec = to_spec(
            Pipeline(
                [
                    AskStage(cast(Any, Engine(power=7))),
                    AskStage(cast(Any, Engine(power=7))),
                ]
            )
        )
        assert spec["params"]["stages"][0] == spec["params"]["stages"][1]

    def test_shared_component_instance_names_both_locations(self):
        from saealib.stages import AskStage

        shared = Engine(power=7)
        with pytest.raises(
            ValidationError, match=r"stages.*1.*shares.*stages.*0.*ComponentRef"
        ):
            to_spec(
                Pipeline([AskStage(cast(Any, shared)), AskStage(cast(Any, shared))])
            )

    def test_reference_cycle_is_rejected_with_location(self):
        values = []
        values.append(values)
        with pytest.raises(ValidationError, match=r"value.*cycle"):
            to_spec(values)

    def test_temporary_property_values_do_not_look_shared(self):
        register("FreshEngine")(FreshEngine)
        register("FreshBox")(FreshBox)

        boxes = [FreshBox(FreshEngine(power)) for power in range(20)]
        spec = to_spec(boxes)

        assert len(spec) == 20
        assert [item["params"]["engine"]["params"]["power"] for item in spec] == list(
            range(20)
        )

    def test_stage_subclasses_keep_mapped_and_private_constructor_values(self):
        register("InheritedPlanStage")(InheritedPlanStage)
        register("PrivateAmountStage")(PrivateAmountStage)

        plan = InheritedPlanStage(RatioEvaluation(0.25))
        private = PrivateAmountStage(7)
        rebuilt_plan = build(to_spec(plan))
        rebuilt_private = build(to_spec(private))

        assert type(rebuilt_plan) is InheritedPlanStage
        assert isinstance(rebuilt_plan._planner, RatioEvaluation)
        assert rebuilt_plan._planner.ratio == 0.25
        assert type(rebuilt_private) is PrivateAmountStage
        assert rebuilt_private._amount == 7

    def test_unreadable_optional_stage_argument_reports_location(self):
        register("UnreadableOptionalStage")(UnreadableOptionalStage)

        with pytest.raises(
            ValidationError, match=r"value\['stages'\]\[0\]\['amount'\]"
        ):
            to_spec(Pipeline([UnreadableOptionalStage(7)]))

    def test_stage_identity_metadata_round_trips_outside_constructor_params(self):
        default_spec = to_spec(CountGenerationStage())
        stage = CountGenerationStage()
        stage.name = "custom_generation"
        stage.label = "Custom generation"
        stage.notation = "g + 1"

        stage_spec = to_spec(stage)
        pipeline_spec = to_spec(Pipeline([stage]))
        restored = build(stage_spec)
        restored_in_pipeline = build(pipeline_spec).stages[0]
        custom = PrivateAmountStage(9)
        custom.name = "private_amount"
        custom.label = "Private amount"
        custom_spec = to_spec(custom)
        restored_custom = build(custom_spec)

        assert "_registry_stage_metadata" not in default_spec
        assert stage_spec["params"]["name"] == "custom_generation"
        assert stage_spec["params"]["label"] == "Custom generation"
        assert stage_spec["params"]["notation"] == "g + 1"
        assert "_registry_stage_metadata" not in stage_spec
        assert (restored.name, restored.label, restored.notation) == (
            "custom_generation",
            "Custom generation",
            "g + 1",
        )
        assert (restored_in_pipeline.name, restored_in_pipeline.label) == (
            "custom_generation",
            "Custom generation",
        )
        assert not {"name", "label", "notation"} & custom_spec["params"].keys()
        assert (restored_custom.name, restored_custom.label) == (
            "private_amount",
            "Private amount",
        )

    def test_constructor_identity_arguments_round_trip(self):
        class RequiredName(Stage):
            def __init__(self, name):
                super().__init__(name=name)
                self.saved_name = name

            def execute(self, state):
                return state

        class OptionalLabel(Stage):
            def __init__(self, label="default"):
                super().__init__(label=label)
                self.saved_label = label

            def execute(self, state):
                return state

        register("RequiredNameRoundTrip")(RequiredName)
        register("OptionalLabelRoundTrip")(OptionalLabel)
        required = RequiredName("custom")
        optional = OptionalLabel("custom")

        required_spec = to_spec(required)
        optional_spec = to_spec(optional)
        rebuilt_required = build(required_spec)
        rebuilt_optional = build(optional_spec)

        assert required_spec["params"]["name"] == "custom"
        assert "_registry_stage_metadata" not in required_spec
        assert rebuilt_required.saved_name == "custom"
        assert optional_spec["params"]["label"] == "custom"
        assert rebuilt_optional.saved_label == "custom"

    def test_overridden_builtin_stage_constructors_use_their_own_signature(self):
        from saealib.pipeline import ComponentRef
        from saealib.policies.evaluation import RatioEvaluation
        from saealib.stages import TopKSelectionStage

        class ScaledSelection(TopKSelectionStage):
            def __init__(self, k=2):
                self.k = k
                super().__init__(k * 2)

        class FixedAsk(AskStage):
            def __init__(self, amount=4):
                self.amount = amount
                super().__init__(ComponentRef("proposer"), n_offspring=amount)

        class AskWithExtra(AskStage):
            def __init__(self, algorithm, extra=1):
                self.extra = extra
                super().__init__(algorithm)

        class PlanFixedRatio(EvaluationPlanStage):
            def __init__(self, ratio=0.5):
                super().__init__(RatioEvaluation(ratio))
                self.ratio = ratio

        for name, stage_type in (
            ("ScaledSelectionRoundTrip", ScaledSelection),
            ("FixedAskRoundTrip", FixedAsk),
            ("AskWithExtraRoundTrip", AskWithExtra),
            ("PlanFixedRatioRoundTrip", PlanFixedRatio),
        ):
            register(name)(stage_type)

        stages = [
            ScaledSelection(3),
            FixedAsk(7),
            PlanFixedRatio(0.25),
        ]
        restored = [build(to_spec(stage)) for stage in stages]

        assert restored[0].k == 3 and restored[0]._k == 6
        assert restored[1].amount == 7 and restored[1]._n_offspring == 7
        assert restored[2].ratio == 0.25
        assert restored[2]._planner.ratio == 0.25

        with pytest.raises(
            ValidationError,
            match=r"value\['algorithm'\].*constructor value is unavailable",
        ):
            to_spec(AskWithExtra(Engine(power=9), extra=3))
        with pytest.raises(
            ValidationError,
            match=r"value\['algorithm'\].*constructor value is unavailable",
        ):
            to_spec(AskWithExtra(ComponentRef("proposer"), extra=3))

    def test_builtin_parent_private_values_are_unavailable_for_overridden_init(self):
        from saealib.pipeline import ComponentRef
        from saealib.stages import TopKSelectionStage

        class Doubled(TopKSelectionStage):
            def __init__(self, k=2):
                super().__init__(k * 2)

        class DoubleAsk(AskStage):
            def __init__(self, n_offspring=3):
                super().__init__(ComponentRef("proposer"), n_offspring=n_offspring * 2)

        class HalfPlan(EvaluationPlanStage):
            def __init__(self, n_eval=4):
                super().__init__(n_eval=n_eval // 2)

        register("DoubledPrivateParentRoundTrip")(Doubled)
        register("DoubleAskPrivateParentRoundTrip")(DoubleAsk)
        register("HalfPlanPrivateParentRoundTrip")(HalfPlan)

        for original, field in (
            (Doubled(5), "k"),
            (DoubleAsk(7), "n_offspring"),
            (HalfPlan(10), "n_eval"),
        ):
            with pytest.raises(
                ValidationError,
                match=rf"value\['{field}'\].*constructor value is unavailable",
            ):
                to_spec(original)

    def test_fixed_pipeline_and_control_subclasses_round_trip(self):
        class FixedPipeline(Pipeline):
            def __init__(self, marker="kept"):
                self.marker = marker
                super().__init__([])

        class FixedRepeat(Repeat):
            def __init__(self, count=2):
                super().__init__(Pipeline([]), count)

        class Thrice(Repeat):
            def __init__(self):
                super().__init__(Pipeline([CountGenerationStage()]), 3)

        register("FixedPipelineRoundTrip")(FixedPipeline)
        register("FixedRepeatRoundTrip")(FixedRepeat)
        register("ThriceRoundTrip")(Thrice)

        original = [FixedPipeline("custom"), FixedRepeat(3), Thrice()]
        specs = [to_spec(value) for value in original]
        rebuilt = [build(spec) for spec in specs]

        assert rebuilt[0].marker == "custom" and rebuilt[0].stages == []
        assert rebuilt[1].count == 3 and rebuilt[1].body.stages == []
        assert rebuilt[2].count == 3
        assert to_spec(rebuilt[0]) == specs[0]
        assert to_spec(rebuilt[1]) == specs[1]
        assert to_spec(rebuilt[2]) == specs[2]

    def test_unreadable_optional_structural_constructor_values_report_location(self):
        from saealib.strategies import TemplateStrategy

        class UnreadableRepeat(Repeat):
            def __init__(self, body, count=1, marker="hidden"):
                super().__init__(body, count)

        class UnreadablePipeline(Pipeline):
            def __init__(self, marker="hidden"):
                super().__init__([])

        class UnreadableTemplate(TemplateStrategy):
            def __init__(self, marker="hidden"):
                super().__init__(Pipeline([]))

        register("UnreadableRepeatRoundTrip")(UnreadableRepeat)
        register("UnreadablePipelineRoundTrip")(UnreadablePipeline)
        register("UnreadableTemplateRoundTrip")(UnreadableTemplate)

        values = [
            UnreadableRepeat(Pipeline([])),
            UnreadablePipeline(),
            UnreadableTemplate(),
        ]
        roots = ("value", "value", "declaration")
        for value, root in zip(values, roots, strict=True):
            with pytest.raises(
                ValidationError, match=rf"{root}\['marker'\].*unavailable"
            ):
                to_spec(value)

    def test_private_fallback_does_not_read_stage_class_attributes(self):
        class RefsParameterStage(Stage):
            def __init__(self, refs=()):
                super().__init__()
                self.extra_refs = refs

            def execute(self, state):
                return state

        register("RefsParameterRoundTrip")(RefsParameterStage)

        with pytest.raises(
            ValidationError,
            match=r"value\['refs'\].*constructor value is unavailable",
        ):
            to_spec(RefsParameterStage((1, 2)))

    def test_stage_identity_metadata_skips_read_only_descriptors(self):
        class ReadOnlyNameStage(Stage):
            def __init__(self):
                self.stages = []

            @property
            def name(self):
                return "read_only"

            def execute(self, state):
                return state

        class UnrestorableNameStage(Stage):
            def __init__(self):
                self._name = "custom"
                self.stages = []

            @property
            def name(self):
                return self._name

            @name.setter
            def name(self, value):
                if value == "custom":
                    raise AttributeError("name is locked")
                self._name = value

            def execute(self, state):
                return state

        register("ReadOnlyNameRoundTrip")(ReadOnlyNameStage)
        register("UnrestorableNameRoundTrip")(UnrestorableNameStage)

        read_only_spec = to_spec(ReadOnlyNameStage())
        assert "_registry_stage_metadata" not in read_only_spec
        assert isinstance(build(read_only_spec), ReadOnlyNameStage)

        with pytest.raises(ValidationError, match="Cannot restore 'name' metadata"):
            build(to_spec(UnrestorableNameStage()))

    def test_async_submit_callback_manager_is_rejected_with_location(self):
        from saealib.callback import CallbackManager
        from saealib.policies.evaluation import EvaluateAll
        from saealib.stages import AsyncEvaluationSubmitStage

        stage = AsyncEvaluationSubmitStage(
            scheduler=object(),
            planner=EvaluateAll(),
            callback_manager=CallbackManager(),
        )

        with pytest.raises(
            ValidationError,
            match=r"value\['callback_manager'\].*not serializable",
        ):
            to_spec(stage)

    def test_shared_acquisition_function_keeps_component_reference_guidance(self):
        from saealib.acquisition import MeanPrediction
        from saealib.stages import AcquisitionStage

        acquisition = MeanPrediction()
        pipeline = Pipeline(
            [AcquisitionStage(acquisition), AcquisitionStage(acquisition)]
        )

        with pytest.raises(ValidationError, match="ComponentRef"):
            to_spec(pipeline)

    def test_stage_identity_metadata_round_trips_with_positional_params(self):
        register("VariadicMetadataStage")(VariadicMetadataStage)
        stage = VariadicMetadataStage("first", "second")
        stage.name = "custom_variadic"

        spec = to_spec(stage)
        restored = build(spec)

        assert spec["params"] == ["first", "second"]
        assert spec["_registry_stage_metadata"] == {"name": "custom_variadic"}
        assert restored.values == ("first", "second")
        assert restored.name == "custom_variadic"

    def test_var_positional_keeps_identity_metadata_and_rejects_unrepresented_params(
        self,
    ):
        class VariadicLabelStage(Stage):
            def __init__(self, *args, label=""):
                self.args = args
                super().__init__(label=label)

            def execute(self, state):
                return state

        class RequiredEntriesPipeline(Pipeline):
            def __init__(self, *entries, name):
                self.entries = entries
                super().__init__(list(entries), name=name)

        class PositionalLabelStage(Stage):
            def __init__(self, label="", *args):
                self.args = args
                super().__init__(label=label)

            def execute(self, state):
                return state

        register("VariadicLabelRoundTrip")(VariadicLabelStage)
        register("RequiredEntriesRoundTrip")(RequiredEntriesPipeline)
        register("PositionalLabelRoundTrip")(PositionalLabelStage)

        original = VariadicLabelStage(1, 2, label="custom")
        spec = to_spec(original)
        restored = build(spec)

        assert spec["params"] == [1, 2]
        assert spec["_registry_stage_metadata"] == {"label": "custom"}
        assert restored.args == original.args == (1, 2)
        assert restored.label == original.label == "custom"

        with pytest.raises(
            ValidationError,
            match=r"value\['name'\].*_registry_spec",
        ):
            to_spec(RequiredEntriesPipeline(CountGenerationStage(), name="named"))

        positional_identity = PositionalLabelStage("custom", 1, 2)
        with pytest.raises(
            ValidationError,
            match=r"value\['label'\].*_registry_spec",
        ):
            to_spec(positional_identity)

    def test_evaluation_plan_with_n_eval_keeps_only_the_explicit_mode(self):
        from saealib.stages import FeedbackStage

        original = EvaluationPlanStage(n_eval=3)
        spec = to_spec(original)
        restored = build(spec)
        feedback = build(to_spec(FeedbackStage()))

        assert "planner" not in spec["params"]
        assert spec["params"]["n_eval"] == 3
        assert restored._n_eval == 3
        assert type(feedback).__name__ == "FeedbackStage"

    def test_custom_stage_with_planner_and_n_eval_preserves_planner(self):
        class PlannerStage(Stage):
            def __init__(self, planner, n_eval=2):
                super().__init__()
                self.planner = planner
                self.n_eval = n_eval
                self._n_eval = n_eval

            def execute(self, state):
                return state

        class OptionalPlannerStage(Stage):
            def __init__(self, planner=None, n_eval=2):
                super().__init__()
                self.planner = planner
                self.n_eval = n_eval
                self._n_eval = n_eval

            def execute(self, state):
                return state

        register("CustomPlannerStageRoundTrip")(PlannerStage)
        register("CustomOptionalPlannerStageRoundTrip")(OptionalPlannerStage)

        originals = [
            PlannerStage(RatioEvaluation(0.25)),
            OptionalPlannerStage(RatioEvaluation(0.4)),
        ]
        for original in originals:
            rebuilt = build(to_spec(original))
            assert type(rebuilt) is type(original)
            assert rebuilt.planner is not None and original.planner is not None
            assert rebuilt.planner.ratio == original.planner.ratio
            assert rebuilt.n_eval == original.n_eval

    def test_explicit_stage_callback_manager_is_rejected_with_location(self):
        from saealib.callback import CallbackManager

        stage = AskStage(cast(Any, Engine(power=7)), cbmanager=CallbackManager())

        with pytest.raises(
            ValidationError, match=r"value\['stages'\]\[0\]\['cbmanager'\]"
        ):
            to_spec(Pipeline([stage]))

    def test_control_subclasses_keep_constructor_arguments(self):
        register("MarkedPipeline")(MarkedPipeline)
        register("MarkedRepeat")(MarkedRepeat)
        register("MarkedLoop")(MarkedLoop)
        register("MarkedBranch")(MarkedBranch)

        original = MarkedPipeline(
            [
                MarkedRepeat(Pipeline([CountGenerationStage()]), 2, marker="repeat"),
                MarkedLoop(
                    Pipeline([CountGenerationStage()]),
                    until=AlwaysCondition(),
                    marker="loop",
                ),
                MarkedBranch(
                    AlwaysCondition(),
                    then=Pipeline([CountGenerationStage()]),
                    marker="branch",
                ),
            ],
            marker="pipeline",
        )

        spec = to_spec(original)
        rebuilt = build(spec)

        assert type(rebuilt) is MarkedPipeline
        assert rebuilt.marker == "pipeline"
        assert rebuilt.stages[0].marker == "repeat"
        assert rebuilt.stages[1].marker == "loop"
        assert rebuilt.stages[2].marker == "branch"
        assert spec["params"]["stages"][0]["params"]["count"] == 2
        assert "marker" in spec["params"]

    def test_local_component_reference_subclass_is_rejected(self):
        class LocalReference(ComponentRef):
            pass

        with pytest.raises(ValidationError, match=r"value.*not importable"):
            to_spec(LocalReference("proposer"))

    @pytest.mark.parametrize(
        "kind", ["stage", "pipeline", "control", "condition", "loop_condition"]
    )
    def test_shared_structural_instances_recommend_separate_instances(self, kind):
        stage = CountGenerationStage()
        body = Pipeline([CountGenerationStage()])
        condition = AlwaysCondition()
        shared = {
            "stage": stage,
            "pipeline": body,
            "control": Repeat(body, 2),
            "condition": condition,
            "loop_condition": condition,
        }[kind]
        if kind == "stage" or kind == "pipeline" or kind == "control":
            entries = [shared, shared]
        else:
            if kind == "condition":
                assert Condition not in type(shared).__mro__
                entries = [
                    Branch(cast(Any, shared), then=CountGenerationStage()),
                    Branch(cast(Any, shared), then=CountGenerationStage()),
                ]
            else:
                entries = [
                    Loop(Pipeline([CountGenerationStage()]), until=cast(Any, shared)),
                    Loop(Pipeline([CountGenerationStage()]), until=cast(Any, shared)),
                ]

        with pytest.raises(ValidationError, match=r"use separate instances") as error:
            to_spec(Pipeline(entries))
        assert "ComponentRef" not in str(error.value)

    def test_template_strategy_uses_source_declaration(self):
        from saealib.stages import AskStage
        from saealib.strategies import TemplateStrategy

        original = TemplateStrategy(
            Pipeline([AskStage(ComponentRef("proposer", "algorithm"))])
        )
        restored = build(to_spec(original))
        assert isinstance(restored, TemplateStrategy)
        assert isinstance(restored.declaration.stages[0], AskStage)
        assert restored.declaration.stages[0]._refs == (
            ("_algorithm", ComponentRef("proposer", "algorithm")),
        )

    def test_lambda_error_names_component_location(self):
        with pytest.raises(ValidationError, match=r"value.*kernel"):
            to_spec(Powered(kernel=lambda value: value))

    def test_component_without_constructor_state_names_declaration_location(self):
        from saealib.stages import EvaluationPlanStage

        declaration = Pipeline(
            [EvaluationPlanStage(cast(Any, ComponentWithoutSerializableState(7)))]
        )
        with pytest.raises(ValidationError, match=r"value.*planner.*required"):
            to_spec(declaration)

    def test_empty_and_single_entry_pipelines_round_trip(self):
        from saealib.stages import CountGenerationStage

        for pipeline in (Pipeline(), Pipeline([CountGenerationStage()])):
            rebuilt = build(to_spec(pipeline))
            assert isinstance(rebuilt, Pipeline)
            assert len(rebuilt.stages) == len(pipeline.stages)

    def test_no_init_class_serializes_with_empty_params(self):
        class NoInit:
            pass

        register("NoInit")(NoInit)
        assert to_spec(NoInit()) == {"type": "NoInit", "params": {}}

    def test_no_init_class_round_trips(self):
        class NoInit:
            pass

        register("NoInit")(NoInit)
        rebuilt = build(to_spec(NoInit()))
        assert isinstance(rebuilt, NoInit)

    def test_var_keyword_attrs_merged_into_params(self):
        obj = KwargsPlugin(alpha=1, beta="two")
        assert to_spec(obj) == {
            "type": "KwargsPlugin",
            "params": {"alpha": 1, "beta": "two"},
        }

    def test_var_keyword_round_trips(self):
        original = KwargsPlugin(alpha=1, beta="two")
        rebuilt = build(to_spec(original))
        assert isinstance(rebuilt, KwargsPlugin)
        assert rebuilt.settings == {"alpha": 1, "beta": "two"}

    def test_var_keyword_merged_alongside_named_params(self):
        obj = KwargsWithNamed(name="custom", scale=2.0)
        assert to_spec(obj) == {
            "type": "KwargsWithNamed",
            "params": {"name": "custom", "scale": 2.0},
        }

    def test_var_keyword_nested_spec_value_recurses(self):
        original = KwargsPlugin(engine=Engine(power=5))
        spec = to_spec(original)
        assert spec == {
            "type": "KwargsPlugin",
            "params": {"engine": {"type": "Engine", "params": {"power": 5}}},
        }
        rebuilt = build(spec)
        assert rebuilt.settings["engine"].power == 5

    def test_var_keyword_missing_attr_raises_validation_error(self):
        class DropsKwargs:
            def __init__(self, **settings):
                pass

        register("DropsKwargs")(DropsKwargs)
        with pytest.raises(ValidationError, match=r"DropsKwargs.*settings"):
            to_spec(DropsKwargs(alpha=1))

    def test_var_keyword_non_mapping_attr_raises_validation_error(self):
        class ListKwargs:
            def __init__(self, **settings):
                self.settings = list(settings)

        register("ListKwargs")(ListKwargs)
        with pytest.raises(ValidationError, match=r"not a.*mapping"):
            to_spec(ListKwargs(alpha=1))

    def test_var_keyword_key_colliding_with_named_param_raises(self):
        class Colliding:
            def __init__(self, name="base", **settings):
                self.name = name
                self.settings = {"name": "shadow", "extra": 1}

        register("Colliding")(Colliding)
        with pytest.raises(ValidationError, match=r"'name'.*collides"):
            to_spec(Colliding())

    def test_lambda_raises_validation_error(self):
        with pytest.raises(ValidationError):
            to_spec(lambda x: x)

    def test_nested_function_raises_validation_error(self):
        def inner(x):
            return x

        with pytest.raises(ValidationError):
            to_spec(inner)


class TestStripParams:
    def test_strips_key_from_flat_spec(self):
        spec = {"type": "Engine", "params": {"power": 1, "dim": 3}}
        stripped = _strip_params(spec, "dim")
        assert stripped == {"type": "Engine", "params": {"power": 1}}

    def test_strips_recursively_from_nested_dict_params(self):
        spec = {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 1, "direction": [1]}},
                "dim": 3,
            },
        }
        stripped = _strip_params(spec, "dim", "direction")
        assert stripped == {
            "type": "Car",
            "params": {"engine": {"type": "Engine", "params": {"power": 1}}},
        }

    def test_strips_recursively_from_list_params(self):
        spec = {
            "type": "Convoy",
            "params": [{"type": "Engine", "params": {"power": 1, "dim": 3}}],
        }
        stripped = _strip_params(spec, "dim")
        assert stripped == {
            "type": "Convoy",
            "params": [{"type": "Engine", "params": {"power": 1}}],
        }

    def test_is_non_destructive(self):
        spec = {"type": "Engine", "params": {"power": 1, "dim": 3}}
        _strip_params(spec, "dim")
        assert spec == {"type": "Engine", "params": {"power": 1, "dim": 3}}


class TestTopLevelExport:
    def test_register_importable_from_top_level(self):
        import saealib

        assert saealib.register is register
        assert "register" in saealib.__all__

    def test_get_build_to_spec_not_exported_at_top_level(self):
        import saealib

        assert "build" not in saealib.__all__
        assert "get" not in saealib.__all__
        assert "to_spec" not in saealib.__all__
        assert not hasattr(saealib, "build")
        assert not hasattr(saealib, "get")
        assert not hasattr(saealib, "to_spec")

    def test_get_build_to_spec_importable_via_registry_namespace(self):
        import saealib.registry

        assert saealib.registry.build is build
        assert saealib.registry.get is get
        assert saealib.registry.to_spec is to_spec


class TestInjectParams:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)

    def test_injects_missing_param(self):
        spec = {"type": "Engine", "params": {}}
        injected = _inject_params(spec, power=250)
        assert injected == {"type": "Engine", "params": {"power": 250}}

    def test_does_not_overwrite_existing_param(self):
        spec = {"type": "Engine", "params": {"power": 42}}
        injected = _inject_params(spec, power=250)
        assert injected == {"type": "Engine", "params": {"power": 42}}

    def test_does_not_inject_param_absent_from_signature(self):
        spec = {"type": "Engine", "params": {}}
        injected = _inject_params(spec, not_a_param=1)
        assert injected == {"type": "Engine", "params": {}}

    def test_injects_into_nested_spec(self):
        spec = {
            "type": "Car",
            "params": {"engine": {"type": "Engine", "params": {}}, "wheels": 4},
        }
        injected = _inject_params(spec, power=99)
        assert injected == {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 99}},
                "wheels": 4,
            },
        }

    def test_is_non_destructive(self):
        spec = {"type": "Engine", "params": {}}
        _inject_params(spec, power=250)
        assert spec == {"type": "Engine", "params": {}}
