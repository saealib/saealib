from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "ACQUISITION_RESULT",
    "ARCHIVES_MAIN",
    "ARCHIVES_PARETO",
    "EVALUATED_OFFSPRING",
    "EVALUATIONS_COUNT",
    "EVALUATIONS_OWNERS",
    "EVALUATIONS_PENDING",
    "EVALUATIONS_PLAN",
    "EVALUATIONS_PLAN_STATE",
    "EVALUATIONS_PLAN_UPDATES",
    "EVALUATION_HANDLES",
    "EVALUATION_NEW_IDS",
    "EVALUATION_REQUEST",
    "EVALUATION_UPDATES",
    "EVALUATION_UPDATE_NEW_IDS",
    "FEEDBACK_ACCUMULATOR",
    "FEEDBACK_RESULT",
    "OPTIMIZATION_STATE_INITIAL_KEYS",
    "PENDING_EVALUATIONS",
    "POPULATIONS_MAIN",
    "PROPOSALS_CURRENT",
    "PROPOSALS_ID_ALLOCATOR",
    "PROPOSALS_OFFSPRING",
    "RUNTIME_ASYNC_FATAL",
    "RUNTIME_CANDIDATE_ID_ALLOCATOR",
    "RUNTIME_DECISION_COUNT",
    "RUNTIME_GENERATION",
    "RUNTIME_REQUEST_ID_ALLOCATOR",
    "RUNTIME_RNG",
    "SCORES",
    "STATE_NAMESPACES",
    "SURROGATES_DEFAULT",
    "SURROGATES_PREDICTIONS",
    "USER_DATA",
    "StateKey",
]


ValueT = TypeVar("ValueT")


STATE_NAMESPACES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    ("populations", "Population state."),
    ("archives", "Archive state."),
    ("proposals", "Proposal state."),
    ("feedback", "Feedback state."),
    ("evaluations", "Evaluation state."),
    ("surrogates", "Surrogate state."),
    ("algorithms", "Algorithm state."),
    ("runtime", "Runtime state."),
    ("user", "User-owned state."),
):
    STATE_NAMESPACES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


@dataclass(frozen=True, kw_only=True)
class StateKey(Generic[ValueT]):
    """Identify one versioned state value."""

    namespace: str
    name: str
    schema_version: int

    def __post_init__(self) -> None:
        validate_name(self.namespace)
        validate_name(self.name)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValidationError("State schema_version must be a positive integer")


POPULATIONS_MAIN = StateKey[object](
    namespace="populations", name="main", schema_version=1
)
ARCHIVES_MAIN = StateKey[object](namespace="archives", name="main", schema_version=1)
ARCHIVES_PARETO = StateKey[object](
    namespace="archives", name="pareto", schema_version=1
)
EVALUATIONS_COUNT = StateKey[object](
    namespace="evaluations", name="count", schema_version=1
)
ACQUISITION_RESULT = StateKey[object](
    namespace="evaluations", name="acquisition_result", schema_version=2
)
SCORES = StateKey[object](namespace="evaluations", name="scores", schema_version=1)
EVALUATED_OFFSPRING = StateKey[object](
    namespace="evaluations", name="evaluated_offspring", schema_version=1
)
EVALUATION_REQUEST = StateKey[object](
    namespace="evaluations", name="request", schema_version=1
)
EVALUATION_UPDATES = StateKey[object](
    namespace="evaluations", name="updates", schema_version=1
)
EVALUATION_UPDATE_NEW_IDS = StateKey[object](
    namespace="evaluations", name="update_new_ids", schema_version=1
)
EVALUATION_NEW_IDS = StateKey[object](
    namespace="evaluations", name="new_ids", schema_version=1
)
EVALUATION_HANDLES = StateKey[object](
    namespace="evaluations", name="handles", schema_version=1
)
EVALUATIONS_OWNERS = StateKey[object](
    namespace="evaluations", name="owners", schema_version=1
)
EVALUATIONS_PLAN = StateKey[object](
    namespace="evaluations", name="plan", schema_version=1
)
EVALUATIONS_PLAN_STATE = StateKey[object](
    namespace="evaluations", name="plan_state", schema_version=1
)
EVALUATIONS_PLAN_UPDATES = StateKey[object](
    namespace="evaluations", name="plan_updates", schema_version=1
)
PENDING_EVALUATIONS = StateKey[object](
    namespace="evaluations", name="pending", schema_version=1
)
EVALUATIONS_PENDING = PENDING_EVALUATIONS
RUNTIME_RNG = StateKey[object](namespace="runtime", name="rng", schema_version=1)
RUNTIME_GENERATION = StateKey[object](
    namespace="runtime", name="generation", schema_version=1
)
RUNTIME_DECISION_COUNT = StateKey[object](
    namespace="runtime", name="decision_count", schema_version=1
)
RUNTIME_ASYNC_FATAL = StateKey[object](
    namespace="runtime", name="async_fatal", schema_version=1
)
RUNTIME_CANDIDATE_ID_ALLOCATOR = StateKey[object](
    namespace="runtime", name="candidate_id_allocator", schema_version=1
)
PROPOSALS_ID_ALLOCATOR = StateKey[object](
    namespace="proposals", name="proposal_id_allocator", schema_version=1
)
RUNTIME_REQUEST_ID_ALLOCATOR = StateKey[object](
    namespace="runtime", name="request_id_allocator", schema_version=1
)
SURROGATES_DEFAULT = StateKey[object](
    namespace="surrogates", name="default", schema_version=1
)
SURROGATES_PREDICTIONS = StateKey[object](
    namespace="surrogates", name="predictions", schema_version=1
)
FEEDBACK_RESULT = StateKey[object](
    namespace="feedback", name="result", schema_version=1
)
FEEDBACK_ACCUMULATOR = StateKey[object](
    namespace="feedback", name="accumulator", schema_version=1
)
PROPOSALS_OFFSPRING = StateKey[object](
    namespace="proposals", name="offspring", schema_version=1
)
# The current proposal ID is transport state: AskStage writes it and the
# feedback delivery path reads it until the proposal's tell has completed.
PROPOSALS_CURRENT = StateKey[object](
    namespace="proposals", name="current", schema_version=1
)
USER_DATA = StateKey[object](namespace="user", name="data", schema_version=1)

# These keys exist before the strategy graph starts.  The transient
# evaluation values are represented by ``OptimizationState`` fields rather
# than by the current StateStore field map, but they are still initialized to
# ``None``/an empty container and are therefore visible to the graph contract.
# ``CompileContext.initial_state_keys`` accepts a caller-specific set so a
# graph that starts from a different state can remain strict.
OPTIMIZATION_STATE_INITIAL_KEYS = frozenset(
    {
        POPULATIONS_MAIN,
        ARCHIVES_MAIN,
        ARCHIVES_PARETO,
        RUNTIME_RNG,
        RUNTIME_GENERATION,
        RUNTIME_DECISION_COUNT,
        RUNTIME_CANDIDATE_ID_ALLOCATOR,
        RUNTIME_REQUEST_ID_ALLOCATOR,
        PROPOSALS_ID_ALLOCATOR,
        EVALUATIONS_COUNT,
        EVALUATIONS_OWNERS,
        EVALUATIONS_PENDING,
        EVALUATIONS_PLAN,
        EVALUATIONS_PLAN_STATE,
        EVALUATIONS_PLAN_UPDATES,
        RUNTIME_ASYNC_FATAL,
        FEEDBACK_ACCUMULATOR,
        USER_DATA,
        PROPOSALS_OFFSPRING,
        EVALUATED_OFFSPRING,
        SCORES,
        ACQUISITION_RESULT,
        SURROGATES_PREDICTIONS,
        EVALUATION_REQUEST,
        EVALUATION_UPDATES,
        EVALUATION_UPDATE_NEW_IDS,
        EVALUATION_NEW_IDS,
        EVALUATION_HANDLES,
        FEEDBACK_RESULT,
    }
)
