"""
Callback module.

This module contains event classes and the callback manager for
the optimization lifecycle.
"""

from saealib.callback.events import (
    AcquisitionEndEvent,
    AcquisitionStartEvent,
    ConfigurationChange,
    ConfigurationCommittedEvent,
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    InitialEvaluationEndEvent,
    InitialEvaluationStartEvent,
    PostAskEvent,
    PostCrossoverEvent,
    PostEvaluationEvent,
    PostMutationEvent,
    PostSurrogateFitEvent,
    RunEndEvent,
    RunStartEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
)
from saealib.callback.handlers import logging_generation, logging_generation_hv
from saealib.callback.manager import CallbackManager

__all__ = [
    "AcquisitionEndEvent",
    "AcquisitionStartEvent",
    "CallbackManager",
    "ConfigurationChange",
    "ConfigurationCommittedEvent",
    "Event",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "InitialEvaluationEndEvent",
    "InitialEvaluationStartEvent",
    "PostAskEvent",
    "PostCrossoverEvent",
    "PostEvaluationEvent",
    "PostMutationEvent",
    "PostSurrogateFitEvent",
    "RunEndEvent",
    "RunStartEvent",
    "SurrogateEndEvent",
    "SurrogateStartEvent",
    "logging_generation",
    "logging_generation_hv",
]
