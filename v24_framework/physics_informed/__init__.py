"""Physics-informed extensions for the locked v24 compensator."""

from .adaptive_compensator import (
    AdaptiveCompensationResult,
    PhysicsAdaptiveCompensator,
)
from .dataset import (
    ConditionSeries,
    HistogramRecord,
    discover_dataset,
    thin_histogram_counts,
)
from .forward_model import (
    PhysicsHistogramGenerator,
    PhysicsParameters,
    load_physics_parameters,
)

__all__ = [
    "AdaptiveCompensationResult",
    "ConditionSeries",
    "HistogramRecord",
    "PhysicsAdaptiveCompensator",
    "PhysicsHistogramGenerator",
    "PhysicsParameters",
    "discover_dataset",
    "load_physics_parameters",
    "thin_histogram_counts",
]
