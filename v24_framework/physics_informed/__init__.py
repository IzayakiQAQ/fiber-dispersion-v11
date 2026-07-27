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
from .fisher_residual import (
    FisherResidualConfig,
    FisherResidualResult,
    PhysicsFisherCompensationPipeline,
    PhysicsFisherCompensationResult,
    PhysicsFisherResidualCorrector,
    PoissonCenterEstimate,
    cross_power_clock_crlb_ps,
)

__all__ = [
    "AdaptiveCompensationResult",
    "ConditionSeries",
    "FisherResidualConfig",
    "FisherResidualResult",
    "HistogramRecord",
    "PhysicsAdaptiveCompensator",
    "PhysicsFisherCompensationPipeline",
    "PhysicsFisherCompensationResult",
    "PhysicsFisherResidualCorrector",
    "PhysicsHistogramGenerator",
    "PhysicsParameters",
    "PoissonCenterEstimate",
    "cross_power_clock_crlb_ps",
    "discover_dataset",
    "load_physics_parameters",
    "thin_histogram_counts",
]
