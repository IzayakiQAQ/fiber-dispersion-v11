"""V25 frozen-physics histogram dispersion compensation."""

from .compensator import CompensationResult, V25Compensator
from .config import FrozenConfig, OperatorSettings, PhysicsParameters
from .physics import DirectionKernels, PhysicsHistogramGenerator

__all__ = [
    "CompensationResult",
    "DirectionKernels",
    "FrozenConfig",
    "OperatorSettings",
    "PhysicsHistogramGenerator",
    "PhysicsParameters",
    "V25Compensator",
]
