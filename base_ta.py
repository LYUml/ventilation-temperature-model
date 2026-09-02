"""Public import facade for the reusable baseline-temperature models."""

from model.base_ta import (
    SUPPORTED_METHODS,
    BaseTaModel,
    calculate_base_ta,
    calculateBaseTa,
)

__all__ = [
    "SUPPORTED_METHODS",
    "BaseTaModel",
    "calculateBaseTa",
    "calculate_base_ta",
]
