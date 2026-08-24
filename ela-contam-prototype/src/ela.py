from __future__ import annotations

import math


MUAIR_KG_M_S = 1.81625e-5
RHOAIR_KG_M3 = 1.20410
RE_TRANSITION = 30.0


def cm2_per_m2_to_m2_per_m2(value: float) -> float:
    """Convert cm² leakage area per m² envelope to m²/m²."""
    if value < 0:
        raise ValueError("normalized ELA cannot be negative")
    return value * 1.0e-4


def total_ela_cm2(normalized_ela_cm2_per_m2: float, wall_area_m2: float) -> float:
    if normalized_ela_cm2_per_m2 < 0 or wall_area_m2 <= 0:
        raise ValueError("ELA must be non-negative and wall area must be positive")
    return normalized_ela_cm2_per_m2 * wall_area_m2


def contam_leak3_coefficients(
    normalized_ela_m2_per_m2: float,
    reference_pressure_pa: float,
    discharge_coefficient: float,
    flow_exponent: float,
) -> tuple[float, float]:
    """Return CONTAM plr_leak3 laminar and turbulent coefficients.

    The equations follow NIST's contam-x-jr-ml setFlowCoef implementation.
    A tiny numerical area is used for an ELA=0 solver check; reported physical
    ELA remains zero.
    """
    if normalized_ela_m2_per_m2 < 0:
        raise ValueError("normalized ELA cannot be negative")
    if reference_pressure_pa <= 0:
        raise ValueError("reference pressure must be positive")
    if discharge_coefficient <= 0:
        raise ValueError("discharge coefficient must be positive")
    if not 0.5 <= flow_exponent <= 1.0:
        raise ValueError("flow exponent must be between 0.5 and 1.0")

    area = max(normalized_ela_m2_per_m2, 1.0e-12)
    diameter = math.sqrt(area)
    cturb = (
        math.sqrt(2.0)
        * discharge_coefficient
        * area
        * reference_pressure_pa ** (0.5 - flow_exponent)
    )
    f_transition = MUAIR_KG_M_S * RE_TRANSITION * diameter
    dp_transition = (
        f_transition / (cturb * math.sqrt(RHOAIR_KG_M3))
    ) ** (1.0 / flow_exponent)
    dp_transition = max(dp_transition, 1.0e-10)
    clam = MUAIR_KG_M_S * f_transition / (RHOAIR_KG_M3 * dp_transition)
    return clam, cturb


def air_density_kg_m3(temperature_k: float, pressure_pa: float) -> float:
    """Ideal-gas dry-air density."""
    if temperature_k <= 0 or pressure_pa <= 0:
        raise ValueError("absolute temperature and pressure must be positive")
    return pressure_pa / (287.055 * temperature_k)
