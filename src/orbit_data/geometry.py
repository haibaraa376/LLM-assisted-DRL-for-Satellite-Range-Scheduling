"""Pure Earth-occultation geometry for inter-satellite links."""

from typing import Tuple

import numpy as np


def segment_clears_earth(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    earth_radius_km: float,
    safety_margin_km: float,
    tangent_is_blocked: bool,
) -> Tuple[bool, float]:
    r1 = np.asarray(r1_km, dtype=np.float64)
    r2 = np.asarray(r2_km, dtype=np.float64)
    if r1.shape != (3,) or r2.shape != (3,):
        raise ValueError("r1_km and r2_km must both have shape (3,)")
    d = r2 - r1
    denominator = float(np.dot(d, d))
    if denominator == 0.0:
        closest_norm = float(np.linalg.norm(r1))
    else:
        u = float(np.clip(-np.dot(r1, d) / denominator, 0.0, 1.0))
        closest_norm = float(np.linalg.norm(r1 + u * d))
    clearance = closest_norm - (float(earth_radius_km) + float(safety_margin_km))
    clear = clearance > 0.0 if tangent_is_blocked else clearance >= 0.0
    return bool(clear), float(clearance)


def segments_clear_earth(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    earth_radius_km: float,
    safety_margin_km: float,
    tangent_is_blocked: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    r1 = np.asarray(r1_km, dtype=np.float64)
    r2 = np.asarray(r2_km, dtype=np.float64)
    if r1.shape != r2.shape or r1.ndim < 1 or r1.shape[-1] != 3:
        raise ValueError("r1_km and r2_km must have identical (...,3) shapes")
    d = r2 - r1
    denominator = np.sum(d * d, axis=-1)
    numerator = -np.sum(r1 * d, axis=-1)
    u = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    u = np.clip(u, 0.0, 1.0)
    closest = r1 + u[..., np.newaxis] * d
    clearance = np.linalg.norm(closest, axis=-1) - (float(earth_radius_km) + float(safety_margin_km))
    clear = clearance > 0.0 if tangent_is_blocked else clearance >= 0.0
    return clear.astype(bool), clearance
