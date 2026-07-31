"""Vectorized orbital propagation and geodetic subpoint calculation."""

from typing import Any, List

import numpy as np
from skyfield.api import wgs84

from .models import PropagationData, TimeGrid


SGP4_ERROR_MESSAGES = {
    0: "success",
    1: "mean eccentricity outside range",
    2: "mean motion less than zero",
    3: "perturbed eccentricity outside range",
    4: "semi-latus rectum less than zero",
    5: "epoch elements are sub-orbital",
    6: "satellite has decayed",
}


def _sgp4_error_codes(satellite: Any, unix_seconds: np.ndarray) -> np.ndarray:
    julian = unix_seconds / 86400.0 + 2440587.5
    whole = np.floor(julian)
    fraction = julian - whole
    errors, _, _ = satellite.model.sgp4_array(whole, fraction)
    return np.asarray(errors, dtype=np.int16)


def propagate_satellites(
    satellites: List[Any],
    grid: TimeGrid,
    fail_on_sgp4_error: bool = True,
) -> PropagationData:
    time_count = len(grid.unix_seconds)
    satellite_count = len(satellites)
    position = np.empty((time_count, satellite_count, 3), dtype=np.float64)
    velocity = np.empty_like(position)
    latitude = np.empty((time_count, satellite_count), dtype=np.float64)
    longitude = np.empty_like(latitude)
    height = np.empty_like(latitude)
    errors = np.empty((time_count, satellite_count), dtype=np.int16)
    for index, satellite in enumerate(satellites):
        geocentric = satellite.at(grid.skyfield_time)
        position[:, index, :] = np.asarray(geocentric.position.km).T
        velocity[:, index, :] = np.asarray(geocentric.velocity.km_per_s).T
        geographic = wgs84.geographic_position_of(geocentric)
        latitude[:, index] = np.asarray(geographic.latitude.degrees)
        longitude[:, index] = np.asarray(geographic.longitude.degrees)
        height[:, index] = np.asarray(geographic.elevation.km)
        errors[:, index] = _sgp4_error_codes(satellite, grid.unix_seconds)
    arrays = (position, velocity, latitude, longitude, height)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("propagation produced NaN or Inf")
    if fail_on_sgp4_error and np.any(errors != 0):
        unique = sorted(int(value) for value in np.unique(errors) if value)
        details = ["{0}: {1}".format(code, SGP4_ERROR_MESSAGES.get(code, "unknown")) for code in unique]
        raise RuntimeError("SGP4 propagation error(s): " + "; ".join(details))
    return PropagationData(position, velocity, latitude, longitude, height, errors)
