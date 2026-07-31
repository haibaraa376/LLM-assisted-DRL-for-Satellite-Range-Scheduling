"""Ground-station ENU pointing and antenna calibration geometry."""

import math
import numpy as np


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{0} must be finite".format(name))
    return value


def az_el_to_enu_unit_vector(azimuth_deg, elevation_deg):
    azimuth = math.radians(_finite(azimuth_deg, "azimuth_deg"))
    elevation = math.radians(_finite(elevation_deg, "elevation_deg"))
    return np.asarray((math.cos(elevation) * math.sin(azimuth), math.cos(elevation) * math.cos(azimuth), math.sin(elevation)), dtype=np.float64)


def pointing_angular_separation_deg(azimuth_1_deg, elevation_1_deg, azimuth_2_deg, elevation_2_deg):
    first = az_el_to_enu_unit_vector(azimuth_1_deg, elevation_1_deg)
    second = az_el_to_enu_unit_vector(azimuth_2_deg, elevation_2_deg)
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    result = math.degrees(math.acos(cosine))
    if not math.isfinite(result):
        raise ValueError("pointing angular separation is not finite")
    return result


def antenna_calibration_time_seconds(previous_azimuth_deg, previous_elevation_deg, next_azimuth_deg, next_elevation_deg, rotation_speed_deg_per_second):
    speed = _finite(rotation_speed_deg_per_second, "rotation_speed_deg_per_second")
    if speed <= 0.0:
        raise ValueError("rotation_speed_deg_per_second must be > 0")
    return pointing_angular_separation_deg(previous_azimuth_deg, previous_elevation_deg, next_azimuth_deg, next_elevation_deg) / speed
