"""Satellite-ground geometry and Skyfield event-window helpers."""

from typing import Any, List, Tuple

import numpy as np
from skyfield.api import wgs84

from .models import GroundStationDefinition, SGLData, TimeGrid


def compute_sgl(
    satellites: List[Any],
    stations: List[GroundStationDefinition],
    grid: TimeGrid,
    minimum_elevation_degrees: float,
    maximum_range_km: float = None,
) -> SGLData:
    shape = (len(grid.unix_seconds), len(satellites), len(stations))
    elevation = np.empty(shape, dtype=np.float32)
    azimuth = np.empty(shape, dtype=np.float32)
    ranges = np.empty(shape, dtype=np.float32)
    for station_index, station_definition in enumerate(stations):
        station = wgs84.latlon(
            station_definition.latitude_deg,
            station_definition.longitude_deg,
            elevation_m=station_definition.elevation_m,
        )
        for satellite_index, satellite in enumerate(satellites):
            topocentric = (satellite - station).at(grid.skyfield_time)
            altitude_angle, azimuth_angle, distance = topocentric.altaz()
            elevation[:, satellite_index, station_index] = np.asarray(altitude_angle.degrees, dtype=np.float32)
            azimuth[:, satellite_index, station_index] = np.asarray(azimuth_angle.degrees, dtype=np.float32)
            ranges[:, satellite_index, station_index] = np.asarray(distance.km, dtype=np.float32)
    available = elevation >= float(minimum_elevation_degrees)
    if maximum_range_km is not None:
        available &= ranges <= float(maximum_range_km)
    if not all(np.all(np.isfinite(value)) for value in (elevation, azimuth, ranges)):
        raise ValueError("SGL geometry produced NaN or Inf")
    return SGLData(available.astype(bool), elevation, azimuth, ranges)


def find_sgl_event_windows(
    satellite: Any,
    station: GroundStationDefinition,
    timescale: Any,
    start_utc: Any,
    end_utc: Any,
    minimum_elevation_degrees: float,
) -> List[Tuple[float, float]]:
    """Reconstruct [start,end) windows without assuming rise/culminate/set triplets."""
    observer = wgs84.latlon(station.latitude_deg, station.longitude_deg, elevation_m=station.elevation_m)
    t0 = timescale.from_datetime(start_utc)
    t1 = timescale.from_datetime(end_utc)
    initial_altitude = float((satellite - observer).at(t0).altaz()[0].degrees)
    visible = initial_altitude >= float(minimum_elevation_degrees)
    current_start = float(start_utc.timestamp()) if visible else None
    event_times, event_codes = satellite.find_events(observer, t0, t1, altitude_degrees=minimum_elevation_degrees)
    windows = []  # type: List[Tuple[float, float]]
    for event_time, code in zip(event_times, event_codes):
        unix_s = float(event_time.utc_datetime().timestamp())
        code_int = int(code)
        if code_int == 0 and not visible:
            current_start = unix_s
            visible = True
        elif code_int == 2 and visible:
            if current_start is not None and unix_s > current_start:
                windows.append((current_start, unix_s))
            current_start = None
            visible = False
    if visible and current_start is not None:
        end_unix_s = float(end_utc.timestamp())
        if end_unix_s > current_start:
            windows.append((current_start, end_unix_s))
    return windows
