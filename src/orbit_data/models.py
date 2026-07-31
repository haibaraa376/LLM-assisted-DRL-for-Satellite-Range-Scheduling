"""Typed data models shared by the orbit-data modules."""

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class SatelliteDefinition:
    satellite_id: str
    numeric_satnum: int
    domain_id: str
    domain_name: str
    orbital_elements: Dict[str, Any]
    isl_neighbor_ids: List[str]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SatelliteDefinition":
        return cls(
            satellite_id=str(value["id"]),
            numeric_satnum=int(value["numeric_satnum"]),
            domain_id=str(value["domain_id"]),
            domain_name=str(value["domain_name"]),
            orbital_elements=dict(value["orbital_elements"]),
            isl_neighbor_ids=list(value.get("ISL_neighbor_ids", [])),
        )


@dataclass(frozen=True)
class GroundStationDefinition:
    station_id: str
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    antenna_rotation_speed_deg_per_second: float

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "GroundStationDefinition":
        rotation_speed = value.get("antenna_rotation_speed_deg_per_second")
        if isinstance(rotation_speed, bool) or not isinstance(rotation_speed, (int, float)):
            raise ValueError("antenna_rotation_speed_deg_per_second must be a numeric value")
        rotation_speed = float(rotation_speed)
        if not math.isfinite(rotation_speed) or rotation_speed <= 0.0:
            raise ValueError("antenna_rotation_speed_deg_per_second must be finite and > 0")
        return cls(
            station_id=str(value["id"]),
            name=str(value["name"]),
            latitude_deg=float(value["latitude_deg"]),
            longitude_deg=float(value["longitude_deg"]),
            elevation_m=float(value["elevation_m"]),
            antenna_rotation_speed_deg_per_second=rotation_speed,
        )


@dataclass(frozen=True)
class TimeGrid:
    start_utc: datetime
    end_utc: datetime
    step_seconds: float
    include_final_endpoint: bool
    unix_seconds: np.ndarray
    iso_utc: np.ndarray
    skyfield_time: Any


@dataclass(frozen=True)
class PropagationData:
    position_gcrs_km: np.ndarray
    velocity_gcrs_km_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    height_km: np.ndarray
    sgp4_error_code: np.ndarray


@dataclass(frozen=True)
class SGLData:
    available: np.ndarray
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    range_km: np.ndarray


@dataclass(frozen=True)
class ISLData:
    candidate_topology: np.ndarray
    clear_line_of_sight: np.ndarray
    available: np.ndarray
    range_km: np.ndarray
    earth_clearance_km: np.ndarray


@dataclass(frozen=True)
class IDLData:
    selected_directed: np.ndarray
    available: np.ndarray
    range_km: np.ndarray
    earth_clearance_km: np.ndarray


@dataclass(frozen=True)
class WindowSpan:
    start_index: int
    end_index_exclusive: int
    start_unix_s: float
    end_unix_s: float
    boundary_source: str = "sampled"
