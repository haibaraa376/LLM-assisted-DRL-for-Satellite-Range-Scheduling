"""Configuration loading, deterministic merge, and strict validation."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from .hashing import sha256_file, sha256_object
from .models import GroundStationDefinition, SatelliteDefinition


class ConfigurationError(ValueError):
    """Raised when a frozen experiment configuration is incomplete or invalid."""


def deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigurationError("YAML root must be a mapping: {0}".format(path))
    return value


def get_path(config: Mapping[str, Any], dotted_path: str) -> Any:
    value = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


REQUIRED_RUNTIME_PATHS = (
    "time.start_utc",
    "time.duration_seconds",
    "time.coarse_step_seconds",
    "time.boundary_refinement_tolerance_seconds",
    "orbit_input.mode",
    "orbit_input.source_file",
    "orbit_input.max_abs_days_from_element_epoch",
    "orbit_input.sgp4_gravity_model",
    "constellation.total_satellites",
    "constellation.task_generating_domain_id",
    "geometry.earth_occlusion.model",
    "geometry.earth_occlusion.sphere_radius_km",
    "geometry.earth_occlusion.safety_margin_km",
    "SGL.minimum_elevation_degrees",
    "SGL.use_skyfield_find_events",
    "IDL.symmetrization_rule",
    "window_generation.minimum_window_duration_seconds",
    "outputs.root_directory",
)


@dataclass(frozen=True)
class ResolvedConfiguration:
    data: Dict[str, Any]
    source_paths: List[Path]
    source_hashes: Dict[str, str]
    resolved_sha256: str

    @property
    def satellites(self) -> List[SatelliteDefinition]:
        return [SatelliteDefinition.from_dict(item) for item in self.data["satellites"]]

    @property
    def ground_station_definitions(self) -> List[GroundStationDefinition]:
        return [GroundStationDefinition.from_dict(item) for item in self.data["ground_stations"]]


def load_configuration(
    paper_path: Path,
    choices_path: Path,
    constellation_path: Path,
    ground_stations_path: Path,
) -> ResolvedConfiguration:
    paths = [Path(paper_path), Path(choices_path), Path(constellation_path), Path(ground_stations_path)]
    merged = {}  # type: Dict[str, Any]
    for path in paths:
        if not path.is_file():
            raise ConfigurationError("Configuration file not found: {0}".format(path))
        merged = deep_merge(merged, _load_yaml(path))
    validate_configuration(merged)
    hashes = {str(path): sha256_file(path) for path in paths}
    return ResolvedConfiguration(merged, paths, hashes, sha256_object(merged))


def load_simplified_configuration(skyfield_path: Path, constellation_path: Path, ground_stations_path: Path) -> ResolvedConfiguration:
    """读取精简后的三份配置，并映射为既有计算模块使用的字段。"""
    paths = [Path(skyfield_path), Path(constellation_path), Path(ground_stations_path)]
    sky, constellation, stations = (_load_yaml(path) for path in paths)
    t, o, g = sky["time"], sky["orbit"], sky["geometry"]
    data = {"time": {"start_utc": t["start_utc"], "duration_seconds": t["duration_seconds"], "coarse_step_seconds": t["step_seconds"], "boundary_refinement_tolerance_seconds": 1, "include_final_endpoint": t["include_final_endpoint"]}, "orbit_input": {"mode": o["mode"], "source_file": str(constellation_path), "max_abs_days_from_element_epoch": o["max_epoch_difference_days"], "sgp4_gravity_model": o["gravity_model"], "sgp4_operation_mode": o["operation_mode"], "require_epoch_proximity_check": True}, "satellites": constellation["satellites"], "ground_stations": stations["ground_stations"], "constellation": {"total_satellites": 15, "task_generating_domain_id": "D2", "domain_assignment": {"communication_satellite_ids": ["cs01","cs02","cs03","cs04","cs05"], "observation_satellite_ids": ["os01","os02","os03","os04","os05"], "navigation_satellite_ids": ["ns01","ns02","ns03","ns04","ns05"]}}, "geometry": {"earth_occlusion": {"model":"sphere", "sphere_radius_km":g["earth_radius_km"], "safety_margin_km":g["safety_margin_km"], "tangent_is_blocked":g["tangent_is_blocked"]}}, "SGL": {"minimum_elevation_degrees":sky["sgl"]["minimum_elevation_deg"], "use_skyfield_find_events":sky["sgl"]["use_find_events"], "require_positive_altitude":sky["sgl"]["require_positive_altitude"], "maximum_range_filter_enabled":False}, "ISL":{"links_are_undirected":sky["isl"]["undirected"], "maximum_range_filter_enabled":False}, "IDL":{"symmetrization_rule":sky["idl"]["symmetrization"], "tie_break_rule":sky["idl"]["tie_break"], "directed_selection":True, "recompute_each_time_sample":True, "maximum_range_filter_enabled":False}, "rates":{"SGL_mbps":sky["rates"]["sgl_mbps"], "ISL_mbps":sky["rates"]["isl_mbps"], "IDL_mbps":sky["rates"]["idl_mbps"], "SGL_bandwidth_hz":sky["rates"]["sgl_bandwidth_hz"]}, "window_generation":{"merge_consecutive_true_samples":True,"refine_transition_boundaries":True,"minimum_window_duration_seconds":sky["windows"]["minimum_duration_seconds"],"bridge_false_gaps_seconds":sky["windows"]["bridge_false_gaps_seconds"]}, "performance":sky["performance"], "environment_data_contract":{"geometry_sample_step_seconds":t["step_seconds"]}, "outputs":{"root_directory":sky["output"]["directory"],"overwrite_existing":True,"positions_npz":"satellite_positions.npz","SGL_availability_npz":"sgl_availability.npz","ISL_availability_npz":"isl_availability.npz","IDL_availability_npz":"idl_availability.npz","SGL_windows_csv":"sgl_windows.csv","ISL_windows_csv":"isl_windows.csv","IDL_windows_csv":"idl_windows.csv","satellites_json":"satellites.json","ground_stations_json":"ground_stations.json","summary_json":"summary.json","validation_json":"validation.json","manifest_yaml":"metadata.json"}, "experiment_identity":{"experiment_id":"rappo_skyfield_day1"}, "validation":{"fail_on_sgp4_error":True,"random_manual_audit_samples_per_link_type":10}}
    validate_configuration(data)
    hashes = {str(path): sha256_file(path) for path in paths}
    return ResolvedConfiguration(data, paths, hashes, sha256_object(data))


def _unresolved_paths(config: Mapping[str, Any], paths: Iterable[str]) -> List[str]:
    missing = []
    for path in paths:
        try:
            value = get_path(config, path)
        except KeyError:
            missing.append(path)
            continue
        if value is None or value == "":
            missing.append(path)
    return missing


def validate_configuration(config: Mapping[str, Any]) -> None:
    failures = []  # type: List[str]
    unresolved = _unresolved_paths(config, REQUIRED_RUNTIME_PATHS)
    if unresolved:
        failures.append("unresolved required paths: {0}".format(", ".join(sorted(unresolved))))

    satellites_raw = config.get("satellites") or []
    stations_raw = config.get("ground_stations") or []
    if len(satellites_raw) != 15:
        failures.append("satellites must contain exactly 15 records")
    if len(stations_raw) != 4:
        failures.append("ground_stations must contain exactly 4 records")

    satellite_ids = [item.get("id") for item in satellites_raw]
    satnums = [item.get("numeric_satnum") for item in satellites_raw]
    if len(set(satellite_ids)) != len(satellite_ids):
        failures.append("satellite ids must be unique")
    if len(set(satnums)) != len(satnums):
        failures.append("numeric_satnum values must be unique")

    expected_domains = {"D1": "communication", "D2": "observation", "D3": "navigation"}
    by_id = {item.get("id"): item for item in satellites_raw}
    if set(item.get("domain_id") for item in satellites_raw) != set(expected_domains):
        failures.append("all three domains D1/D2/D3 must exist")
    for item in satellites_raw:
        satellite_id = str(item.get("id"))
        domain_id = item.get("domain_id")
        if expected_domains.get(domain_id) != item.get("domain_name"):
            failures.append("{0}: domain id/name mismatch".format(satellite_id))
        elements = item.get("orbital_elements") or {}
        required_elements = (
            "epoch_utc", "eccentricity", "argument_of_perigee_deg", "inclination_deg",
            "mean_anomaly_deg", "mean_motion_rev_per_day", "raan_deg", "bstar",
            "mean_motion_dot_rev_per_day2", "mean_motion_ddot_rev_per_day3",
        )
        for field in required_elements:
            if elements.get(field) is None:
                failures.append("satellites.{0}.orbital_elements.{1}".format(satellite_id, field))
        if elements.get("eccentricity") is not None and not 0 <= float(elements["eccentricity"]) < 1:
            failures.append("{0}: eccentricity must be in [0,1)".format(satellite_id))
        neighbors = list(item.get("ISL_neighbor_ids") or [])
        if len(neighbors) != 4 or len(set(neighbors)) != 4:
            failures.append("{0}: exactly four unique ISL neighbors required".format(satellite_id))
        if satellite_id in neighbors:
            failures.append("{0}: self-loop in ISL neighbors".format(satellite_id))
        for neighbor_id in neighbors:
            neighbor = by_id.get(neighbor_id)
            if neighbor is None:
                failures.append("{0}: unknown neighbor {1}".format(satellite_id, neighbor_id))
            elif neighbor.get("domain_id") != domain_id:
                failures.append("{0}: cross-domain neighbor {1}".format(satellite_id, neighbor_id))

    if config.get("ISL", {}).get("links_are_undirected"):
        for item in satellites_raw:
            for neighbor_id in item.get("ISL_neighbor_ids") or []:
                if item.get("id") not in (by_id.get(neighbor_id, {}).get("ISL_neighbor_ids") or []):
                    failures.append("asymmetric ISL adjacency: {0}-{1}".format(item.get("id"), neighbor_id))

    assignments = config.get("constellation", {}).get("domain_assignment", {})
    assigned = []  # type: List[str]
    for field in ("communication_satellite_ids", "observation_satellite_ids", "navigation_satellite_ids"):
        values = assignments.get(field)
        if not isinstance(values, Sequence) or isinstance(values, str):
            failures.append("constellation.domain_assignment.{0} must be a list".format(field))
        else:
            assigned.extend(values)
    if len(assigned) != len(set(assigned)):
        failures.append("domain membership lists overlap")
    if set(assigned) != set(satellite_ids):
        failures.append("domain membership must assign every satellite exactly once")

    try:
        start = str(get_path(config, "time.start_utc"))
        if not start.endswith("Z"):
            raise ValueError
        datetime.fromisoformat(start[:-1] + "+00:00")
    except (KeyError, ValueError):
        failures.append("time.start_utc must be ISO-8601 UTC with Z suffix")
    if config.get("time", {}).get("duration_seconds") != 86400:
        failures.append("time.duration_seconds must equal paper-reported 86400")

    station_ids = [item.get("id") for item in stations_raw]
    if len(set(station_ids)) != len(station_ids):
        failures.append("ground station ids must be unique")
    for item in stations_raw:
        station_id = item.get("id")
        for field in ("name", "latitude_deg", "longitude_deg", "elevation_m", "antenna_rotation_speed_deg_per_second"):
            if item.get(field) is None:
                failures.append("ground_stations.{0}.{1}".format(station_id, field))
        if item.get("latitude_deg") is not None and not -90 <= float(item["latitude_deg"]) <= 90:
            failures.append("{0}: invalid latitude".format(station_id))
        if item.get("longitude_deg") is not None and not -180 <= float(item["longitude_deg"]) <= 180:
            failures.append("{0}: invalid longitude".format(station_id))
        rotation_speed = item.get("antenna_rotation_speed_deg_per_second")
        if rotation_speed is not None:
            if isinstance(rotation_speed, bool) or not isinstance(rotation_speed, (int, float)) or not math.isfinite(float(rotation_speed)) or float(rotation_speed) <= 0.0:
                failures.append("{0}: antenna_rotation_speed_deg_per_second must be finite and > 0".format(station_id))

    if config.get("IDL", {}).get("symmetrization_rule") not in ("none", "union", "mutual_only"):
        failures.append("IDL.symmetrization_rule must be none, union, or mutual_only")
    if config.get("IDL", {}).get("tie_break_rule") != "lowest_satellite_id":
        failures.append("IDL.tie_break_rule currently only supports lowest_satellite_id")
    if config.get("IDL", {}).get("directed_selection") is not True:
        failures.append("IDL.directed_selection must be true because raw directed selections are mandatory")
    if config.get("IDL", {}).get("recompute_each_time_sample") is not True:
        failures.append("IDL.recompute_each_time_sample must be true; cached selection is not implemented")
    if config.get("time", {}).get("boundary_refinement_tolerance_seconds", 0) <= 0:
        failures.append("time.boundary_refinement_tolerance_seconds must be > 0")
    if config.get("environment_data_contract", {}).get("geometry_sample_step_seconds") != config.get("time", {}).get("coarse_step_seconds"):
        failures.append("environment_data_contract.geometry_sample_step_seconds must equal time.coarse_step_seconds")
    chunk = config.get("performance", {}).get("chunk_size_time")
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0):
        failures.append("performance.chunk_size_time must be null, 0, or a positive integer")
    if failures:
        raise ConfigurationError("Configuration validation failed:\n- " + "\n- ".join(failures))
