"""Deterministic NPZ, CSV, and JSON exporters following output_schema.yaml."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .models import GroundStationDefinition, IDLData, ISLData, SGLData, SatelliteDefinition, TimeGrid, WindowSpan
from .windows import match_refined_event_window, merge_boolean_series_to_windows


COMMON_WINDOW_COLUMNS = [
    "window_id", "link_type", "source_id", "target_id", "source_domain", "target_domain",
    "start_index", "end_index_exclusive", "start_time_utc", "end_time_utc", "duration_seconds",
    "sample_count", "mean_range_km", "min_range_km", "max_range_km",
    "transmission_rate_mbps", "generation_config_sha256",
]
SGL_EXTRA_COLUMNS = [
    "max_elevation_deg", "mean_elevation_deg", "azimuth_at_start_deg", "azimuth_at_end_deg", "boundary_source",
]
ISL_EXTRA_COLUMNS = ["topology_rule", "earth_clearance_min_km", "boundary_source"]
IDL_EXTRA_COLUMNS = [
    "selection_rule", "symmetrization_rule", "selected_by_source", "earth_clearance_min_km", "boundary_source",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: {0}".format(type(value).__name__))


def write_json(path: Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=_json_default)
        stream.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")
                formatted[key] = "{0:.9f}".format(value) if isinstance(value, float) else value
            writer.writerow(formatted)


def unix_to_iso_utc(unix_s: float) -> str:
    return datetime.fromtimestamp(float(unix_s), tz=timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sample_slice(span: WindowSpan, length: int) -> slice:
    return slice(span.start_index, min(span.end_index_exclusive, length))


def _base_window_row(
    link_type: str,
    source_id: str,
    target_id: str,
    source_domain: str,
    target_domain: str,
    span: WindowSpan,
    ranges: np.ndarray,
    rate_mbps: float,
    config_sha256: str,
) -> Dict[str, Any]:
    values = ranges[_sample_slice(span, len(ranges))]
    return {
        "window_id": "",
        "link_type": link_type,
        "source_id": source_id,
        "target_id": target_id,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "start_index": span.start_index,
        "end_index_exclusive": span.end_index_exclusive,
        "start_time_utc": unix_to_iso_utc(span.start_unix_s),
        "end_time_utc": unix_to_iso_utc(span.end_unix_s),
        "duration_seconds": float(span.end_unix_s - span.start_unix_s),
        "sample_count": int(max(0, span.end_index_exclusive - span.start_index)),
        "mean_range_km": float(np.mean(values)),
        "min_range_km": float(np.min(values)),
        "max_range_km": float(np.max(values)),
        "transmission_rate_mbps": float(rate_mbps),
        "generation_config_sha256": config_sha256,
    }


def _finalize_window_ids(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows.sort(key=lambda item: (item["link_type"], item["source_id"], item["target_id"], item["start_time_utc"]))
    counters = {}  # type: Dict[str, int]
    for row in rows:
        link_type = str(row["link_type"])
        counters[link_type] = counters.get(link_type, 0) + 1
        row["window_id"] = "{0}-{1:06d}".format(link_type, counters[link_type])
    return rows


def build_sgl_window_rows(
    data: SGLData,
    grid: TimeGrid,
    satellites: List[SatelliteDefinition],
    stations: List[GroundStationDefinition],
    minimum_duration_seconds: float,
    rate_mbps: float,
    config_sha256: str,
    refined_event_windows: Optional[Dict[Tuple[int, int], List[Tuple[float, float]]]] = None,
    merge_consecutive_true_samples: bool = True,
    bridge_false_gaps_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    for sat_index, satellite in enumerate(satellites):
        for station_index, station in enumerate(stations):
            spans = merge_boolean_series_to_windows(
                data.available[:, sat_index, station_index], grid.unix_seconds, minimum_duration_seconds, merge_consecutive_true_samples, bridge_false_gaps_seconds
            )
            if refined_event_windows is not None:
                event_windows = refined_event_windows.get((sat_index, station_index), [])
                spans = [match_refined_event_window(span, event_windows) for span in spans]
            for span in spans:
                sample = _sample_slice(span, len(grid.unix_seconds))
                row = _base_window_row(
                    "SGL", satellite.satellite_id, station.station_id, satellite.domain_name, "ground",
                    span, data.range_km[:, sat_index, station_index], rate_mbps, config_sha256,
                )
                elevation = data.elevation_deg[sample, sat_index, station_index]
                azimuth = data.azimuth_deg[:, sat_index, station_index]
                row.update({
                    "max_elevation_deg": float(np.max(elevation)),
                    "mean_elevation_deg": float(np.mean(elevation)),
                    "azimuth_at_start_deg": float(azimuth[span.start_index]),
                    "azimuth_at_end_deg": float(azimuth[min(span.end_index_exclusive - 1, len(azimuth) - 1)]),
                    "boundary_source": span.boundary_source,
                })
                rows.append(row)
    return _finalize_window_ids(rows)


def build_isl_window_rows(
    data: ISLData,
    grid: TimeGrid,
    satellites: List[SatelliteDefinition],
    minimum_duration_seconds: float,
    rate_mbps: float,
    config_sha256: str,
    merge_consecutive_true_samples: bool = True,
    bridge_false_gaps_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    for source_index in range(len(satellites)):
        for target_index in range(source_index + 1, len(satellites)):
            if not data.candidate_topology[source_index, target_index]:
                continue
            spans = merge_boolean_series_to_windows(
                data.available[:, source_index, target_index], grid.unix_seconds, minimum_duration_seconds, merge_consecutive_true_samples, bridge_false_gaps_seconds
            )
            for span in spans:
                row = _base_window_row(
                    "ISL", satellites[source_index].satellite_id, satellites[target_index].satellite_id,
                    satellites[source_index].domain_name, satellites[target_index].domain_name, span,
                    data.range_km[:, source_index, target_index], rate_mbps, config_sha256,
                )
                row.update({
                    "topology_rule": "configured_adjacent",
                    "earth_clearance_min_km": float(np.min(data.earth_clearance_km[_sample_slice(span, len(grid.unix_seconds)), source_index, target_index])),
                    "boundary_source": span.boundary_source,
                })
                rows.append(row)
    return _finalize_window_ids(rows)


def build_idl_window_rows(
    data: IDLData,
    grid: TimeGrid,
    satellites: List[SatelliteDefinition],
    minimum_duration_seconds: float,
    rate_mbps: float,
    config_sha256: str,
    symmetrization_rule: str,
    merge_consecutive_true_samples: bool = True,
    bridge_false_gaps_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = []  # type: List[Dict[str, Any]]
    for source_index in range(len(satellites)):
        for target_index in range(source_index + 1, len(satellites)):
            if satellites[source_index].domain_id == satellites[target_index].domain_id:
                continue
            spans = merge_boolean_series_to_windows(
                data.available[:, source_index, target_index], grid.unix_seconds, minimum_duration_seconds, merge_consecutive_true_samples, bridge_false_gaps_seconds
            )
            for span in spans:
                sample = _sample_slice(span, len(grid.unix_seconds))
                forward = bool(np.any(data.selected_directed[sample, source_index, target_index]))
                reverse = bool(np.any(data.selected_directed[sample, target_index, source_index]))
                selected_by = "both" if forward and reverse else satellites[source_index if forward else target_index].satellite_id
                row = _base_window_row(
                    "IDL", satellites[source_index].satellite_id, satellites[target_index].satellite_id,
                    satellites[source_index].domain_name, satellites[target_index].domain_name, span,
                    data.range_km[:, source_index, target_index], rate_mbps, config_sha256,
                )
                row.update({
                    "selection_rule": "nearest_visible_satellite_in_each_other_domain",
                    "symmetrization_rule": symmetrization_rule,
                    "selected_by_source": selected_by,
                    "earth_clearance_min_km": float(np.min(data.earth_clearance_km[sample, source_index, target_index])),
                    "boundary_source": span.boundary_source,
                })
                rows.append(row)
    return _finalize_window_ids(rows)


def export_npz_files(
    root: Path,
    names: Mapping[str, str],
    grid: TimeGrid,
    satellite_ids: np.ndarray,
    satellite_domain_ids: np.ndarray,
    satellite_domain_names: np.ndarray,
    ground_station_ids: np.ndarray,
    propagation: Any,
    sgl: SGLData,
    isl: ISLData,
    idl: IDLData,
) -> None:
    np.savez_compressed(
        root / names["positions_npz"], timestamps_unix_s=grid.unix_seconds,
        timestamps_iso_utc=grid.iso_utc, satellite_ids=satellite_ids,
        position_gcrs_km=propagation.position_gcrs_km, velocity_gcrs_km_s=propagation.velocity_gcrs_km_s,
        latitude_deg=propagation.latitude_deg, longitude_deg=propagation.longitude_deg,
        height_km=propagation.height_km, sgp4_error_code=propagation.sgp4_error_code,
    )
    np.savez_compressed(
        root / names["SGL_availability_npz"], available=sgl.available, elevation_deg=sgl.elevation_deg,
        azimuth_deg=sgl.azimuth_deg, range_km=sgl.range_km,
        satellite_ids=satellite_ids, ground_station_ids=ground_station_ids,
    )
    np.savez_compressed(
        root / names["ISL_availability_npz"], candidate_topology=isl.candidate_topology,
        clear_line_of_sight=isl.clear_line_of_sight, available=isl.available,
        range_km=isl.range_km, earth_clearance_km=isl.earth_clearance_km,
        timestamps_unix_s=grid.unix_seconds, timestamps_iso_utc=grid.iso_utc,
        satellite_ids=satellite_ids, satellite_domain_ids=satellite_domain_ids,
        satellite_domain_names=satellite_domain_names,
    )
    np.savez_compressed(
        root / names["IDL_availability_npz"], selected_directed=idl.selected_directed,
        available=idl.available, range_km=idl.range_km, earth_clearance_km=idl.earth_clearance_km,
        timestamps_unix_s=grid.unix_seconds, timestamps_iso_utc=grid.iso_utc,
        satellite_ids=satellite_ids, satellite_domain_ids=satellite_domain_ids,
        satellite_domain_names=satellite_domain_names,
    )
