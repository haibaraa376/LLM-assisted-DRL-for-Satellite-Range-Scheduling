"""第一天Skyfield数据层的生成、检查和摘要命令入口。"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from skyfield.api import load

from .config import ConfigurationError, ResolvedConfiguration, load_simplified_configuration
from .io import (
    COMMON_WINDOW_COLUMNS, IDL_EXTRA_COLUMNS, ISL_EXTRA_COLUMNS, SGL_EXTRA_COLUMNS,
    build_idl_window_rows, build_isl_window_rows, build_sgl_window_rows, export_npz_files,
    write_csv, write_json,
)
from .hashing import sha256_file
from .idl import compute_idl
from .isl import compute_isl
from .propagation import propagate_satellites
from .reporting import build_data_summary, generate_audit_figures
from .satellites import build_satellites
from .sgl import compute_sgl, find_sgl_event_windows
from .time_grid import build_time_grid, parse_utc_z
from .validation import build_validation_report, validate_saved_data


LOGGER = logging.getLogger("orbit_data")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _config_from_args(args: argparse.Namespace) -> ResolvedConfiguration:
    return load_simplified_configuration(Path(args.skyfield), Path(args.constellation), Path(args.ground_stations))


def _validate_epoch_proximity(resolved: ResolvedConfiguration) -> None:
    if not resolved.data["orbit_input"]["require_epoch_proximity_check"]:
        return
    start = parse_utc_z(resolved.data["time"]["start_utc"])
    maximum = float(resolved.data["orbit_input"]["max_abs_days_from_element_epoch"])
    stale = []
    for definition in resolved.satellites:
        epoch = parse_utc_z(str(definition.orbital_elements["epoch_utc"]))
        delta = abs((start - epoch).total_seconds()) / 86400.0
        if delta > maximum:
            stale.append("{0}={1:.3f}d".format(definition.satellite_id, delta))
    if stale:
        raise ConfigurationError("orbit epoch proximity check failed: " + ", ".join(stale))


def _prepare_output_root(root: Path, overwrite_existing: bool) -> None:
    if root.exists() and any(root.iterdir()) and not overwrite_existing:
        raise FileExistsError("output directory is non-empty and overwrite_existing=false: {0}".format(root))
    root.mkdir(parents=True, exist_ok=True)


def _chunked_link_data(compute, positions, chunk_size, *args):
    if not chunk_size:
        return compute(positions, *args)
    chunks = [compute(positions[start:start + chunk_size], *args) for start in range(0, len(positions), chunk_size)]
    first = chunks[0]
    values = {}
    for name in first.__dataclass_fields__:
        value = getattr(first, name)
        values[name] = value if value.ndim == 2 else np.concatenate([getattr(chunk, name) for chunk in chunks], axis=0)
    return type(first)(**values)


def _event_windows(resolved: ResolvedConfiguration, satellites: List[Any], timescale: Any) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    if not resolved.data["SGL"]["use_skyfield_find_events"]:
        return {}
    result = {}
    for sat_index, satellite in enumerate(satellites):
        for station_index, station in enumerate(resolved.ground_station_definitions):
            result[(sat_index, station_index)] = find_sgl_event_windows(
                satellite, station, timescale, parse_utc_z(resolved.data["time"]["start_utc"]),
                parse_utc_z(resolved.data["time"]["start_utc"]) + timedelta(seconds=resolved.data["time"]["duration_seconds"]),
                resolved.data["SGL"]["minimum_elevation_degrees"],
            )
    return result


def generate(args: argparse.Namespace) -> int:
    stage_start = time.perf_counter()
    start_time_utc = _utc_now()
    resolved = _config_from_args(args)
    _validate_epoch_proximity(resolved)
    output_root = Path(resolved.data["outputs"]["root_directory"])
    if getattr(args, "output_root", None):
        if not args.allow_override:
            raise ConfigurationError("--output-root requires --allow-override")
        output_root = Path(args.output_root)
    overwrite = bool(resolved.data["outputs"]["overwrite_existing"])
    if getattr(args, "overwrite", False):
        if not args.allow_override:
            raise ConfigurationError("--overwrite requires --allow-override")
        overwrite = True
    _prepare_output_root(output_root, overwrite)
    timings = {}  # type: Dict[str, float]

    timescale = load.timescale(builtin=True)
    grid = build_time_grid(timescale, resolved.data["time"]["start_utc"], resolved.data["time"]["duration_seconds"], resolved.data["time"]["coarse_step_seconds"], resolved.data["time"]["include_final_endpoint"])
    definitions = resolved.satellites
    stations = resolved.ground_station_definitions
    satellites = build_satellites(resolved.data, definitions, timescale)
    if len(satellites) != len(definitions):
        raise ValueError("orbit input produced {0} satellites; expected {1}".format(len(satellites), len(definitions)))

    marker = time.perf_counter()
    propagation = propagate_satellites(satellites, grid, resolved.data["validation"]["fail_on_sgp4_error"])
    timings["propagation_seconds"] = time.perf_counter() - marker
    sgl_maximum = resolved.data["SGL"]["maximum_range_km"] if resolved.data["SGL"].get("maximum_range_filter_enabled") else None
    marker = time.perf_counter()
    sgl = compute_sgl(satellites, stations, grid, resolved.data["SGL"]["minimum_elevation_degrees"], sgl_maximum)
    if resolved.data["SGL"]["require_positive_altitude"]:
        sgl = type(sgl)(sgl.available & (propagation.height_km[:, :, None] > 0.0), sgl.elevation_deg, sgl.azimuth_deg, sgl.range_km)
    timings["sgl_seconds"] = time.perf_counter() - marker
    earth = resolved.data["geometry"]["earth_occlusion"]
    isl_maximum = resolved.data["ISL"]["maximum_range_km"] if resolved.data["ISL"]["maximum_range_filter_enabled"] else None
    marker = time.perf_counter()
    chunk_size = resolved.data.get("performance", {}).get("chunk_size_time") or None
    isl = _chunked_link_data(compute_isl, propagation.position_gcrs_km, chunk_size, definitions, earth["sphere_radius_km"], earth["safety_margin_km"], earth["tangent_is_blocked"], isl_maximum)
    timings["isl_seconds"] = time.perf_counter() - marker
    idl_maximum = resolved.data["IDL"]["maximum_range_km"] if resolved.data["IDL"]["maximum_range_filter_enabled"] else None
    marker = time.perf_counter()
    idl = _chunked_link_data(compute_idl, propagation.position_gcrs_km, chunk_size, definitions, earth["sphere_radius_km"], earth["safety_margin_km"], earth["tangent_is_blocked"], resolved.data["IDL"]["symmetrization_rule"], idl_maximum)
    timings["idl_seconds"] = time.perf_counter() - marker

    events = _event_windows(resolved, satellites, timescale) if resolved.data["window_generation"]["refine_transition_boundaries"] else {}
    minimum_duration = resolved.data["window_generation"]["minimum_window_duration_seconds"]
    rates = {"SGL": resolved.data["rates"]["SGL_mbps"], "ISL": resolved.data["rates"]["ISL_mbps"], "IDL": resolved.data["rates"]["IDL_mbps"]}
    window = resolved.data["window_generation"]
    window_args = (window["merge_consecutive_true_samples"], window["bridge_false_gaps_seconds"])
    sgl_rows = build_sgl_window_rows(sgl, grid, definitions, stations, minimum_duration, rates["SGL"], resolved.resolved_sha256, events, *window_args)
    isl_rows = build_isl_window_rows(isl, grid, definitions, minimum_duration, rates["ISL"], resolved.resolved_sha256, *window_args)
    idl_rows = build_idl_window_rows(idl, grid, definitions, minimum_duration, rates["IDL"], resolved.resolved_sha256, resolved.data["IDL"]["symmetrization_rule"], *window_args)

    outputs = resolved.data["outputs"]
    satellite_ids = np.asarray([item.satellite_id for item in definitions])
    station_ids = np.asarray([item.station_id for item in stations])
    domain_ids = np.asarray([item.domain_id for item in definitions])
    domain_names = np.asarray([item.domain_name for item in definitions])
    export_npz_files(output_root, outputs, grid, satellite_ids, domain_ids, domain_names, station_ids, propagation, sgl, isl, idl)
    write_csv(output_root / outputs["SGL_windows_csv"], sgl_rows, COMMON_WINDOW_COLUMNS + SGL_EXTRA_COLUMNS)
    write_csv(output_root / outputs["ISL_windows_csv"], isl_rows, COMMON_WINDOW_COLUMNS + ISL_EXTRA_COLUMNS)
    write_csv(output_root / outputs["IDL_windows_csv"], idl_rows, COMMON_WINDOW_COLUMNS + IDL_EXTRA_COLUMNS)
    write_json(output_root / outputs["satellites_json"], [{"id": item.satellite_id, "numeric_satnum": item.numeric_satnum, "domain_id": item.domain_id, "domain_name": item.domain_name, "orbital_elements": item.orbital_elements, "element_epoch_delta_days": 0.0} for item in definitions])
    write_json(output_root / outputs["ground_stations_json"], [{"id": item.station_id, "name": item.name, "latitude_deg": item.latitude_deg, "longitude_deg": item.longitude_deg, "elevation_m": item.elevation_m, "antenna_rotation_speed_deg_per_second": item.antenna_rotation_speed_deg_per_second} for item in stations])
    run_id = resolved.data["experiment_identity"]["experiment_id"]
    summary = build_data_summary(run_id, grid, definitions, stations, sgl, isl, idl, sgl_rows, isl_rows, idl_rows)
    timings["total_before_reporting_seconds"] = time.perf_counter() - stage_start
    summary["performance"] = {"stage_timings_seconds": timings, "chunk_size_time": chunk_size}
    write_json(output_root / outputs["summary_json"], summary)
    generate_audit_figures(output_root / "figures", grid, definitions, stations, propagation, sgl, isl, idl, sgl_rows, isl_rows, idl_rows)

    pre_validation_files = [path for path in output_root.rglob("*") if path.is_file()]
    output_hashes = {str(path): sha256_file(path) for path in pre_validation_files}
    report = build_validation_report(
        grid, propagation, sgl, isl, idl, definitions, resolved.data["SGL"]["minimum_elevation_degrees"],
        sgl_rows, isl_rows, idl_rows, rates, resolved.source_hashes, resolved.source_hashes,
        output_hashes, resolved.data["validation"]["random_manual_audit_samples_per_link_type"],
    )
    write_json(output_root / outputs["validation_json"], report)
    # 精简工程只保留运行环境和场景尺寸，不保留审批、Git 或逐文件哈希链。
    write_json(output_root / outputs["manifest_yaml"], {"generated_at_utc": start_time_utc, "python_version": sys.version.split()[0], "config_files": ["configs/skyfield.yaml", "configs/constellation.yaml", "configs/ground_stations.yaml"], "satellite_count": len(definitions), "ground_station_count": len(stations), "duration_seconds": resolved.data["time"]["duration_seconds"], "step_seconds": resolved.data["time"]["coarse_step_seconds"], "interval": "[start,end)", "rates_mbps": rates})
    # 生成完成后重新读取文件校验，避免把内存中的中间对象当作最终输出。
    write_json(output_root / outputs["validation_json"], validate_saved_data(output_root, resolved))
    LOGGER.info("generation completed status=%s output=%s", report["status"], output_root)
    return 0 if report["status"] != "FAIL" else 2


def validate_existing(args: argparse.Namespace) -> int:
    """检查既有数据并写回简洁验证报告，不调用Skyfield。"""
    report = validate_saved_data(Path(args.data_root), _config_from_args(args))
    write_json(Path(args.data_root) / "validation.json", report)
    print("Skyfield数据检查：{0}".format(report["status"]))
    return 0 if report["status"] == "PASS" else 2


def report_existing(data_root: Path) -> int:
    path = data_root / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skyfield", default="configs/skyfield.yaml")
    parser.add_argument("--constellation", default="configs/constellation.yaml")
    parser.add_argument("--ground-stations", default="configs/ground_stations.yaml", dest="ground_stations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    _add_config_arguments(generate_parser)
    generate_parser.set_defaults(output_root=None, overwrite=False, allow_override=True)
    for name in ("check", "summary"):
        child = subparsers.add_parser(name)
        child.add_argument("--data-root", default="data/skyfield")
        if name == "check":
            _add_config_arguments(child)
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if args.command == "generate":
            return generate(args)
        if args.command == "check":
            return validate_existing(args)
        if args.command == "summary":
            return report_existing(Path(args.data_root))
        parser.error("unknown command")
    except (ConfigurationError, ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        LOGGER.error("%s", error)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
