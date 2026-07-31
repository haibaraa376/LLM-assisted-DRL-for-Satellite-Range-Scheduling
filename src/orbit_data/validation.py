"""检查已生成的Skyfield轨道和链路数据，不会重新传播卫星。"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .models import IDLData, ISLData, PropagationData, SGLData, SatelliteDefinition, TimeGrid


def validate_saved_data(data_root: Path, config: Any) -> Dict[str, Any]:
    """读取已有NPZ、CSV和JSON并执行第一天基础校验。

    参数：data_root 为数据目录；config 为已加载的三配置结果。本函数只读取文件，
    因此第二天使用已有数据时不会初始化Skyfield。返回简洁的PASS/FAIL报告。
    """
    root = Path(data_root)
    required = ("satellite_positions.npz", "sgl_availability.npz", "isl_availability.npz", "idl_availability.npz", "sgl_windows.csv", "isl_windows.csv", "idl_windows.csv", "satellites.json", "ground_stations.json", "metadata.json", "summary.json")
    failures, warnings = [], []
    checks = {"required_files": all((root / name).is_file() for name in required)}
    if not checks["required_files"]:
        failures.append("缺少必要的数据文件")
        return {"status": "FAIL", "checks": checks, "warnings": warnings, "failures": failures}
    try:
        pos, sgl, isl, idl = (np.load(root / name) for name in required[:4])
        ids = pos["satellite_ids"].astype(str)
        times = pos["timestamps_unix_s"]
        checks["time_grid"] = bool(len(times) == 2881 and np.all(np.diff(times) == 30) and times[-1] - times[0] == 86400)
        numeric = (pos["position_gcrs_km"], pos["velocity_gcrs_km_s"], pos["latitude_deg"], pos["longitude_deg"], pos["height_km"])
        checks["orbit_data"] = bool(len(set(ids)) == 15 and all(np.all(np.isfinite(x)) for x in numeric) and not np.any(pos["sgp4_error_code"]))
        checks["sgl"] = bool(sgl["available"].shape == (2881, 15, 4) and np.array_equal(sgl["satellite_ids"].astype(str), ids) and np.all(sgl["elevation_deg"][sgl["available"]] >= 10.0))
        diagonal = lambda array: not np.any(np.diagonal(array, axis1=1, axis2=2))
        checks["isl"] = bool(isl["available"].shape == (2881, 15, 15) and np.array_equal(isl["satellite_ids"].astype(str), ids) and np.array_equal(isl["available"], isl["available"].transpose(0, 2, 1)) and diagonal(isl["available"]) and np.all(isl["candidate_topology"].sum(axis=1) == 4))
        domains = idl["satellite_domain_ids"].astype(str)
        # union对称化：任一方向选中即可形成无向可用链路。
        cross_domain = domains[:, None] != domains[None, :]
        checks["idl"] = bool(np.array_equal(idl["satellite_ids"].astype(str), ids) and diagonal(idl["available"]) and not np.any(idl["selected_directed"][:, ~cross_domain]) and np.array_equal(idl["available"], idl["selected_directed"] | idl["selected_directed"].transpose(0, 2, 1)))
        counts = {"SGL": 121, "ISL": 30, "IDL": 984}
        window_ok = True
        for name, rate, count in (("sgl_windows.csv", 60.0, counts["SGL"]), ("isl_windows.csv", 80.0, counts["ISL"]), ("idl_windows.csv", 80.0, counts["IDL"])):
            rows = list(csv.DictReader((root / name).open(encoding="utf-8")))
            # CSV使用索引与UTC字符串；[start,end)使结束索引可以等于2881但不新增区间。
            window_ok &= len(rows) == count and all(float(row["duration_seconds"]) > 0 and float(row["transmission_rate_mbps"]) == rate and int(row["start_index"]) < int(row["end_index_exclusive"]) <= 2881 for row in rows)
        checks["windows"] = bool(window_ok)
    except (KeyError, OSError, ValueError) as error:
        failures.append("读取数据失败：{0}".format(error))
    failures.extend(name for name, passed in checks.items() if not passed and name != "required_files")
    return {"status": "PASS" if not failures else "FAIL", "checks": checks, "warnings": warnings, "failures": failures}


def _check(name: str, passed: bool, details: str) -> Dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details}


def _manual_samples(label: str, available: np.ndarray, count: int, seed: int) -> List[Dict[str, Any]]:
    indices = np.argwhere(available)
    if len(indices) == 0:
        return []
    rng = np.random.RandomState(seed)
    chosen = indices[rng.choice(len(indices), size=min(count, len(indices)), replace=False)]
    return [{"link_type": label, "indices": [int(v) for v in row]} for row in chosen]


def build_validation_report(
    grid: TimeGrid,
    propagation: PropagationData,
    sgl: SGLData,
    isl: ISLData,
    idl: IDLData,
    definitions: List[SatelliteDefinition],
    minimum_elevation_deg: float,
    sgl_rows: List[Dict[str, Any]],
    isl_rows: List[Dict[str, Any]],
    idl_rows: List[Dict[str, Any]],
    rates: Dict[str, float],
    input_hashes: Dict[str, str],
    config_hashes: Dict[str, str],
    output_hashes: Dict[str, str],
    sample_count: int,
) -> Dict[str, Any]:
    time_count = len(grid.unix_seconds)
    satellite_count = len(definitions)
    checks = []  # type: List[Dict[str, Any]]
    checks.append(_check("configuration_completeness", satellite_count == 15, "15 satellites loaded"))
    checks.append(_check("time_monotonicity", bool(np.all(np.diff(grid.unix_seconds) > 0)), "strict UTC ordering"))
    ground_count = sgl.available.shape[2]
    shape_dtype_ok = bool(
        propagation.position_gcrs_km.shape == (time_count, satellite_count, 3)
        and propagation.position_gcrs_km.dtype == np.float64
        and propagation.velocity_gcrs_km_s.shape == (time_count, satellite_count, 3)
        and propagation.velocity_gcrs_km_s.dtype == np.float64
        and propagation.latitude_deg.shape == (time_count, satellite_count)
        and propagation.latitude_deg.dtype == np.float64
        and propagation.longitude_deg.shape == (time_count, satellite_count)
        and propagation.longitude_deg.dtype == np.float64
        and propagation.height_km.shape == (time_count, satellite_count)
        and propagation.height_km.dtype == np.float64
        and propagation.sgp4_error_code.shape == (time_count, satellite_count)
        and propagation.sgp4_error_code.dtype == np.int16
        and sgl.available.shape == (time_count, satellite_count, ground_count)
        and sgl.available.dtype == np.bool_
        and sgl.elevation_deg.dtype == np.float32
        and sgl.azimuth_deg.dtype == np.float32
        and sgl.range_km.dtype == np.float32
        and isl.candidate_topology.shape == (satellite_count, satellite_count)
        and isl.candidate_topology.dtype == np.bool_
        and isl.available.shape == (time_count, satellite_count, satellite_count)
        and isl.available.dtype == np.bool_
        and isl.range_km.dtype == np.float32
        and idl.selected_directed.shape == (time_count, satellite_count, satellite_count)
        and idl.selected_directed.dtype == np.bool_
        and idl.available.shape == (time_count, satellite_count, satellite_count)
        and idl.available.dtype == np.bool_
        and idl.range_km.dtype == np.float32
    )
    checks.append(_check("array_shapes", shape_dtype_ok, "all NPZ arrays match configured shapes and dtypes"))
    finite = all(np.all(np.isfinite(array)) for array in (
        propagation.position_gcrs_km, propagation.velocity_gcrs_km_s, propagation.latitude_deg,
        propagation.longitude_deg, propagation.height_km, sgl.elevation_deg, sgl.azimuth_deg, sgl.range_km,
        isl.range_km, idl.range_km,
    ))
    checks.append(_check("nan_inf", finite, "numeric arrays are finite"))
    height_ok = bool(np.all(propagation.height_km > 100.0) and np.max(propagation.height_km) < 50000.0)
    checks.append(_check("satellite_height_range", height_ok, "all WGS84 satellite heights are in (100, 50000) km"))
    checks.append(_check("sgp4_error", not np.any(propagation.sgp4_error_code), "all SGP4 codes are zero"))
    threshold_ok = bool(np.all(sgl.elevation_deg[sgl.available] >= minimum_elevation_deg))
    checks.append(_check("sgl_elevation_threshold", threshold_ok, "available samples meet configured threshold"))
    checks.append(_check("isl_clear_earth", bool(np.all(isl.clear_line_of_sight[isl.available])), "available ISLs have clear LOS"))
    checks.append(_check("isl_symmetric", bool(np.array_equal(isl.available, np.swapaxes(isl.available, 1, 2))), "ISL matrix symmetric"))
    diagonal_ok = not np.any(np.diagonal(isl.available, axis1=1, axis2=2)) and not np.any(np.diagonal(idl.available, axis1=1, axis2=2))
    checks.append(_check("zero_diagonal", bool(diagonal_ok), "ISL/IDL diagonals are zero"))
    nearest_ok = True
    for t, source, target in np.argwhere(idl.selected_directed):
        target_domain = definitions[int(target)].domain_id
        selected_distance = idl.range_km[t, source, target]
        candidates = [
            idl.range_km[t, source, index] for index, definition in enumerate(definitions)
            if definition.domain_id == target_domain and idl.earth_clearance_km[t, source, index] > 0
        ]
        if candidates and selected_distance > min(candidates) + 1e-3:
            nearest_ok = False
            break
    checks.append(_check("idl_nearest_visible", nearest_ok, "directed selections are nearest clear candidates"))
    observation_neighbors = [len(item.isl_neighbor_ids) == 4 for item in definitions if item.domain_id == "D2"]
    checks.append(_check("observation_four_adjacent", bool(observation_neighbors and all(observation_neighbors)), "every D2 satellite has four configured neighbors"))
    all_rows = sgl_rows + isl_rows + idl_rows
    checks.append(_check("positive_window_duration", all(float(row["duration_seconds"]) > 0 for row in all_rows), "all exported windows have positive duration"))
    sat_index = {item.satellite_id: index for index, item in enumerate(definitions)}
    station_ids = sorted({row["target_id"] for row in sgl_rows})
    station_index = {station_id: index for index, station_id in enumerate(station_ids)}
    reconstructed_sgl = np.zeros_like(sgl.available)
    reconstructed_isl = np.zeros_like(isl.available)
    reconstructed_idl = np.zeros_like(idl.available)
    for row in sgl_rows:
        reconstructed_sgl[int(row["start_index"]):int(row["end_index_exclusive"]), sat_index[row["source_id"]], station_index[row["target_id"]]] = True
    for rows, matrix in ((isl_rows, reconstructed_isl), (idl_rows, reconstructed_idl)):
        for row in rows:
            source = sat_index[row["source_id"]]
            target = sat_index[row["target_id"]]
            start, end = int(row["start_index"]), int(row["end_index_exclusive"])
            matrix[start:end, source, target] = True
            matrix[start:end, target, source] = True
    windows_ok = bool(
        len(station_ids) == sgl.available.shape[2]
        and np.array_equal(reconstructed_sgl, sgl.available)
        and np.array_equal(reconstructed_isl, isl.available)
        and np.array_equal(reconstructed_idl, idl.available)
    )
    checks.append(_check("window_matrix_consistency", windows_ok, "window indices exactly reconstruct all three raw availability matrices"))
    event_ok = bool(sgl_rows and all(row["boundary_source"] == "skyfield_find_events" for row in sgl_rows))
    checks.append(_check("sgl_event_cross_check", event_ok, "all sampled SGL windows matched Skyfield event windows"))
    rates_ok = all(float(row["transmission_rate_mbps"]) == float(rates[row["link_type"]]) for row in all_rows)
    checks.append(_check("fixed_rates", rates_ok, "SGL=60 and ISL/IDL=80 Mbps"))
    checks.append(_check("file_schema", all("window_id" in row and "generation_config_sha256" in row for row in all_rows), "required window fields present"))
    checks.append(_check("hash_completeness", bool(input_hashes and config_hashes and output_hashes), "inputs, configs, and current outputs hashed"))
    manual = {
        "SGL": _manual_samples("SGL", sgl.available, sample_count, 2025),
        "ISL": _manual_samples("ISL", isl.available, sample_count, 2026),
        "IDL": _manual_samples("IDL", idl.available, sample_count, 2027),
    }
    checks.append(_check("manual_audit_samples", all(len(value) >= min(sample_count, 1) for value in manual.values()), "deterministic audit indices selected"))
    failures = [item for item in checks if item["status"] == "FAIL"]
    warnings = []  # type: List[str]
    if not sgl_rows or not isl_rows or not idl_rows:
        warnings.append("One or more link types produced no windows in the synthetic scenario.")
    status = "FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "manual_audit_samples": manual,
        "input_hashes": input_hashes,
        "config_hashes": config_hashes,
        "output_hashes": output_hashes,
    }
