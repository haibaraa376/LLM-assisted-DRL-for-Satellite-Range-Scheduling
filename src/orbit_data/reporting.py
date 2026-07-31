"""Audit summaries and figures; numerical source-of-truth remains NPZ/CSV."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import GroundStationDefinition, IDLData, ISLData, PropagationData, SGLData, SatelliteDefinition, TimeGrid


def build_data_summary(
    run_id: str,
    grid: TimeGrid,
    definitions: List[SatelliteDefinition],
    stations: List[GroundStationDefinition],
    sgl: SGLData,
    isl: ISLData,
    idl: IDLData,
    sgl_rows: List[Dict[str, Any]],
    isl_rows: List[Dict[str, Any]],
    idl_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    counts = {}
    for item in definitions:
        counts[item.domain_name] = counts.get(item.domain_name, 0) + 1
    all_durations = [float(row["duration_seconds"]) for row in sgl_rows + isl_rows + idl_rows]
    quantiles = {} if not all_durations else {
        str(q): float(np.quantile(all_durations, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    per_satellite = {}
    for satellite in definitions:
        satellite_id = satellite.satellite_id
        per_satellite[satellite_id] = {
            "SGL_window_count": sum(row["source_id"] == satellite_id for row in sgl_rows),
            "ISL_window_count": sum(satellite_id in (row["source_id"], row["target_id"]) for row in isl_rows),
            "IDL_window_count": sum(satellite_id in (row["source_id"], row["target_id"]) for row in idl_rows),
        }
    per_station = {station.station_id: {"SGL_window_count": sum(row["target_id"] == station.station_id for row in sgl_rows)} for station in stations}
    return {
        "run_identity": {"run_id": run_id},
        "time_grid": {"start_utc": str(grid.iso_utc[0]), "end_utc": str(grid.iso_utc[-1]), "sample_count": len(grid.unix_seconds), "step_seconds": grid.step_seconds, "interval_convention": "[start,end)"},
        "satellite_counts_by_domain": counts,
        "ground_station_count": len(stations),
        "SGL_statistics": {"window_count": len(sgl_rows), "available_sample_count": int(np.sum(sgl.available))},
        "ISL_statistics": {"window_count": len(isl_rows), "available_sample_count": int(np.sum(isl.available))},
        "IDL_statistics": {"window_count": len(idl_rows), "available_sample_count": int(np.sum(idl.available))},
        "window_duration_quantiles": quantiles,
        "per_satellite_statistics": per_satellite,
        "per_ground_station_statistics": per_station,
        "warnings": ["Synthetic representative orbits/stations are reproduction choices, not paper-reported identities."],
    }


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def generate_audit_figures(
    figures_root: Path,
    grid: TimeGrid,
    definitions: List[SatelliteDefinition],
    stations: List[GroundStationDefinition],
    propagation: PropagationData,
    sgl: SGLData,
    isl: ISLData,
    idl: IDLData,
    sgl_rows: List[Dict[str, Any]],
    isl_rows: List[Dict[str, Any]],
    idl_rows: List[Dict[str, Any]],
) -> List[Path]:
    figures_root.mkdir(parents=True, exist_ok=True)
    hours = (grid.unix_seconds - grid.unix_seconds[0]) / 3600.0
    outputs = []  # type: List[Path]

    plt.figure(figsize=(10, 5))
    for index, item in enumerate(definitions):
        plt.plot(hours, propagation.height_km[:, index], linewidth=0.8, label=item.satellite_id)
    plt.xlabel("Time since start (h)"); plt.ylabel("WGS84 height (km)"); plt.title("Satellite height over 24 hours"); plt.legend(ncol=3, fontsize=7)
    outputs.append(figures_root / "satellite_height_24h.png"); _save(outputs[-1])

    plt.figure(figsize=(10, 5))
    for index, station in enumerate(stations):
        plt.step(hours, np.sum(sgl.available[:, :, index], axis=1), where="post", label=station.station_id)
    plt.xlabel("Time since start (h)"); plt.ylabel("Visible satellites"); plt.title("Visible satellite count by ground station"); plt.legend()
    outputs.append(figures_root / "ground_station_visible_count.png"); _save(outputs[-1])

    labels = [item.satellite_id for item in definitions]
    counts = [sum(row["source_id"] == label for row in sgl_rows) for label in labels]
    plt.figure(figsize=(10, 5)); plt.bar(labels, counts); plt.xticks(rotation=45); plt.ylabel("SGL windows"); plt.title("SGL window count per satellite")
    outputs.append(figures_root / "sgl_window_count_per_satellite.png"); _save(outputs[-1])

    plt.figure(figsize=(10, 5)); plt.plot(hours, np.sum(isl.available, axis=(1, 2)) / 2.0, label="ISL"); plt.plot(hours, np.sum(idl.available, axis=(1, 2)) / 2.0, label="IDL"); plt.xlabel("Time since start (h)"); plt.ylabel("Active undirected links"); plt.title("Active ISL and IDL links"); plt.legend()
    outputs.append(figures_root / "active_isl_idl_links.png"); _save(outputs[-1])

    plt.figure(figsize=(10, 5))
    for label, rows in (("SGL", sgl_rows), ("ISL", isl_rows), ("IDL", idl_rows)):
        durations = [row["duration_seconds"] for row in rows]
        if durations:
            plt.hist(durations, bins=20, alpha=0.5, label=label)
    plt.xlabel("Window duration (s)"); plt.ylabel("Count"); plt.title("Transmission-window duration distributions"); plt.legend()
    outputs.append(figures_root / "window_duration_distribution.png"); _save(outputs[-1])

    totals = []
    for satellite_id in labels:
        totals.append(sum(row["duration_seconds"] for row in sgl_rows + isl_rows + idl_rows if satellite_id in (row["source_id"], row["target_id"])))
    plt.figure(figsize=(10, 5)); plt.bar(labels, np.asarray(totals) / 3600.0); plt.xticks(rotation=45); plt.ylabel("Total linked-window time (h)"); plt.title("Total window duration per satellite")
    outputs.append(figures_root / "window_total_per_satellite.png"); _save(outputs[-1])

    switches = []
    for source in range(len(definitions)):
        count = 0
        previous = None
        for time_index in range(idl.selected_directed.shape[0]):
            current = tuple(np.flatnonzero(idl.selected_directed[time_index, source]))
            if previous is not None and current != previous:
                count += 1
            previous = current
        switches.append(count)
    plt.figure(figsize=(10, 5)); plt.bar(labels, switches); plt.xticks(rotation=45); plt.ylabel("Target-set changes"); plt.title("IDL target switch count per satellite")
    outputs.append(figures_root / "idl_target_switch_count.png"); _save(outputs[-1])
    return outputs
