"""Nearest-visible inter-domain link selection with deterministic tie-breaking."""

from typing import Dict, List

import numpy as np

from .geometry import segments_clear_earth
from .models import IDLData, SatelliteDefinition


def compute_idl(
    position_gcrs_km: np.ndarray,
    definitions: List[SatelliteDefinition],
    earth_radius_km: float,
    safety_margin_km: float,
    tangent_is_blocked: bool,
    symmetrization_rule: str,
    maximum_range_km: float = None,
) -> IDLData:
    positions = np.asarray(position_gcrs_km, dtype=np.float64)
    time_count, satellite_count, _ = positions.shape
    satellite_ids = [item.satellite_id for item in definitions]
    domain_indices = {}  # type: Dict[str, List[int]]
    for index, definition in enumerate(definitions):
        domain_indices.setdefault(definition.domain_id, []).append(index)
    selected = np.zeros((time_count, satellite_count, satellite_count), dtype=bool)
    ranges = np.zeros((time_count, satellite_count, satellite_count), dtype=np.float32)
    clearance = np.zeros_like(ranges)
    clear = np.zeros_like(selected)
    for source_index in range(satellite_count):
        for target_index in range(source_index + 1, satellite_count):
            if definitions[source_index].domain_id == definitions[target_index].domain_id:
                continue
            pair_clear, pair_clearance = segments_clear_earth(
                positions[:, source_index, :], positions[:, target_index, :],
                earth_radius_km, safety_margin_km, tangent_is_blocked,
            )
            pair_range = np.linalg.norm(positions[:, target_index, :] - positions[:, source_index, :], axis=-1)
            for matrix, values in ((clear, pair_clear), (ranges, pair_range), (clearance, pair_clearance)):
                matrix[:, source_index, target_index] = values
                matrix[:, target_index, source_index] = values

    for time_index in range(time_count):
        for source_index, source_definition in enumerate(definitions):
            for target_domain, candidates in domain_indices.items():
                if target_domain == source_definition.domain_id:
                    continue
                eligible = []
                for target_index in candidates:
                    if not clear[time_index, source_index, target_index]:
                        continue
                    distance = float(ranges[time_index, source_index, target_index])
                    if maximum_range_km is not None and distance > float(maximum_range_km):
                        continue
                    eligible.append((distance, satellite_ids[target_index], target_index))
                if eligible:
                    _, _, chosen_index = min(eligible, key=lambda item: (item[0], item[1]))
                    selected[time_index, source_index, chosen_index] = True
    if symmetrization_rule == "none":
        available = selected.copy()
    elif symmetrization_rule == "union":
        available = selected | np.swapaxes(selected, 1, 2)
    elif symmetrization_rule == "mutual_only":
        available = selected & np.swapaxes(selected, 1, 2)
    else:
        raise ValueError("unknown IDL symmetrization rule: {0}".format(symmetrization_rule))
    indices = np.arange(satellite_count)
    available[:, indices, indices] = False
    selected[:, indices, indices] = False
    return IDLData(selected, available, ranges, clearance)


def brute_force_nearest_visible(
    time_index: int,
    source_index: int,
    target_domain: str,
    clear: np.ndarray,
    ranges: np.ndarray,
    definitions: List[SatelliteDefinition],
) -> int:
    eligible = [
        (float(ranges[time_index, source_index, index]), definitions[index].satellite_id, index)
        for index in range(len(definitions))
        if definitions[index].domain_id == target_domain and clear[time_index, source_index, index]
    ]
    return -1 if not eligible else min(eligible, key=lambda item: (item[0], item[1]))[2]
