"""Configured intra-domain topology and time-varying ISL availability."""

from typing import List, Tuple

import numpy as np

from .geometry import segments_clear_earth
from .models import ISLData, SatelliteDefinition


def build_candidate_topology(definitions: List[SatelliteDefinition]) -> np.ndarray:
    ids = [item.satellite_id for item in definitions]
    index = {satellite_id: i for i, satellite_id in enumerate(ids)}
    topology = np.zeros((len(ids), len(ids)), dtype=bool)
    for source in definitions:
        source_index = index[source.satellite_id]
        for target_id in source.isl_neighbor_ids:
            target_index = index[target_id]
            topology[source_index, target_index] = True
    np.fill_diagonal(topology, False)
    if not np.array_equal(topology, topology.T):
        raise ValueError("configured undirected ISL topology is not symmetric")
    return topology


def canonical_candidate_pairs(topology: np.ndarray) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(topology.shape[0]) for j in range(i + 1, topology.shape[1]) if topology[i, j]]


def compute_isl(
    position_gcrs_km: np.ndarray,
    definitions: List[SatelliteDefinition],
    earth_radius_km: float,
    safety_margin_km: float,
    tangent_is_blocked: bool,
    maximum_range_km: float = None,
) -> ISLData:
    positions = np.asarray(position_gcrs_km, dtype=np.float64)
    time_count, satellite_count, _ = positions.shape
    topology = build_candidate_topology(definitions)
    clear = np.zeros((time_count, satellite_count, satellite_count), dtype=bool)
    available = np.zeros_like(clear)
    ranges = np.zeros((time_count, satellite_count, satellite_count), dtype=np.float32)
    clearance = np.zeros_like(ranges)
    for source_index, target_index in canonical_candidate_pairs(topology):
        source = positions[:, source_index, :]
        target = positions[:, target_index, :]
        pair_clear, pair_clearance = segments_clear_earth(
            source, target, earth_radius_km, safety_margin_km, tangent_is_blocked
        )
        pair_range = np.linalg.norm(target - source, axis=-1)
        pair_available = pair_clear.copy()
        if maximum_range_km is not None:
            pair_available &= pair_range <= float(maximum_range_km)
        for matrix, values in ((clear, pair_clear), (available, pair_available), (ranges, pair_range), (clearance, pair_clearance)):
            matrix[:, source_index, target_index] = values
            matrix[:, target_index, source_index] = values
    return ISLData(topology, clear, available, ranges, clearance)
