"""Deterministic boolean-series window merging and transition refinement."""

from typing import Callable, List, Tuple

import numpy as np

from .models import WindowSpan


def merge_boolean_series_to_windows(
    available: np.ndarray,
    timestamps_unix_s: np.ndarray,
    minimum_duration_seconds: float = 0.0,
    merge_consecutive_true_samples: bool = True,
    bridge_false_gaps_seconds: float = 0.0,
) -> List[WindowSpan]:
    values = np.asarray(available, dtype=bool)
    times = np.asarray(timestamps_unix_s, dtype=np.float64)
    if values.ndim != 1 or times.ndim != 1 or len(values) != len(times):
        raise ValueError("available and timestamps must be same-length 1-D arrays")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("timestamps must be strictly increasing")
    if not merge_consecutive_true_samples:
        values = values.copy()
        values[1:] &= ~values[:-1]
    if bridge_false_gaps_seconds > 0.0:
        values = values.copy()
        false_indices = np.flatnonzero(~values)
        for index in false_indices:
            if 0 < index < len(values) - 1 and values[index - 1] and values[index + 1] and times[index + 1] - times[index - 1] <= bridge_false_gaps_seconds:
                values[index] = True
    windows = []  # type: List[WindowSpan]
    start = None
    for index, is_available in enumerate(values):
        if is_available and start is None:
            start = index
        if not is_available and start is not None:
            end = index
            if times[end] - times[start] >= minimum_duration_seconds:
                windows.append(WindowSpan(start, end, float(times[start]), float(times[end])))
            start = None
    if start is not None and len(times) > 1:
        end = len(times)
        if times[-1] - times[start] >= minimum_duration_seconds:
            windows.append(WindowSpan(start, end, float(times[start]), float(times[-1])))
    return windows


def refine_boolean_transition(
    predicate: Callable[[float], bool],
    left_unix_s: float,
    right_unix_s: float,
    tolerance_seconds: float,
) -> float:
    left = float(left_unix_s)
    right = float(right_unix_s)
    left_value = bool(predicate(left))
    right_value = bool(predicate(right))
    if left_value == right_value:
        raise ValueError("transition endpoints must have different boolean states")
    while right - left > float(tolerance_seconds):
        middle = (left + right) / 2.0
        if bool(predicate(middle)) == left_value:
            left = middle
        else:
            right = middle
    return (left + right) / 2.0


def reconstruct_boolean_series_from_windows(length: int, windows: List[WindowSpan]) -> np.ndarray:
    result = np.zeros(int(length), dtype=bool)
    for window in windows:
        if not (0 <= window.start_index <= window.end_index_exclusive <= length):
            raise ValueError("window indices outside series")
        result[window.start_index:window.end_index_exclusive] = True
    return result


def match_refined_event_window(
    sampled_window: WindowSpan,
    event_windows: List[Tuple[float, float]],
) -> WindowSpan:
    best = None
    best_overlap = -1.0
    for start, end in event_windows:
        overlap = max(0.0, min(sampled_window.end_unix_s, end) - max(sampled_window.start_unix_s, start))
        if overlap > best_overlap:
            best = (start, end)
            best_overlap = overlap
    if best is None or best_overlap <= 0.0:
        return sampled_window
    return WindowSpan(
        sampled_window.start_index,
        sampled_window.end_index_exclusive,
        float(best[0]),
        float(best[1]),
        "skyfield_find_events",
    )
