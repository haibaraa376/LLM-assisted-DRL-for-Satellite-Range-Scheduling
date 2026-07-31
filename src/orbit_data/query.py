"""Unambiguous UTC [start,end) queries over frozen availability arrays."""

import bisect
from datetime import datetime
import numpy as np


def time_index_for_utc(unix_seconds, utc_unix_s):
    values = np.asarray(unix_seconds, dtype=np.float64)
    index = bisect.bisect_left(values.tolist(), float(utc_unix_s))
    if index == len(values) or values[index] != float(utc_unix_s):
        raise ValueError("UTC instant is not an exact grid point")
    return index


def interval_index_for_utc(unix_seconds, utc_unix_s):
    values = np.asarray(unix_seconds, dtype=np.float64)
    instant = float(utc_unix_s)
    if instant < values[0] or instant >= values[-1]:
        raise ValueError("UTC instant is outside [start,end) availability interval")
    return int(np.searchsorted(values, instant, side="right") - 1)


def is_available_at(available, unix_seconds, utc_unix_s, *indices):
    return bool(np.asarray(available)[(interval_index_for_utc(unix_seconds, utc_unix_s),) + tuple(indices)])


def available_duration_within(available, unix_seconds, start_unix_s, end_unix_s, *indices):
    if end_unix_s < start_unix_s:
        raise ValueError("end must not precede start")
    total = 0.0
    values = np.asarray(unix_seconds, dtype=np.float64)
    matrix = np.asarray(available)
    for index in range(len(values) - 1):
        left, right = values[index], values[index + 1]
        overlap = max(0.0, min(float(end_unix_s), right) - max(float(start_unix_s), left))
        if overlap and matrix[(index,) + tuple(indices)]:
            total += overlap
    return total
