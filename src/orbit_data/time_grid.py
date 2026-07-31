"""UTC-only deterministic time-grid construction."""

from datetime import datetime, timedelta, timezone
from typing import Any, List

import numpy as np

from .models import TimeGrid


def parse_utc_z(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("UTC timestamp must end with Z: {0}".format(value))
    result = datetime.fromisoformat(value[:-1] + "+00:00")
    if result.utcoffset() != timedelta(0):
        raise ValueError("timestamp is not UTC: {0}".format(value))
    return result.astimezone(timezone.utc)


def isoformat_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_time_grid(
    timescale: Any,
    start_utc_text: str,
    duration_seconds: float,
    step_seconds: float,
    include_final_endpoint: bool,
) -> TimeGrid:
    if duration_seconds <= 0 or step_seconds <= 0:
        raise ValueError("duration_seconds and step_seconds must be positive")
    start = parse_utc_z(start_utc_text)
    end = start + timedelta(seconds=float(duration_seconds))
    offsets = np.arange(0.0, float(duration_seconds), float(step_seconds), dtype=np.float64)
    if include_final_endpoint and (len(offsets) == 0 or offsets[-1] != float(duration_seconds)):
        offsets = np.append(offsets, float(duration_seconds))
    datetimes = [start + timedelta(seconds=float(offset)) for offset in offsets]  # type: List[datetime]
    unix_seconds = np.asarray([item.timestamp() for item in datetimes], dtype=np.float64)
    if len(unix_seconds) > 1 and not np.all(np.diff(unix_seconds) > 0):
        raise ValueError("time grid is not strictly monotonic")
    iso_utc = np.asarray([isoformat_utc_z(item) for item in datetimes])
    return TimeGrid(start, end, float(step_seconds), include_final_endpoint, unix_seconds, iso_utc, timescale.from_datetimes(datetimes))
