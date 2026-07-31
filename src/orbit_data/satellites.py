"""Satellite constructors and centralized SGP4 unit conversions."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from sgp4.api import Satrec, WGS72, WGS72OLD, WGS84, jday
from skyfield.api import EarthSatellite

from .models import SatelliteDefinition
from .time_grid import parse_utc_z


MINUTES_PER_DAY = 1440.0
SGP4_EPOCH_JD_OFFSET = 2433281.5


def degrees_to_radians(value_deg: float) -> float:
    return math.radians(float(value_deg))


def mean_motion_rev_per_day_to_rad_per_min(value_rev_per_day: float) -> float:
    return float(value_rev_per_day) * 2.0 * math.pi / MINUTES_PER_DAY


def mean_motion_dot_rev_per_day2_to_rad_per_min2(value_rev_per_day2: float) -> float:
    return float(value_rev_per_day2) * 2.0 * math.pi / (MINUTES_PER_DAY ** 2)


def mean_motion_ddot_rev_per_day3_to_rad_per_min3(value_rev_per_day3: float) -> float:
    return float(value_rev_per_day3) * 2.0 * math.pi / (MINUTES_PER_DAY ** 3)


def utc_datetime_to_sgp4_epoch_days(value: datetime) -> float:
    value = value.astimezone(timezone.utc)
    second = value.second + value.microsecond / 1_000_000.0
    jd, fraction = jday(value.year, value.month, value.day, value.hour, value.minute, second)
    return float(jd + fraction - SGP4_EPOCH_JD_OFFSET)


def gravity_model_constant(name: str) -> Any:
    models = {"WGS72": WGS72, "WGS72OLD": WGS72OLD, "WGS84": WGS84}
    try:
        return models[str(name).upper()]
    except KeyError:
        raise ValueError("unsupported SGP4 gravity model: {0}".format(name))


def build_satrec_from_raw_elements(
    definition: SatelliteDefinition,
    gravity_model: str,
    operation_mode: str,
) -> Satrec:
    elements = definition.orbital_elements
    epoch = parse_utc_z(str(elements["epoch_utc"]))
    satrec = Satrec()
    satrec.sgp4init(
        gravity_model_constant(gravity_model),
        "i" if str(operation_mode).lower() == "improved" else "a",
        definition.numeric_satnum,
        utc_datetime_to_sgp4_epoch_days(epoch),
        float(elements.get("bstar", 0.0)),
        mean_motion_dot_rev_per_day2_to_rad_per_min2(float(elements.get("mean_motion_dot_rev_per_day2", 0.0))),
        mean_motion_ddot_rev_per_day3_to_rad_per_min3(float(elements.get("mean_motion_ddot_rev_per_day3", 0.0))),
        float(elements["eccentricity"]),
        degrees_to_radians(float(elements["argument_of_perigee_deg"])),
        degrees_to_radians(float(elements["inclination_deg"])),
        degrees_to_radians(float(elements["mean_anomaly_deg"])),
        mean_motion_rev_per_day_to_rad_per_min(float(elements["mean_motion_rev_per_day"])),
        degrees_to_radians(float(elements["raan_deg"])),
    )
    return satrec


def build_raw_satellites(
    definitions: Iterable[SatelliteDefinition],
    timescale: Any,
    gravity_model: str,
    operation_mode: str,
) -> List[Any]:
    result = []
    for definition in definitions:
        satrec = build_satrec_from_raw_elements(definition, gravity_model, operation_mode)
        satellite = EarthSatellite.from_satrec(satrec, timescale)
        satellite.name = definition.satellite_id
        result.append(satellite)
    return result


def load_tle_satellites(path: Path, timescale: Any) -> List[Any]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("1 "):
            name = "sat-{0}".format(len(result) + 1)
            line1, line2 = lines[index], lines[index + 1]
            index += 2
        else:
            name, line1, line2 = lines[index:index + 3]
            index += 3
        result.append(EarthSatellite(line1, line2, name, timescale))
    return result


def _load_omm_records(path: Path) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    if suffix == ".json":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return list(value if isinstance(value, list) else value.get("records", [value]))
    raise ValueError("OMM input currently supports local CSV or JSON files")


def load_omm_satellites(path: Path, timescale: Any) -> List[Any]:
    return [EarthSatellite.from_omm(timescale, record) for record in _load_omm_records(path)]


def build_satellites(config: Mapping[str, Any], definitions: List[SatelliteDefinition], timescale: Any) -> List[Any]:
    orbit = config["orbit_input"]
    mode = orbit["mode"]
    if mode == "raw_sgp4_elements":
        return build_raw_satellites(definitions, timescale, orbit["sgp4_gravity_model"], orbit["sgp4_operation_mode"])
    if mode == "tle":
        return load_tle_satellites(Path(orbit["source_file"]), timescale)
    if mode == "omm":
        return load_omm_satellites(Path(orbit["source_file"]), timescale)
    raise ValueError("unsupported orbit_input.mode: {0}".format(mode))
