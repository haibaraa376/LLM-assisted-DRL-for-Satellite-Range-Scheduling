"""加载第一天冻结的轨道与链路数据，并提供窗口查询接口。

本模块只读取NPZ、CSV和JSON，不调用Skyfield传播轨道。时间统一使用相对
仿真起点的秒数；窗口采用 ``[start, end)`` 语义。
"""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from orbit_data.pointing import az_el_to_enu_unit_vector


@dataclass(frozen=True)
class TransmissionWindow:
    """一条链路的连续可用窗口，时间单位为秒、速率单位为Mbps。"""

    window_id: str
    link_type: str
    source_id: str
    target_id: str
    start_time_s: float
    end_time_s: float
    rate_mbps: float
    start_index: Optional[int] = None
    end_index_exclusive: Optional[int] = None


class WindowIndex:
    """将三类CSV窗口组织为可快速查询的内存索引。"""

    def __init__(self, windows: List[TransmissionWindow]):
        self.all_windows = tuple(windows)
        self.items = {}  # type: Dict[Tuple[str, str, str], List[TransmissionWindow]]
        for window in windows:
            key = self._key(window.link_type, window.source_id, window.target_id)
            self.items.setdefault(key, []).append(window)
        for values in self.items.values():
            values.sort(key=lambda item: (item.start_time_s, item.end_time_s, item.window_id))

    @staticmethod
    def _key(link_type: str, source_id: str, target_id: str):
        """SGL保留方向；ISL和IDL用排序端点作为无向窗口键。"""
        if link_type == "SGL":
            return link_type, source_id, target_id
        first, second = sorted((source_id, target_id))
        return link_type, first, second

    def find_active_window(self, link_type, source_id, target_id, time_s):
        """返回满足 ``start <= time < end`` 的活动窗口，否则返回None。"""
        key = self._key(link_type, source_id, target_id)
        return next(
            (
                window
                for window in self.items.get(key, ())
                if window.start_time_s <= time_s < window.end_time_s
            ),
            None,
        )

    def find_next_window(self, link_type, source_id, target_id, time_s):
        """返回当前或未来第一个尚未结束的窗口。"""
        key = self._key(link_type, source_id, target_id)
        return next(
            (window for window in self.items.get(key, ()) if window.end_time_s > time_s),
            None,
        )

    def find_overlapping_window(
        self,
        link_type,
        source_id,
        target_id,
        interval_start_s,
        interval_end_s,
    ):
        """返回与给定 ``[start,end)`` 区间存在非空交集的第一个窗口。"""
        key = self._key(link_type, source_id, target_id)
        return next(
            (
                window
                for window in self.items.get(key, ())
                if max(interval_start_s, window.start_time_s)
                < min(interval_end_s, window.end_time_s)
            ),
            None,
        )

    def get_link_rate_mbps(self, link_type, source_id, target_id):
        """返回指定链路的固定速率；没有窗口时返回None。"""
        key = self._key(link_type, source_id, target_id)
        windows = self.items.get(key, ())
        return windows[0].rate_mbps if windows else None


@dataclass
class SkyfieldDataset:
    """调度环境所需的第一天只读数据集合。"""

    timestamps_unix_s: np.ndarray
    satellite_ids: Tuple[str, ...]
    ground_station_ids: Tuple[str, ...]
    satellite_domain_ids: Tuple[str, ...]
    satellite_domain_names: Tuple[str, ...]
    sgl_available: np.ndarray
    sgl_elevation_deg: np.ndarray
    sgl_azimuth_deg: np.ndarray
    sgl_range_km: np.ndarray
    isl_available: np.ndarray
    isl_range_km: np.ndarray
    idl_available: np.ndarray
    idl_selected_directed: np.ndarray
    idl_range_km: np.ndarray
    satellite_index: Dict[str, int]
    ground_station_index: Dict[str, int]
    antenna_rotation_speed_by_station: Dict[str, float]
    rates_mbps: Dict[str, float]
    windows: WindowIndex

    @property
    def step_count(self):
        """2881个状态点对应2880个30秒决策区间。"""
        return len(self.timestamps_unix_s) - 1

    def get_link_type(self, source_id, target_id):
        """根据端点类型和业务域确定SGL、ISL或IDL。"""
        if source_id not in self.satellite_index:
            raise ValueError("未知源卫星：{0}".format(source_id))
        if target_id in self.ground_station_index:
            return "SGL"
        if target_id not in self.satellite_index:
            raise ValueError("未知传输目标：{0}".format(target_id))
        source = self.satellite_index[source_id]
        target = self.satellite_index[target_id]
        return "ISL" if self.satellite_domain_ids[source] == self.satellite_domain_ids[target] else "IDL"

    def is_available_at_step(self, source_id, target_id, step_index):
        """查询一个离散状态点的链路可用性；终点不能用于开始传输。"""
        if not 0 <= step_index < self.step_count:
            return False
        link_type = self.get_link_type(source_id, target_id)
        source = self.satellite_index[source_id]
        if link_type == "SGL":
            return bool(self.sgl_available[step_index, source, self.ground_station_index[target_id]])
        matrix = self.isl_available if link_type == "ISL" else self.idl_available
        return bool(matrix[step_index, source, self.satellite_index[target_id]])

    def get_sgl_pointing_vector(self, satellite_id, ground_station_id, time_s):
        """返回任意时刻地面站局部ENU坐标中的单位指向向量。

        先将两个端点的方位角/仰角转成单位向量再插值，避免359度到1度
        直接插值时沿错误方向绕行。终点86400秒可以查询但不能发起传输。
        """
        if not 0.0 <= time_s <= 86400.0:
            raise ValueError("SGL指向查询时间必须位于[0,86400]秒")
        satellite = self.satellite_index[satellite_id]
        station = self.ground_station_index[ground_station_id]
        if time_s == 86400.0:
            index, fraction = 2879, 1.0
        else:
            index = int(time_s // 30.0)
            fraction = (time_s - index * 30.0) / 30.0
        first = az_el_to_enu_unit_vector(
            self.sgl_azimuth_deg[index, satellite, station],
            self.sgl_elevation_deg[index, satellite, station],
        )
        second = az_el_to_enu_unit_vector(
            self.sgl_azimuth_deg[index + 1, satellite, station],
            self.sgl_elevation_deg[index + 1, satellite, station],
        )
        vector = (1.0 - fraction) * first + fraction * second
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            raise ValueError("ENU指向向量插值结果无法归一化")
        return vector / norm


def angular_separation_deg(first_vector, second_vector):
    """返回两个三维单位向量的夹角，单位为度。"""
    first = np.asarray(first_vector, dtype=float)
    second = np.asarray(second_vector, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0.0:
        raise ValueError("计算夹角的向量长度必须大于0")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _parse_utc_timestamp(value):
    """解析带时区的ISO-8601 UTC时间，保留小数秒。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("窗口UTC时间缺少时区：{0}".format(value))
    return parsed


def _load_windows(root, simulation_start_utc):
    """从三个CSV读取连续窗口，并换算为相对仿真起点的秒数。

    ``start_index`` 和 ``end_index_exclusive`` 仅保留为离散索引元数据；
    真实起止时间始终来自CSV的UTC字段，避免SGL事件边界被量化到30秒。
    """
    windows = []
    start_utc = _parse_utc_timestamp(simulation_start_utc)
    for filename in ("sgl_windows.csv", "isl_windows.csv", "idl_windows.csv"):
        with (root / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                start_time_s = (
                    _parse_utc_timestamp(row["start_time_utc"]) - start_utc
                ).total_seconds()
                end_time_s = (
                    _parse_utc_timestamp(row["end_time_utc"]) - start_utc
                ).total_seconds()
                # 浮点时间转换不得让窗口越过24小时仿真终点。
                end_time_s = min(end_time_s, 86400.0)
                if not 0.0 <= start_time_s < end_time_s <= 86400.0:
                    raise ValueError("窗口时间超出24小时范围：{0}".format(row["window_id"]))
                windows.append(
                    TransmissionWindow(
                        window_id=row["window_id"],
                        link_type=row["link_type"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        start_time_s=start_time_s,
                        end_time_s=end_time_s,
                        rate_mbps=float(row["transmission_rate_mbps"]),
                        start_index=int(row["start_index"]),
                        end_index_exclusive=int(row["end_index_exclusive"]),
                    )
                )
    return WindowIndex(windows)


def load_skyfield_dataset(data_root=Path("data/skyfield")):
    """加载并校验第一天数据，返回调度环境使用的只读数据对象。"""
    root = Path(data_root)
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise ValueError("第一天Skyfield数据校验未通过")

    with np.load(root / "satellite_positions.npz") as positions, np.load(
        root / "sgl_availability.npz"
    ) as sgl, np.load(root / "isl_availability.npz") as isl, np.load(
        root / "idl_availability.npz"
    ) as idl:
        satellite_ids = tuple(positions["satellite_ids"].astype(str))
        station_ids = tuple(sgl["ground_station_ids"].astype(str))
        timestamps = positions["timestamps_unix_s"].copy()
        if len(satellite_ids) != 15 or len(station_ids) != 4 or len(timestamps) != 2881:
            raise ValueError("第一天数据规模必须为15星、4站和2881个时间点")
        if not np.all(np.diff(timestamps) == 30.0):
            raise ValueError("第一天时间轴必须严格按30秒递增")
        for link_data, name in ((isl, "ISL"), (idl, "IDL")):
            if satellite_ids != tuple(link_data["satellite_ids"].astype(str)):
                raise ValueError("{0}卫星ID顺序与轨道数据不一致".format(name))
            if not np.array_equal(timestamps, link_data["timestamps_unix_s"]):
                raise ValueError("{0}时间轴与轨道数据不一致".format(name))
        numeric_arrays = (
            sgl["elevation_deg"],
            sgl["azimuth_deg"],
            sgl["range_km"],
            isl["range_km"],
            idl["range_km"],
        )
        if not all(np.all(np.isfinite(array)) for array in numeric_arrays):
            raise ValueError("第一天链路数组包含NaN或Inf")
        if not np.array_equal(isl["available"], isl["available"].transpose(0, 2, 1)):
            raise ValueError("ISL可用矩阵必须对称")
        expected_union = idl["selected_directed"] | idl["selected_directed"].transpose(0, 2, 1)
        if not np.array_equal(idl["available"], expected_union):
            raise ValueError("IDL主矩阵不符合union对称化")

        # 固定随机抽查有向选择：目标必须来自其他域，并是该目标域内最近的无遮挡卫星。
        domains = isl["satellite_domain_ids"].astype(str)
        selected_indices = np.argwhere(idl["selected_directed"])
        rng = np.random.RandomState(2025)
        if len(selected_indices):
            chosen = selected_indices[
                rng.choice(len(selected_indices), size=min(20, len(selected_indices)), replace=False)
            ]
            clearance = idl["earth_clearance_km"]
            ranges = idl["range_km"]
            for time_index, source, target in chosen:
                if domains[source] == domains[target]:
                    raise ValueError("IDL有向选择错误连接了同一业务域")
                candidates = [
                    index
                    for index, domain in enumerate(domains)
                    if domain == domains[target]
                    and clearance[time_index, source, index] > 0.0
                ]
                if candidates and ranges[time_index, source, target] > min(
                    ranges[time_index, source, index] for index in candidates
                ) + 1e-3:
                    raise ValueError("IDL抽查发现目标不是其他域内最近可见卫星")
        station_records = json.loads((root / "ground_stations.json").read_text(encoding="utf-8"))
        rotation_speeds = {
            item["id"]: float(item["antenna_rotation_speed_deg_per_second"])
            for item in station_records
        }
        # 显式使用UTC，避免运行机器的本地时区影响相对秒数。
        simulation_start_utc = datetime.fromtimestamp(
            float(timestamps[0]), timezone.utc
        ).isoformat()
        window_index = _load_windows(root, simulation_start_utc)
        rates = {
            link_type: window_index.get_link_rate_mbps(link_type, source, target)
            for link_type, source, target in (
                ("SGL", "cs01", "gs01"),
                ("ISL", "cs01", "cs02"),
                ("IDL", "cs01", "os01"),
            )
        }
        if rates != {"SGL": 60.0, "ISL": 80.0, "IDL": 80.0}:
            raise ValueError("固定链路速率必须为SGL=60、ISL/IDL=80 Mbps")
        return SkyfieldDataset(
            timestamps_unix_s=timestamps,
            satellite_ids=satellite_ids,
            ground_station_ids=station_ids,
            satellite_domain_ids=tuple(isl["satellite_domain_ids"].astype(str)),
            satellite_domain_names=tuple(isl["satellite_domain_names"].astype(str)),
            sgl_available=sgl["available"].copy(),
            sgl_elevation_deg=sgl["elevation_deg"].copy(),
            sgl_azimuth_deg=sgl["azimuth_deg"].copy(),
            sgl_range_km=sgl["range_km"].copy(),
            isl_available=isl["available"].copy(),
            isl_range_km=isl["range_km"].copy(),
            idl_available=idl["available"].copy(),
            idl_selected_directed=idl["selected_directed"].copy(),
            idl_range_km=idl["range_km"].copy(),
            satellite_index={item: index for index, item in enumerate(satellite_ids)},
            ground_station_index={item: index for index, item in enumerate(station_ids)},
            antenna_rotation_speed_by_station=rotation_speeds,
            rates_mbps=rates,
            windows=window_index,
        )
