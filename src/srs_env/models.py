"""定义第二天调度环境使用的任务、动作和状态数据类。"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class TaskStatus(str, Enum):
    """任务在环境中的生命周期状态。"""
    NOT_ARRIVED = "not_arrived"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class TaskDefinition:
    """不可变任务参数；数据量单位为Mbit，时间单位为秒。"""
    task_id: str
    source_satellite_id: str
    target_ground_station_id: str
    priority: int
    data_size_mbit: float
    survival_time_s: float
    arrival_time_s: float
    expiration_time_s: float
    nominal_sgl_duration_s: float

    def __post_init__(self):
        """拒绝不符合固定场景和论文任务字段关系的定义。"""
        if not 1 <= self.priority <= 10:
            raise ValueError("任务优先级必须位于1到10")
        if not np.isfinite(self.data_size_mbit) or self.data_size_mbit <= 0:
            raise ValueError("任务数据量必须是正有限数")
        if not 0 <= self.arrival_time_s <= 86400:
            raise ValueError("任务到达时间必须位于24小时范围内")
        if abs(self.expiration_time_s - (self.arrival_time_s + self.survival_time_s)) > 1e-6:
            raise ValueError("任务过期时间必须等于到达时间加生存时间")
        if self.source_satellite_id not in {"os01", "os02", "os03", "os04", "os05"}:
            raise ValueError("任务源卫星必须属于D2观测域")
        if self.target_ground_station_id not in {"gs01", "gs02", "gs03", "gs04"}:
            raise ValueError("任务目标必须是地面站")


@dataclass
class TaskState:
    """任务运行状态，卫星持有量数组顺序与第一天 satellite_ids 一致。"""
    definition: TaskDefinition
    status: TaskStatus
    data_on_satellites_mbit: np.ndarray
    delivered_to_ground_mbit: float = 0.0
    completion_time_s: Optional[float] = None

    @property
    def remaining_data_mbit(self):
        """返回尚未送达目标地面站的数据量。"""
        return float(self.definition.data_size_mbit - self.delivered_to_ground_mbit)

    def total_accounted_data_mbit(self):
        """返回用于数据守恒检查的卫星持有量与已送达量之和。"""
        return float(self.data_on_satellites_mbit.sum() + self.delivered_to_ground_mbit)


@dataclass
class SatelliteState:
    """卫星资源统计；任务数据不在此处重复保存。"""
    satellite_id: str
    domain_id: str
    transmitted_seconds_by_task: Dict[str, float]


@dataclass
class GroundStationState:
    """地面站天线状态，方向向量采用局部ENU坐标。"""
    ground_station_id: str
    antenna_rotation_speed_deg_per_second: float
    last_transmission_end_s: Optional[float] = None
    last_pointing_enu: Optional[np.ndarray] = None


@dataclass(frozen=True)
class TransmissionSubAction:
    """描述一个任务在当前时隙内的一次传输请求。"""

    task_id: str
    target_id: str
    transmission_ratio: float
    start_offset: float

    def __post_init__(self):
        """任务ID和目标ID不能为空字符串。"""
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("传输子动作的task_id不能为空")
        if not isinstance(self.target_id, str) or not self.target_id:
            raise ValueError("传输子动作的target_id不能为空")


@dataclass(frozen=True)
class SatelliteCompositeAction:
    """一颗卫星在一个时隙中提交的复合动作。

    ``transmissions`` 最多包含4个子动作；合法动作最多包含3条星间传输
    和1条SGL。环境会对全部卫星的子动作进行统一排序和全局仲裁。
    """

    transmissions: Tuple[TransmissionSubAction, ...] = ()

    def __post_init__(self):
        """检查复合动作容器、数量和每个子动作的类型。"""
        if not isinstance(self.transmissions, tuple):
            raise ValueError("复合动作transmissions必须是tuple")
        if len(self.transmissions) > 4:
            raise ValueError("复合动作最多包含4个传输子动作")
        if not all(
            isinstance(item, TransmissionSubAction) for item in self.transmissions
        ):
            raise ValueError("复合动作中每一项都必须是TransmissionSubAction")


@dataclass(frozen=True)
class TransmissionRecord:
    """记录一次接受或拒绝的传输尝试，不默认写入磁盘。"""
    source_satellite_id: str
    subaction_index: int
    composite_source_id: str
    target_id: Optional[str]
    task_id: Optional[str]
    link_type: Optional[str]
    requested_ratio: float
    requested_start_s: float
    accepted: bool
    transmitted_data_mbit: float
    actual_start_s: Optional[float]
    actual_end_s: Optional[float]
    rate_mbps: Optional[float]
    violation_codes: Tuple[str, ...]
    projected: bool


@dataclass(frozen=True)
class ReservedInterval:
    """当前时隙内占用一个卫星或地面站资源的 [start,end) 区间。"""
    start_s: float
    end_s: float
    owner_id: str
