"""提供窗口边界、资源区间和地面站天线校准的基础约束函数。"""

from .data import angular_separation_deg


INVALID_AGENT = "INVALID_AGENT"
INVALID_TASK = "INVALID_TASK"
TASK_NOT_ARRIVED = "TASK_NOT_ARRIVED"
TASK_EXPIRED = "TASK_EXPIRED"
TASK_COMPLETED = "TASK_COMPLETED"
SOURCE_HAS_NO_DATA = "SOURCE_HAS_NO_DATA"
INVALID_TARGET = "INVALID_TARGET"
TARGET_GROUND_STATION_MISMATCH = "TARGET_GROUND_STATION_MISMATCH"
LINK_NOT_AVAILABLE = "LINK_NOT_AVAILABLE"
START_BEFORE_WINDOW = "START_BEFORE_WINDOW"
START_AT_OR_AFTER_SLOT_END = "START_AT_OR_AFTER_SLOT_END"
END_AFTER_WINDOW = "END_AFTER_WINDOW"
GROUND_STATION_RESOURCE_CONFLICT = "GROUND_STATION_RESOURCE_CONFLICT"
CALIBRATION_TIME_INSUFFICIENT = "CALIBRATION_TIME_INSUFFICIENT"
ZERO_REQUESTED_DATA = "ZERO_REQUESTED_DATA"
ZERO_FEASIBLE_CAPACITY = "ZERO_FEASIBLE_CAPACITY"
COMPOSITE_ACTION_TOO_LARGE = "COMPOSITE_ACTION_TOO_LARGE"
INTER_SATELLITE_OUTGOING_LIMIT_EXCEEDED = (
    "INTER_SATELLITE_OUTGOING_LIMIT_EXCEEDED"
)
SGL_PER_SLOT_LIMIT_EXCEEDED = "SGL_PER_SLOT_LIMIT_EXCEEDED"
TASK_ALREADY_SCHEDULED_THIS_SLOT = "TASK_ALREADY_SCHEDULED_THIS_SLOT"
PHYSICAL_LINK_ALREADY_USED_THIS_SLOT = "PHYSICAL_LINK_ALREADY_USED_THIS_SLOT"
INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED = (
    "INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED"
)
SAME_SLOT_FORWARDING_NOT_ALLOWED = "SAME_SLOT_FORWARDING_NOT_ALLOWED"
DUPLICATE_TASK_IN_COMPOSITE_ACTION = "DUPLICATE_TASK_IN_COMPOSITE_ACTION"
DUPLICATE_TARGET_LINK_IN_COMPOSITE_ACTION = (
    "DUPLICATE_TARGET_LINK_IN_COMPOSITE_ACTION"
)


def intervals_overlap(first_start, first_end, second_start, second_end, tolerance=1e-9):
    """判断两个 [start,end) 资源区间是否重叠；端点相接不算冲突。"""
    return first_start < second_end - tolerance and second_start < first_end - tolerance


def check_start_inside_window(start_s, window, tolerance=1e-9):
    """检查开始时刻满足窗口的左闭右开语义。"""
    return window.start_time_s - tolerance <= start_s < window.end_time_s - tolerance


def required_calibration_time_seconds(previous_pointing, next_pointing, rotation_speed):
    """按ENU三维夹角和天线转速计算校准所需秒数。"""
    return angular_separation_deg(previous_pointing, next_pointing) / rotation_speed


def make_physical_inter_satellite_link_key(first_satellite_id, second_satellite_id):
    """返回星间物理链路的无向键，不修改任何环境状态。

    将两个卫星ID排序可让 ``A→B`` 与 ``B→A`` 映射到同一资源，确保同一
    30秒时隙内不能通过改变方向绕过物理链路唯一性约束。
    """
    return tuple(sorted((first_satellite_id, second_satellite_id)))


def can_reserve_with_capacity(
    existing_intervals,
    new_interval,
    capacity,
    tolerance_s,
):
    """判断加入新区间后是否仍满足并发容量，不修改传入的区间列表。

    参数中的时间单位均为秒，区间统一采用 ``[start,end)``。只扫描与
    ``new_interval`` 相交的已有区间；相同时刻先处理结束事件，因此首尾
    相接不占用同一份容量。``capacity`` 必须是正整数。
    """
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("资源容量必须是正整数")
    if tolerance_s < 0:
        raise ValueError("时间容差不能为负数")
    if new_interval.end_s <= new_interval.start_s + tolerance_s:
        raise ValueError("新区间必须具有正持续时间")

    # 将已有区间裁剪到候选覆盖范围，避免无关区间影响扫描结果。
    relevant = [new_interval]
    for interval in existing_intervals:
        if intervals_overlap(
            interval.start_s,
            interval.end_s,
            new_interval.start_s,
            new_interval.end_s,
            tolerance_s,
        ):
            relevant.append(interval)

    events = []
    for interval in relevant:
        start_s = max(interval.start_s, new_interval.start_s)
        end_s = min(interval.end_s, new_interval.end_s)
        if end_s > start_s + tolerance_s:
            events.append((start_s, 1))
            events.append((end_s, -1))

    active = 0
    # -1排在+1之前，落实左闭右开区间的端点语义。
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        if active > capacity:
            return False
    return True


def maximum_concurrent_usage(intervals):
    """返回区间集合的最大并发数；时间单位为秒且不修改输入。"""
    events = []
    for interval in intervals:
        events.append((interval.start_s, 1))
        events.append((interval.end_s, -1))
    active = 0
    maximum = 0
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        maximum = max(maximum, active)
    return maximum
