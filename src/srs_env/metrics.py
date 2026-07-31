"""实现论文及时性指标与修正后的平均U负载均衡指标。"""

import numpy as np


def compute_total_window_seconds_by_satellite(dataset):
    """按CSV中每条链路窗口分别求和，返回15颗卫星的总窗口秒数。

    不对时间重叠窗口做并集去重，因为论文公式12按目标链路分别累加。
    """
    totals = np.zeros(len(dataset.satellite_ids), dtype=float)
    for window in dataset.windows.all_windows:
        duration = window.end_time_s - window.start_time_s
        if window.source_id in dataset.satellite_index:
            totals[dataset.satellite_index[window.source_id]] += duration
        if window.link_type != "SGL" and window.target_id in dataset.satellite_index:
            totals[dataset.satellite_index[window.target_id]] += duration
    return totals


def timeliness_contribution(task, transmitted_data_mbit, actual_start_s, tolerance=1e-9):
    """计算一次成功传输对及时性的贡献，数据量单位为Mbit、时间单位为秒。"""
    if transmitted_data_mbit <= tolerance:
        return 0.0
    weight = (task.expiration_time_s - actual_start_s) / task.survival_time_s
    if weight < -tolerance or weight > 1.0 + tolerance:
        raise ValueError("及时性开始时间超出任务生命周期")
    return transmitted_data_mbit / task.data_size_mbit * task.priority * float(np.clip(weight, 0.0, 1.0))


def load_balance(outgoing_seconds, total_window_seconds, b_max=1.0):
    """按修正公式11返回总负载均衡值和任务均值。

    这里的平均项是平均U，不是论文印刷错误中的 mean(1-U)。
    """
    outgoing_seconds = np.asarray(outgoing_seconds, dtype=float)
    total_window_seconds = np.asarray(total_window_seconds, dtype=float)
    if outgoing_seconds.ndim != 2 or outgoing_seconds.shape[1] != len(total_window_seconds):
        raise ValueError("传输时长矩阵与卫星窗口总时长的形状不一致")
    if outgoing_seconds.shape[0] == 0:
        return 0.0, 0.0, 0.0
    utilization = np.divide(
        outgoing_seconds,
        total_window_seconds[None, :],
        out=np.zeros_like(outgoing_seconds),
        where=total_window_seconds[None, :] > 0,
    )
    mean_utilization = utilization.mean(axis=1, keepdims=True)
    standard_deviation = np.sqrt(
        np.mean((utilization - mean_utilization) ** 2, axis=1)
    )
    scores = b_max - standard_deviation
    return float(scores.sum()), float(scores.mean()), float(standard_deviation.mean())
