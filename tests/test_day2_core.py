"""验证第二天任务、窗口、指向和论文指标的核心语义。"""

import json

import numpy as np
import pytest

from orbit_data.pointing import az_el_to_enu_unit_vector
from srs_env.config import load_task_config
from srs_env.data import (
    TransmissionWindow,
    WindowIndex,
    _load_windows,
    angular_separation_deg,
    load_skyfield_dataset,
)
from srs_env.constraints import can_reserve_with_capacity
from srs_env.metrics import load_balance, timeliness_contribution
from srs_env.models import (
    ReservedInterval,
    SatelliteCompositeAction,
    TaskDefinition,
    TransmissionSubAction,
)
from srs_env.tasks import generate_task_database, survival_time, write_task_database


def test_survival_time_mapping():
    """优先级1—10必须映射到固定T1—T4。"""
    config = load_task_config()
    assert [survival_time(priority, config) for priority in (1, 4, 7, 10)] == [21600, 10800, 3600, 1800]


def test_task_database_is_reproducible():
    """相同配置和种子必须生成完全相同的任务定义。"""
    config = load_task_config()
    dataset = load_skyfield_dataset()
    assert generate_task_database(config, dataset) == generate_task_database(config, dataset)


def test_task_splits_are_disjoint_and_complete(tmp_path):
    """70/15/15划分必须互斥且覆盖全部1000项任务。"""
    config = load_task_config()
    tasks = generate_task_database(config, load_skyfield_dataset())
    splits = write_task_database(tasks, config, tmp_path)
    sets = {name: set(values) for name, values in splits.items()}
    assert {name: len(values) for name, values in sets.items()} == {"train": 700, "validation": 150, "test": 150}
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"])
    assert set.union(*sets.values()) == {task.task_id for task in tasks}
    # JSONL不允许NaN，并且能够逐行重新读取。
    assert len((tmp_path / "task_database.jsonl").read_text(encoding="utf-8").splitlines()) == 1000
    json.loads((tmp_path / "task_database.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_window_uses_left_closed_right_open_interval():
    """窗口起点可用，终点不可用。"""
    window = TransmissionWindow("w1", "SGL", "os01", "gs01", 10.0, 20.0, 60.0)
    index = WindowIndex([window])
    assert index.find_active_window("SGL", "os01", "gs01", 10.0) == window
    assert index.find_active_window("SGL", "os01", "gs01", 20.0) is None


def test_continuous_window_times_preserve_fractional_seconds(tmp_path):
    """CSV中的连续UTC边界不得被量化到30秒网格。"""
    header = (
        "window_id,link_type,source_id,target_id,start_index,"
        "end_index_exclusive,start_time_utc,end_time_utc,transmission_rate_mbps\n"
    )
    rows = {
        "sgl_windows.csv": "w-sgl,SGL,os01,gs01,1501,1502,2025-01-01T12:30:33.715757Z,2025-01-01T12:30:44.250000Z,60\n",
        "isl_windows.csv": "w-isl,ISL,os01,os02,0,2881,2025-01-01T00:00:00Z,2025-01-02T00:00:00Z,80\n",
        "idl_windows.csv": "w-idl,IDL,os01,cs01,0,2881,2025-01-01T00:00:00Z,2025-01-02T00:00:00Z,80\n",
    }
    for filename, row in rows.items():
        (tmp_path / filename).write_text(header + row, encoding="utf-8")
    windows = _load_windows(tmp_path, "2025-01-01T00:00:00+00:00")
    sgl = windows.find_active_window("SGL", "os01", "gs01", 45034.0)
    assert np.isclose(sgl.start_time_s, 45033.715757)
    assert np.isclose(sgl.end_time_s, 45044.25)
    assert sgl.start_time_s % 30.0 != 0.0


def test_full_day_window_ends_at_86400_seconds(tmp_path):
    """即使end_index_exclusive为2881，真实UTC终点仍必须是86400秒。"""
    header = (
        "window_id,link_type,source_id,target_id,start_index,"
        "end_index_exclusive,start_time_utc,end_time_utc,transmission_rate_mbps\n"
    )
    row = "w-{0},{0},os01,{1},0,2881,2025-01-01T00:00:00Z,2025-01-02T00:00:00Z,{2}\n"
    (tmp_path / "sgl_windows.csv").write_text(header + row.format("SGL", "gs01", 60), encoding="utf-8")
    (tmp_path / "isl_windows.csv").write_text(header + row.format("ISL", "os02", 80), encoding="utf-8")
    (tmp_path / "idl_windows.csv").write_text(header + row.format("IDL", "cs01", 80), encoding="utf-8")
    windows = _load_windows(tmp_path, "2025-01-01T00:00:00+00:00")
    assert all(window.end_time_s == 86400.0 for window in windows.all_windows)


def test_pointing_vector_interpolation_avoids_azimuth_wraparound():
    """359度与1度先转ENU向量后插值，方向应接近0度而非180度。"""
    first = az_el_to_enu_unit_vector(359.0, 20.0)
    second = az_el_to_enu_unit_vector(1.0, 20.0)
    middle = first + second
    middle /= np.linalg.norm(middle)
    expected = az_el_to_enu_unit_vector(0.0, 20.0)
    assert angular_separation_deg(middle, expected) < 0.01


def test_timeliness_matches_manual_example():
    """公式9—10的手工示例贡献应为4.0。"""
    task = TaskDefinition("task", "os01", "gs01", 10, 100.0, 100.0, 0.0, 100.0, 100.0 / 60.0)
    assert timeliness_contribution(task, 50.0, 20.0) == 4.0


def test_corrected_load_balance_uses_mean_utilization():
    """均匀利用率得满分；偏斜利用率降低；卫星置换不改变结果。"""
    windows = np.full(15, 100.0)
    equal = np.full((1, 15), 10.0)
    skewed = np.zeros((1, 15))
    skewed[0, 0] = 100.0
    equal_score, _, _ = load_balance(equal, windows)
    skewed_score, _, _ = load_balance(skewed, windows)
    permuted_score, _, _ = load_balance(skewed[:, ::-1], windows)
    assert equal_score == 1.0
    assert skewed_score < equal_score
    assert np.isclose(skewed_score, permuted_score)


def test_composite_action_size_and_continuous_projection_contract():
    """复合动作最多4项，连续值由环境负责投影而非数据类静默修改。"""
    transmissions = tuple(
        TransmissionSubAction("t{0}".format(index), "cs01", 2.0, -1.0)
        for index in range(4)
    )
    assert len(SatelliteCompositeAction(transmissions).transmissions) == 4
    with pytest.raises(ValueError, match="最多包含4个"):
        SatelliteCompositeAction(
            transmissions
            + (TransmissionSubAction("t4", "cs01", 1.0, 0.0),)
        )


def test_capacity_reservation_uses_left_closed_right_open_intervals():
    """容量3允许三重叠、拒绝第四重叠，并允许首尾相接和非重叠区间。"""
    existing = [
        ReservedInterval(0.0, 10.0, "first"),
        ReservedInterval(0.0, 10.0, "second"),
    ]
    third = ReservedInterval(0.0, 10.0, "third")
    assert can_reserve_with_capacity(existing, third, 3, 1.0e-9)
    assert not can_reserve_with_capacity(existing + [third], third, 3, 1.0e-9)
    assert can_reserve_with_capacity(
        existing + [third],
        ReservedInterval(10.0, 20.0, "touching"),
        3,
        1.0e-9,
    )
    assert can_reserve_with_capacity(
        existing,
        ReservedInterval(20.0, 30.0, "separate"),
        3,
        1.0e-9,
    )
