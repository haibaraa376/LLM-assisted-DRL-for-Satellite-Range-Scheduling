"""用小型确定性数据验证第二天环境及12个复合并发场景。"""

import numpy as np
import pytest

from srs_env.config import load_environment_config
from srs_env.constraints import (
    CALIBRATION_TIME_INSUFFICIENT,
    DUPLICATE_TARGET_LINK_IN_COMPOSITE_ACTION,
    INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED,
    INTER_SATELLITE_OUTGOING_LIMIT_EXCEEDED,
    PHYSICAL_LINK_ALREADY_USED_THIS_SLOT,
    SAME_SLOT_FORWARDING_NOT_ALLOWED,
    SGL_PER_SLOT_LIMIT_EXCEEDED,
    TASK_ALREADY_SCHEDULED_THIS_SLOT,
    TASK_EXPIRED,
    TASK_NOT_ARRIVED,
)
from srs_env.data import SkyfieldDataset, TransmissionWindow, WindowIndex
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.metrics import timeliness_contribution
from srs_env.models import (
    SatelliteCompositeAction,
    TaskDefinition,
    TaskStatus,
    TransmissionSubAction,
)


SATELLITES = tuple(
    ["cs{:02d}".format(index) for index in range(1, 6)]
    + ["os{:02d}".format(index) for index in range(1, 6)]
    + ["ns{:02d}".format(index) for index in range(1, 6)]
)
DOMAINS = tuple(["D1"] * 5 + ["D2"] * 5 + ["D3"] * 5)
STATIONS = ("gs01", "gs02", "gs03", "gs04")


def make_dataset():
    """构造4个时隙全可见数据，便于单独观察资源仲裁结果。"""
    time_count = 5
    satellite_count = len(SATELLITES)
    station_count = len(STATIONS)
    sgl = np.ones((time_count, satellite_count, station_count), dtype=bool)
    isl = np.zeros((time_count, satellite_count, satellite_count), dtype=bool)
    idl = np.zeros_like(isl)
    windows = []
    for source_index, source in enumerate(SATELLITES):
        for station in STATIONS:
            windows.append(
                TransmissionWindow(
                    "SGL-{0}-{1}".format(source, station),
                    "SGL",
                    source,
                    station,
                    0.0,
                    120.0,
                    60.0,
                )
            )
        for target_index in range(source_index + 1, satellite_count):
            target = SATELLITES[target_index]
            link_type = (
                "ISL"
                if DOMAINS[source_index] == DOMAINS[target_index]
                else "IDL"
            )
            matrix = isl if link_type == "ISL" else idl
            matrix[:, source_index, target_index] = True
            matrix[:, target_index, source_index] = True
            windows.append(
                TransmissionWindow(
                    "{0}-{1}-{2}".format(link_type, source, target),
                    link_type,
                    source,
                    target,
                    0.0,
                    120.0,
                    80.0,
                )
            )
    azimuth = np.zeros(
        (time_count, satellite_count, station_count),
        dtype=np.float32,
    )
    azimuth[:, SATELLITES.index("os02"), :] = 180.0
    elevation = np.full_like(azimuth, 20.0)
    return SkyfieldDataset(
        timestamps_unix_s=np.arange(time_count, dtype=float) * 30.0,
        satellite_ids=SATELLITES,
        ground_station_ids=STATIONS,
        satellite_domain_ids=DOMAINS,
        satellite_domain_names=tuple(
            {"D1": "communication", "D2": "observation", "D3": "navigation"}[
                value
            ]
            for value in DOMAINS
        ),
        sgl_available=sgl,
        sgl_elevation_deg=elevation,
        sgl_azimuth_deg=azimuth,
        sgl_range_km=np.ones_like(elevation),
        isl_available=isl,
        isl_range_km=np.ones_like(isl, dtype=np.float32),
        idl_available=idl,
        idl_selected_directed=idl.copy(),
        idl_range_km=np.ones_like(idl, dtype=np.float32),
        satellite_index={value: index for index, value in enumerate(SATELLITES)},
        ground_station_index={value: index for index, value in enumerate(STATIONS)},
        antenna_rotation_speed_by_station={value: 3.0 for value in STATIONS},
        rates_mbps={"SGL": 60.0, "ISL": 80.0, "IDL": 80.0},
        windows=WindowIndex(windows),
    )


def task(
    task_id="task",
    source="os01",
    station="gs01",
    size=600.0,
    arrival=0.0,
    lifetime=120.0,
    priority=10,
):
    """创建符合固定D2初始源规则的测试任务。"""
    return TaskDefinition(
        task_id,
        source,
        station,
        priority,
        size,
        lifetime,
        arrival,
        arrival + lifetime,
        size / 60.0,
    )


def make_env(tasks):
    """创建并重置确定性测试环境。"""
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        make_dataset(),
        load_environment_config(),
        tasks,
    )
    environment.reset()
    return environment


def sub(task_id, target_id, ratio=1.0, offset=0.0):
    """创建一个传输子动作。"""
    return TransmissionSubAction(task_id, target_id, ratio, offset)


def composite(*transmissions):
    """创建复合动作；空参数即为空闲。"""
    return SatelliteCompositeAction(tuple(transmissions))


def set_holdings(environment, task_id, holdings):
    """将测试任务数据放到指定卫星，同时保持任务总数据量不变。"""
    state = environment.tasks[task_id]
    state.data_on_satellites_mbit.fill(0.0)
    for satellite_id, amount in holdings.items():
        state.data_on_satellites_mbit[
            environment.dataset.satellite_index[satellite_id]
        ] = amount


def codes(info):
    """返回本时隙所有拒绝记录的违反代码集合。"""
    return {
        code
        for record in info["transmission_records"]
        if not record.accepted
        for code in record.violation_codes
    }


def test_empty_composite_action_is_idle():
    """空元组表示空闲且不生成传输记录。"""
    environment = make_env([task()])
    _, _, _, _, info = environment.step({"os01": composite()})
    assert info["submitted_subaction_count"] == 0
    assert info["transmission_records"] == []


def test_sgl_complete_and_partial_transmission():
    """SGL分别验证完整下传和受30秒容量限制的部分下传。"""
    complete = make_env([task()])
    _, _, _, _, info = complete.step(
        {"os01": composite(sub("task", "gs01"))}
    )
    assert complete.tasks["task"].status == TaskStatus.COMPLETED
    assert info["transmission_records"][0].accepted
    assert info["delivered_timeliness_raw"] == pytest.approx(
        info["timeliness_raw"]
    )
    assert np.isclose(complete.tasks["task"].total_accounted_data_mbit(), 600.0)

    partial = make_env([task(size=3000.0)])
    partial.step({"os01": composite(sub("task", "gs01"))})
    assert partial.tasks["task"].delivered_to_ground_mbit == 1800.0
    assert partial.tasks["task"].data_on_satellites_mbit[
        SATELLITES.index("os01")
    ] == 1200.0


def test_isl_and_idl_relay_preserve_data():
    """ISL与IDL只搬移数据，不产生地面送达量或副本。"""
    for target, expected_type in (("os02", "ISL"), ("cs01", "IDL")):
        environment = make_env([task()])
        _, _, _, _, info = environment.step(
            {"os01": composite(sub("task", target, 0.5))}
        )
        state = environment.tasks["task"]
        assert info["transmission_records"][0].link_type == expected_type
        assert state.data_on_satellites_mbit[SATELLITES.index(target)] == 300.0
        assert state.delivered_to_ground_mbit == 0.0
        assert np.isclose(state.total_accounted_data_mbit(), 600.0)


def test_task_lifetime_arrival_expiration_and_final_boundary():
    """保留未到达、严格过期、expiration可开始和终点语义。"""
    future = make_env([task("future", arrival=30.0, lifetime=30.0)])
    _, _, _, _, info = future.step(
        {"os01": composite(sub("future", "gs01"))}
    )
    assert TASK_NOT_ARRIVED in codes(info)

    expiring = make_env([task("expiring", lifetime=30.0)])
    expiring.step({})
    _, _, _, _, info = expiring.step(
        {"os01": composite(sub("expiring", "gs01", 0.1))}
    )
    assert info["transmission_records"][0].accepted
    expiring.step({})
    _, _, terminated, _, info = expiring.step(
        {"os01": composite(sub("expiring", "gs01", 0.1))}
    )
    assert terminated
    assert TASK_EXPIRED in codes(info)
    with pytest.raises(RuntimeError, match="episode已经结束"):
        expiring.step({})


def test_ground_station_calibration_is_preserved():
    """同站换向间隔不足时仍由三维ENU校准约束拒绝。"""
    environment = make_env(
        [task("first", "os01", size=60.0), task("second", "os02", size=60.0)]
    )
    _, _, _, _, info = environment.step(
        {
            "os01": composite(sub("first", "gs01", 1.0, 0.0)),
            "os02": composite(sub("second", "gs01", 1.0, 0.1)),
        }
    )
    assert CALIBRATION_TIME_INSUFFICIENT in codes(info)


def test_mid_slot_window_and_status_synchronization():
    """候选识别连续窗口，推进后的到达与过期状态同步到观测。"""
    dataset = make_dataset()
    dataset.windows = WindowIndex(
        [TransmissionWindow("mid", "SGL", "os01", "gs01", 10.0, 20.0, 60.0)]
    )
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        [task()],
    )
    environment.reset()
    assert ("task", "gs01") in environment.get_action_candidates("os01")

    arriving = make_env([task("arriving", arrival=30.0, lifetime=60.0)])
    observations, _, _, _, info = arriving.step({})
    assert observations["os01"]["candidate_tasks"][0]["task_id"] == "arriving"
    assert info["active_task_count"] == 1

    expired = make_env([task("expired", lifetime=10.0)])
    observations, _, _, _, info = expired.step({})
    assert observations["os01"]["candidate_tasks"] == []
    assert info["expired_task_count"] == 1


def test_scenario_1_three_concurrent_inter_satellite_transmissions():
    """同一源的3条不同星间链路可并发接受。"""
    tasks = [task("t{0}".format(index)) for index in range(1, 4)]
    environment = make_env(tasks)
    action = composite(
        *(sub("t{0}".format(index), "cs0{0}".format(index)) for index in range(1, 4))
    )
    _, _, _, _, info = environment.step({"os01": action})
    assert info["accepted_subaction_count"] == 3
    assert info["max_observed_inter_interface_usage"] == 3


def test_scenario_2_three_inter_plus_one_sgl_are_independent():
    """3条星间传输和1条独立SGL可在同一源卫星同时执行。"""
    tasks = [task("t{0}".format(index)) for index in range(1, 5)]
    environment = make_env(tasks)
    action = composite(
        sub("t1", "cs01"),
        sub("t2", "cs02"),
        sub("t3", "cs03"),
        sub("t4", "gs01"),
    )
    _, _, _, _, info = environment.step({"os01": action})
    assert info["accepted_subaction_count"] == 4
    assert info["accepted_idl_count"] == 3
    assert info["accepted_sgl_count"] == 1


def test_scenario_3_fourth_inter_satellite_outgoing_is_rejected():
    """即使接口时间可用，同一源发起的第4条星间传输仍被拒绝。"""
    tasks = [task("t{0}".format(index)) for index in range(1, 5)]
    environment = make_env(tasks)
    action = composite(
        *(sub("t{0}".format(index), "cs0{0}".format(index)) for index in range(1, 5))
    )
    _, _, _, _, info = environment.step({"os01": action})
    assert info["accepted_subaction_count"] == 3
    assert INTER_SATELLITE_OUTGOING_LIMIT_EXCEEDED in codes(info)


def test_scenario_4_second_non_overlapping_sgl_is_rejected():
    """同一源每时隙只能发起1条SGL，与时间是否重叠无关。"""
    environment = make_env(
        [
            task("first", station="gs01", size=60.0),
            task("second", station="gs02", size=60.0),
        ]
    )
    _, _, _, _, info = environment.step(
        {
            "os01": composite(
                sub("first", "gs01", 1.0, 0.0),
                sub("second", "gs02", 1.0, 0.5),
            )
        }
    )
    assert info["accepted_subaction_count"] == 1
    assert SGL_PER_SLOT_LIMIT_EXCEEDED in codes(info)


def test_scenario_5_task_is_globally_unique_per_slot():
    """任务分布在两星时也只能接受全局排序靠前的一条链路。"""
    environment = make_env([task()])
    set_holdings(environment, "task", {"os01": 300.0, "os02": 300.0})
    _, _, _, _, info = environment.step(
        {
            "os01": composite(sub("task", "cs01")),
            "os02": composite(sub("task", "cs02")),
        }
    )
    assert info["accepted_subaction_count"] == 1
    assert TASK_ALREADY_SCHEDULED_THIS_SLOT in codes(info)


def test_scenario_6_physical_link_is_unique_in_both_directions():
    """同向重复目标及反向使用同一无向物理链路都只能接受一条。"""
    same_direction = make_env([task("a"), task("b")])
    _, _, _, _, same_info = same_direction.step(
        {"os01": composite(sub("a", "cs01"), sub("b", "cs01"))}
    )
    assert same_info["accepted_subaction_count"] == 1
    assert DUPLICATE_TARGET_LINK_IN_COMPOSITE_ACTION in codes(same_info)

    opposite = make_env([task("a"), task("b")])
    set_holdings(opposite, "b", {"cs01": 600.0})
    _, _, _, _, opposite_info = opposite.step(
        {
            "os01": composite(sub("a", "cs01")),
            "cs01": composite(sub("b", "os01")),
        }
    )
    assert opposite_info["accepted_subaction_count"] == 1
    assert PHYSICAL_LINK_ALREADY_USED_THIS_SLOT in codes(opposite_info)


def test_scenario_7_receiver_interface_capacity_is_three():
    """同一接收星最多同时占用3个星间接口。"""
    tasks = [
        task("t{0}".format(index), "os0{0}".format(index))
        for index in range(1, 5)
    ]
    environment = make_env(tasks)
    actions = {
        "os0{0}".format(index): composite(
            sub("t{0}".format(index), "cs01")
        )
        for index in range(1, 5)
    }
    _, _, _, _, info = environment.step(actions)
    assert info["accepted_subaction_count"] == 3
    assert INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED in codes(info)


def test_scenario_8_transmit_and_receive_share_inter_interfaces():
    """一星2收1发允许，第4条重叠收发因共享容量被拒绝。"""
    tasks = [
        task("r1", "os01"),
        task("r2", "os02"),
        task("send", "os03"),
        task("r3", "os04"),
    ]
    environment = make_env(tasks)
    set_holdings(environment, "send", {"cs01": 600.0})
    _, _, _, _, info = environment.step(
        {
            "os01": composite(sub("r1", "cs01")),
            "os02": composite(sub("r2", "cs01")),
            "cs01": composite(sub("send", "ns01")),
            "os04": composite(sub("r3", "cs01")),
        }
    )
    assert info["accepted_subaction_count"] == 3
    assert INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED in codes(info)


def test_scenario_9_same_slot_forwarding_is_blocked():
    """B在槽初无数据时不能发送本时隙将从A收到的数据。"""
    environment = make_env([task()])
    _, _, _, _, info = environment.step(
        {
            "os01": composite(sub("task", "os02")),
            "os02": composite(sub("task", "cs01")),
        }
    )
    assert info["accepted_subaction_count"] == 1
    assert SAME_SLOT_FORWARDING_NOT_ALLOWED in codes(info)
    assert info["same_slot_forwarding_blocked_count"] == 1
    assert environment.tasks["task"].data_on_satellites_mbit[
        SATELLITES.index("os02")
    ] > 0.0


def test_scenario_10_atomic_commit_preserves_each_task():
    """并发搬移后每个任务的源减量和目标增量完全对应。"""
    tasks = [task("a"), task("b"), task("c")]
    environment = make_env(tasks)
    environment.step(
        {
            "os01": composite(
                sub("a", "cs01", 0.25),
                sub("b", "cs02", 0.5),
                sub("c", "cs03", 0.75),
            )
        }
    )
    for task_id, expected in (("a", 150.0), ("b", 300.0), ("c", 450.0)):
        state = environment.tasks[task_id]
        assert np.isclose(state.total_accounted_data_mbit(), 600.0)
        target = "cs0{0}".format("abc".index(task_id) + 1)
        assert state.data_on_satellites_mbit[SATELLITES.index(target)] == expected


def test_scenario_11_concurrent_timeliness_is_summed():
    """并发成功子动作的及时性等于逐条手工贡献之和。"""
    tasks = [task("high", priority=10), task("low", priority=4)]
    environment = make_env(tasks)
    environment.step(
        {
            "os01": composite(
                sub("high", "cs01", 0.5),
                sub("low", "cs02", 0.25),
            )
        }
    )
    expected = timeliness_contribution(tasks[0], 300.0, 0.0)
    expected += timeliness_contribution(tasks[1], 150.0, 0.0)
    assert np.isclose(environment.timeliness_raw, expected)
    # 两条均为星间中继：旧及时性继续累计，新送达及时性保持为零。
    assert environment.delivered_timeliness_raw == pytest.approx(0.0)


def test_scenario_12_arbitration_ignores_action_dictionary_order():
    """改变动作字典插入顺序不会改变全局接受与拒绝结果。"""
    tasks = [task("high", "os01", priority=10), task("low", "os02", priority=1)]

    def run(items):
        environment = make_env(tasks)
        _, _, _, _, info = environment.step(dict(items))
        return [
            (
                record.source_satellite_id,
                record.task_id,
                record.accepted,
                record.violation_codes,
            )
            for record in info["transmission_records"]
        ]

    first = [
        ("os01", composite(sub("high", "cs01"))),
        ("os02", composite(sub("low", "cs01"))),
    ]
    second = list(reversed(first))
    assert run(first) == run(second)
