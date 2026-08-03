"""实现支持复合并发动作的跨域卫星数据传输调度环境。

每个决策时隙为30秒。一颗卫星的一次策略调用对应一个复合动作，其中最多
包含4个不同任务的子动作：最多3条ISL/IDL和1条独立SGL。环境先基于时隙
开始快照完成全局仲裁，再原子提交数据变化，以避免零时延多跳和数据复制。
第二天仍不定义奖励，15颗卫星的奖励恒为0.0。
"""

from typing import Dict, Iterable, Optional

import numpy as np

from . import constraints as violation
from .constraints import (
    can_reserve_with_capacity,
    intervals_overlap,
    make_physical_inter_satellite_link_key,
    maximum_concurrent_usage,
    required_calibration_time_seconds,
)
from .metrics import (
    compute_total_window_seconds_by_satellite,
    load_balance,
    timeliness_contribution,
)
from .models import (
    GroundStationState,
    ReservedInterval,
    SatelliteCompositeAction,
    SatelliteState,
    TaskState,
    TaskStatus,
    TransmissionRecord,
)


class CrossDomainSatelliteRangeSchedulingEnv:
    """15卫星结构化多智能体调度环境。

    输入动作中的时间单位为秒、数据量单位为Mbit。每颗卫星每时隙最多提交
    4个子动作；星间发送上限和星间接口容量均为3，SGL发送上限为1。本类的
    ``step`` 会修改任务、卫星、地面站和累计指标状态，其余查询方法只读。
    """

    def __init__(
        self,
        dataset,
        config,
        tasks: Optional[Iterable] = None,
        task_database=None,
        task_splits=None,
    ):
        """创建环境，但不开始episode；任务和链路单位分别为Mbit和秒。"""
        self.dataset = dataset
        self.config = config["environment"] if "environment" in config else config
        self.task_definitions = list(tasks or ())
        self.task_database = task_database
        self.task_splits = task_splits
        self.step_seconds = float(self.config["decision_step_seconds"])
        self.horizon_seconds = float(self.config["horizon_seconds"])
        self.data_tolerance = float(self.config["numerical"]["data_tolerance_mbit"])
        self.time_tolerance = float(
            self.config["numerical"]["time_tolerance_seconds"]
        )
        self.inter_interface_capacity = int(
            self.config["resources"]["inter_satellite_interface_count"]
        )
        self.max_inter_outgoing = int(
            self.config["concurrency"][
                "max_outgoing_inter_satellite_links_per_satellite_per_slot"
            ]
        )
        self.max_sgl_outgoing = int(
            self.config["concurrency"]["max_sgl_links_per_satellite_per_slot"]
        )
        self.total_window_seconds = compute_total_window_seconds_by_satellite(dataset)
        self.step_index = 0
        self.terminated = False

    def reset(self, seed=2025, split=None, task_ids=None, task_count=None):
        """重置episode并返回局部观测和任务选择信息。

        本方法会清空运行状态。任务数据量单位为Mbit，时间单位为秒；复合动作
        的并发上限不会因重置参数改变。
        """
        definitions = self._select_tasks(seed, split, task_ids, task_count)
        self.step_index = 0
        self.current_time_s = 0.0
        self.terminated = False
        self.tasks = {}
        for task in definitions:
            holdings = np.zeros(len(self.dataset.satellite_ids), dtype=float)
            source = self.dataset.satellite_index[task.source_satellite_id]
            holdings[source] = task.data_size_mbit
            status = (
                TaskStatus.ACTIVE
                if task.arrival_time_s <= 0.0
                else TaskStatus.NOT_ARRIVED
            )
            self.tasks[task.task_id] = TaskState(task, status, holdings)

        self.task_index = {
            task_id: index for index, task_id in enumerate(self.tasks)
        }
        self.outgoing_seconds = np.zeros(
            (len(self.tasks), len(self.dataset.satellite_ids))
        )
        self.satellite_states = {
            satellite_id: SatelliteState(
                satellite_id,
                self.dataset.satellite_domain_ids[index],
                {},
            )
            for index, satellite_id in enumerate(self.dataset.satellite_ids)
        }
        self.ground_station_states = {
            station_id: GroundStationState(
                station_id,
                self.dataset.antenna_rotation_speed_by_station[station_id],
            )
            for station_id in self.dataset.ground_station_ids
        }
        self.timeliness_raw = 0.0
        # 旧指标继续累计全部成功链路；新增指标只累计真正送达地面的SGL。
        self.delivered_timeliness_raw = 0.0
        self.accepted_total = 0
        self.rejected_total = 0
        return self._observations(), {"selected_task_ids": list(self.tasks)}

    def _select_tasks(self, seed, split, task_ids, task_count):
        """从注入任务或数据库确定本episode的任务，不修改数据库。"""
        if task_ids is not None:
            source = self.task_database or {
                task.task_id: task for task in self.task_definitions
            }
            unknown = set(task_ids) - set(source)
            if unknown:
                raise ValueError("reset包含未知任务ID：{0}".format(sorted(unknown)))
            return [source[task_id] for task_id in task_ids]
        if split is not None:
            if self.task_database is None or self.task_splits is None:
                raise ValueError("按split重置需要task_database和task_splits")
            from .tasks import sample_episode_tasks

            split_ids = self.task_splits[split]
            count = task_count if task_count is not None else min(200, len(split_ids))
            return sample_episode_tasks(self.task_database, split_ids, count, seed)
        return list(self.task_definitions)

    def get_action_candidates(self, satellite_id):
        """返回当前30秒时隙可尝试的 ``(task_id, target_id)`` 列表。

        结果只反映任务持有量和窗口相交关系，不预先判断全局并发冲突，也不
        修改状态；真正的3条星间和1条SGL上限由 ``step`` 统一仲裁。
        """
        if satellite_id not in self.dataset.satellite_index:
            raise ValueError("未知卫星：{0}".format(satellite_id))
        if self.terminated:
            return []
        source = self.dataset.satellite_index[satellite_id]
        slot_start = self.current_time_s
        slot_end = slot_start + self.step_seconds
        candidates = []
        for state in self.tasks.values():
            if state.status != TaskStatus.ACTIVE:
                continue
            if state.data_on_satellites_mbit[source] <= self.data_tolerance:
                continue
            station = state.definition.target_ground_station_id
            if self.dataset.windows.find_overlapping_window(
                "SGL", satellite_id, station, slot_start, slot_end
            ) is not None:
                candidates.append((state.definition.task_id, station))
            for target in self.dataset.satellite_ids:
                if target == satellite_id:
                    continue
                link_type = self.dataset.get_link_type(satellite_id, target)
                window = self.dataset.windows.find_overlapping_window(
                    link_type,
                    satellite_id,
                    target,
                    slot_start,
                    slot_end,
                )
                if window is not None:
                    candidates.append((state.definition.task_id, target))
        return sorted(candidates)

    def get_global_state(self):
        """返回只读结构化全局状态；数组时间单位为秒、数据单位为Mbit。"""
        matrix_index = min(self.step_index, self.dataset.step_count - 1)
        balance, balance_mean, mean_std = self._load_balance_values()
        return {
            "time_s": self.current_time_s,
            "step_index": self.step_index,
            "task_statuses": {
                task_id: {
                    "status": state.status.value,
                    "delivered_to_ground_mbit": state.delivered_to_ground_mbit,
                    "data_on_satellites_mbit": (
                        state.data_on_satellites_mbit.copy()
                    ),
                }
                for task_id, state in self.tasks.items()
            },
            "sgl_available": self.dataset.sgl_available[matrix_index].copy(),
            "isl_available": self.dataset.isl_available[matrix_index].copy(),
            "idl_available": self.dataset.idl_available[matrix_index].copy(),
            "ground_station_states": self.ground_station_states,
            "timeliness_raw": self.timeliness_raw,
            "delivered_timeliness_raw": self.delivered_timeliness_raw,
            "load_balance_raw": balance,
            "load_balance_mean_per_task": balance_mean,
            "mean_utilization_std": mean_std,
            "completed_task_count": self._count(TaskStatus.COMPLETED),
            "expired_task_count": self._count(TaskStatus.EXPIRED),
        }

    def step(self, actions):
        """执行一个30秒复合动作时隙并返回标准五元组。

        actions`必须是卫星ID到 SatelliteCompositeAction的字典，缺失
        卫星视为空闲。方法按“快照、展开、排序、仲裁、原子提交”修改状态；
        每星最多4个子动作、3条星间发送及1条SGL，奖励始终为0.0。
        """
        if self.terminated:
            raise RuntimeError("episode已经结束，终点状态不能开始新传输")
        normalized = self._normalize_actions(actions)
        self._update_task_statuses()

        # 所有源持有量检查只读取槽初副本。当前时隙收到的数据只能在下一时隙
        # 使用，从而避免30秒时隙内部出现零时延多跳转发。
        slot_start_holdings = {
            task_id: state.data_on_satellites_mbit.copy()
            for task_id, state in self.tasks.items()
        }
        candidates, records, submitted = self._build_candidates(
            normalized,
            slot_start_holdings,
        )
        accepted, arbitration_records, reservations = self._arbitrate(candidates)
        records.extend(arbitration_records)
        records.extend(self._commit_accepted(accepted))
        records.sort(key=self._record_sort_key)

        self._check_data_conservation()
        accepted_count = sum(record.accepted for record in records)
        rejected_count = len(records) - accepted_count
        self.accepted_total += accepted_count
        self.rejected_total += rejected_count
        self._advance_time()
        info = self._build_info(records, submitted, reservations)
        rewards = {
            satellite_id: 0.0 for satellite_id in self.dataset.satellite_ids
        }
        return self._observations(), rewards, self.terminated, False, info

    def _normalize_actions(self, actions):
        """校验动作容器并补齐空复合动作，不修改调用方字典。"""
        if not isinstance(actions, dict):
            raise ValueError("actions必须是卫星ID到复合动作的字典")
        unknown = set(actions) - set(self.dataset.satellite_ids)
        if unknown:
            raise ValueError("存在未知卫星动作：{0}".format(sorted(unknown)))
        normalized = {}
        for satellite_id in self.dataset.satellite_ids:
            action = actions.get(satellite_id, SatelliteCompositeAction())
            if not isinstance(action, SatelliteCompositeAction):
                raise ValueError("每颗卫星的动作必须是SatelliteCompositeAction")
            if len(action.transmissions) > 4:
                raise ValueError("复合动作最多包含4个传输子动作")
            normalized[satellite_id] = action
        return normalized

    def _build_candidates(self, actions, slot_start_holdings):
        """展开、投影并基础校验全部子动作，不修改任务数据。"""
        candidates = []
        records = []
        submitted = sum(len(action.transmissions) for action in actions.values())
        slot_end = self.current_time_s + self.step_seconds
        inbound_intents = {
            (subaction.task_id, subaction.target_id)
            for action in actions.values()
            for subaction in action.transmissions
            if subaction.target_id in self.dataset.satellite_index
        }

        for source_id in self.dataset.satellite_ids:
            action = actions[source_id]
            seen_tasks = set()
            seen_targets = set()
            for subaction_index, subaction in enumerate(action.transmissions):
                common = {
                    "source_id": source_id,
                    "subaction_index": subaction_index,
                    "subaction": subaction,
                }
                if subaction.task_id in seen_tasks:
                    records.append(
                        self._rejected(
                            common,
                            0.0,
                            self.current_time_s,
                            None,
                            violation.DUPLICATE_TASK_IN_COMPOSITE_ACTION,
                            False,
                        )
                    )
                    continue
                seen_tasks.add(subaction.task_id)
                if subaction.target_id in seen_targets:
                    records.append(
                        self._rejected(
                            common,
                            0.0,
                            self.current_time_s,
                            None,
                            violation.DUPLICATE_TARGET_LINK_IN_COMPOSITE_ACTION,
                            False,
                        )
                    )
                    continue
                seen_targets.add(subaction.target_id)

                try:
                    raw_ratio = float(subaction.transmission_ratio)
                    raw_offset = float(subaction.start_offset)
                except (TypeError, ValueError):
                    records.append(
                        self._rejected(
                            common,
                            0.0,
                            self.current_time_s,
                            None,
                            "NON_FINITE_ACTION",
                            False,
                        )
                    )
                    continue
                if not np.isfinite(raw_ratio) or not np.isfinite(raw_offset):
                    records.append(
                        self._rejected(
                            common,
                            0.0,
                            self.current_time_s,
                            None,
                            "NON_FINITE_ACTION",
                            False,
                        )
                    )
                    continue

                ratio = float(np.clip(raw_ratio, 0.0, 1.0))
                offset = float(np.clip(raw_offset, 0.0, 1.0))
                projected = ratio != raw_ratio or offset != raw_offset
                start = self.current_time_s + offset * self.step_seconds
                candidate, code = self._validate_subaction(
                    common,
                    ratio,
                    start,
                    slot_end,
                    projected,
                    slot_start_holdings,
                    inbound_intents,
                )
                if code is not None:
                    records.append(
                        self._rejected(
                            common,
                            ratio,
                            start,
                            candidate.get("link_type") if candidate else None,
                            code,
                            projected,
                            candidate.get("rate") if candidate else None,
                        )
                    )
                else:
                    candidates.append(candidate)
        return candidates, records, submitted

    def _validate_subaction(
        self,
        common,
        ratio,
        start,
        slot_end,
        projected,
        slot_start_holdings,
        inbound_intents,
    ):
        """基础校验一个子动作并计算可行数据量，不修改环境状态。"""
        source_id = common["source_id"]
        subaction = common["subaction"]
        state = self.tasks.get(subaction.task_id)
        if state is None:
            return {}, violation.INVALID_TASK
        if (
            state.status == TaskStatus.NOT_ARRIVED
            or start < state.definition.arrival_time_s - self.time_tolerance
        ):
            return {}, violation.TASK_NOT_ARRIVED
        if (
            state.status == TaskStatus.EXPIRED
            or start > state.definition.expiration_time_s + self.time_tolerance
        ):
            return {}, violation.TASK_EXPIRED
        if state.status == TaskStatus.COMPLETED:
            return {}, violation.TASK_COMPLETED
        if start >= slot_end - self.time_tolerance:
            return {}, violation.START_AT_OR_AFTER_SLOT_END
        if subaction.target_id == source_id:
            return {}, violation.INVALID_TARGET

        source_index = self.dataset.satellite_index[source_id]
        held_at_slot_start = slot_start_holdings[subaction.task_id][source_index]
        if held_at_slot_start <= self.data_tolerance:
            code = (
                violation.SAME_SLOT_FORWARDING_NOT_ALLOWED
                if (subaction.task_id, source_id) in inbound_intents
                else violation.SOURCE_HAS_NO_DATA
            )
            return {}, code
        if ratio * state.definition.data_size_mbit <= self.data_tolerance:
            return {}, violation.ZERO_REQUESTED_DATA

        try:
            link_type = self.dataset.get_link_type(source_id, subaction.target_id)
        except ValueError:
            return {}, violation.INVALID_TARGET
        partial = {"link_type": link_type}
        if (
            link_type == "SGL"
            and subaction.target_id != state.definition.target_ground_station_id
        ):
            return partial, violation.TARGET_GROUND_STATION_MISMATCH
        overlapping = self.dataset.windows.find_overlapping_window(
            link_type,
            source_id,
            subaction.target_id,
            self.current_time_s,
            slot_end,
        )
        if overlapping is None:
            return partial, violation.LINK_NOT_AVAILABLE
        window = self.dataset.windows.find_active_window(
            link_type,
            source_id,
            subaction.target_id,
            start,
        )
        if window is None:
            return partial, violation.LINK_NOT_AVAILABLE

        rate = float(window.rate_mbps)
        partial["rate"] = rate
        feasible_seconds = min(slot_end - start, window.end_time_s - start)
        requested = ratio * state.definition.data_size_mbit
        amount = min(
            requested,
            held_at_slot_start,
            state.remaining_data_mbit,
            max(0.0, feasible_seconds) * rate,
        )
        if amount <= self.data_tolerance:
            return partial, violation.ZERO_FEASIBLE_CAPACITY
        end = start + amount / rate
        candidate = {
            **common,
            "state": state,
            "link_type": link_type,
            "ratio": ratio,
            "start": start,
            "end": end,
            "rate": rate,
            "amount": amount,
            "projected": projected,
            "sort_key": (
                start,
                -state.definition.priority,
                state.definition.expiration_time_s,
                source_id,
                subaction.task_id,
                subaction.target_id,
                common["subaction_index"],
            ),
        }
        return candidate, None

    def _arbitrate(self, candidates):
        """按固定全局顺序仲裁候选，返回接受项、拒绝记录和资源预留。"""
        scheduled_tasks = set()
        used_physical_links = set()
        inter_outgoing = {item: 0 for item in self.dataset.satellite_ids}
        sgl_outgoing = {item: 0 for item in self.dataset.satellite_ids}
        inter_reservations = {
            item: [] for item in self.dataset.satellite_ids
        }
        ground_reservations = {
            item: [] for item in self.dataset.ground_station_ids
        }
        temporary_ground_state = {
            station_id: {
                "last_end": state.last_transmission_end_s,
                "last_pointing": state.last_pointing_enu,
            }
            for station_id, state in self.ground_station_states.items()
        }
        accepted = []
        records = []

        # 显式排序保证结果与动作字典和输入元组的迭代顺序无关。
        for candidate in sorted(candidates, key=lambda item: item["sort_key"]):
            code = self._arbitration_violation(
                candidate,
                scheduled_tasks,
                used_physical_links,
                inter_outgoing,
                sgl_outgoing,
                inter_reservations,
                ground_reservations,
                temporary_ground_state,
            )
            if code is not None:
                records.append(
                    self._rejected(
                        candidate,
                        candidate["ratio"],
                        candidate["start"],
                        candidate["link_type"],
                        code,
                        candidate["projected"],
                        candidate["rate"],
                    )
                )
                continue

            accepted.append(candidate)
            scheduled_tasks.add(candidate["subaction"].task_id)
            interval = ReservedInterval(
                candidate["start"],
                candidate["end"],
                candidate["subaction"].task_id,
            )
            source_id = candidate["source_id"]
            target_id = candidate["subaction"].target_id
            if candidate["link_type"] == "SGL":
                sgl_outgoing[source_id] += 1
                ground_reservations[target_id].append(interval)
                temporary_ground_state[target_id]["last_end"] = candidate["end"]
                temporary_ground_state[target_id]["last_pointing"] = (
                    self.dataset.get_sgl_pointing_vector(
                        source_id,
                        target_id,
                        candidate["end"],
                    )
                )
            else:
                inter_outgoing[source_id] += 1
                physical_key = make_physical_inter_satellite_link_key(
                    source_id,
                    target_id,
                )
                used_physical_links.add(physical_key)
                # 星间发送和接收都占用接口，二者共享每星3接口容量。
                inter_reservations[source_id].append(interval)
                inter_reservations[target_id].append(interval)

        return accepted, records, inter_reservations

    def _arbitration_violation(
        self,
        candidate,
        scheduled_tasks,
        used_physical_links,
        inter_outgoing,
        sgl_outgoing,
        inter_reservations,
        ground_reservations,
        temporary_ground_state,
    ):
        """按提示词G1—G6顺序返回首个全局仲裁违反代码。"""
        subaction = candidate["subaction"]
        source_id = candidate["source_id"]
        target_id = subaction.target_id
        if subaction.task_id in scheduled_tasks:
            return violation.TASK_ALREADY_SCHEDULED_THIS_SLOT
        if candidate["link_type"] != "SGL":
            if inter_outgoing[source_id] >= self.max_inter_outgoing:
                return violation.INTER_SATELLITE_OUTGOING_LIMIT_EXCEEDED
        elif sgl_outgoing[source_id] >= self.max_sgl_outgoing:
            return violation.SGL_PER_SLOT_LIMIT_EXCEEDED

        interval = ReservedInterval(
            candidate["start"],
            candidate["end"],
            subaction.task_id,
        )
        if candidate["link_type"] != "SGL":
            physical_key = make_physical_inter_satellite_link_key(
                source_id,
                target_id,
            )
            if physical_key in used_physical_links:
                return violation.PHYSICAL_LINK_ALREADY_USED_THIS_SLOT
            for endpoint in (source_id, target_id):
                if not can_reserve_with_capacity(
                    inter_reservations[endpoint],
                    interval,
                    self.inter_interface_capacity,
                    self.time_tolerance,
                ):
                    return violation.INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED
            return None

        station_intervals = ground_reservations[target_id]
        if any(
            intervals_overlap(
                interval.start_s,
                interval.end_s,
                reserved.start_s,
                reserved.end_s,
                self.time_tolerance,
            )
            for reserved in station_intervals
        ):
            return violation.GROUND_STATION_RESOURCE_CONFLICT

        station_state = temporary_ground_state[target_id]
        if station_state["last_end"] is not None:
            next_pointing = self.dataset.get_sgl_pointing_vector(
                source_id,
                target_id,
                candidate["start"],
            )
            calibration = required_calibration_time_seconds(
                station_state["last_pointing"],
                next_pointing,
                self.ground_station_states[
                    target_id
                ].antenna_rotation_speed_deg_per_second,
            )
            gap = candidate["start"] - station_state["last_end"]
            if gap + self.time_tolerance < calibration:
                return violation.CALIBRATION_TIME_INSUFFICIENT
        return None

    def _commit_accepted(self, accepted):
        """原子提交全部已接受传输并更新指标，返回接受记录。

        仲裁期间不写任务数据；此处先汇总所有Mbit增减量，再统一写入，从而
        保证任何子动作都不能消费当前时隙刚收到的数据。
        """
        holding_deltas = {
            task_id: np.zeros(len(self.dataset.satellite_ids), dtype=float)
            for task_id in self.tasks
        }
        delivered_deltas = {task_id: 0.0 for task_id in self.tasks}
        records = []
        for candidate in accepted:
            state = candidate["state"]
            task_id = state.definition.task_id
            source_id = candidate["source_id"]
            target_id = candidate["subaction"].target_id
            source_index = self.dataset.satellite_index[source_id]
            amount = candidate["amount"]
            holding_deltas[task_id][source_index] -= amount
            if candidate["link_type"] == "SGL":
                delivered_deltas[task_id] += amount
            else:
                target_index = self.dataset.satellite_index[target_id]
                holding_deltas[task_id][target_index] += amount

        for task_id, delta in holding_deltas.items():
            state = self.tasks[task_id]
            state.data_on_satellites_mbit += delta
            state.data_on_satellites_mbit[
                np.abs(state.data_on_satellites_mbit) <= self.data_tolerance
            ] = 0.0
            state.delivered_to_ground_mbit += delivered_deltas[task_id]

        for candidate in accepted:
            state = candidate["state"]
            source_id = candidate["source_id"]
            target_id = candidate["subaction"].target_id
            duration = candidate["amount"] / candidate["rate"]
            contribution = timeliness_contribution(
                state.definition,
                candidate["amount"],
                candidate["start"],
                self.time_tolerance,
            )
            self.timeliness_raw += contribution
            if candidate["link_type"] == "SGL":
                self.delivered_timeliness_raw += contribution
            task_row = self.task_index[state.definition.task_id]
            source_index = self.dataset.satellite_index[source_id]
            self.outgoing_seconds[task_row, source_index] += duration
            transmitted = self.satellite_states[
                source_id
            ].transmitted_seconds_by_task
            transmitted[state.definition.task_id] = (
                transmitted.get(state.definition.task_id, 0.0) + duration
            )
            if candidate["link_type"] == "SGL":
                station_state = self.ground_station_states[target_id]
                station_state.last_transmission_end_s = candidate["end"]
                station_state.last_pointing_enu = (
                    self.dataset.get_sgl_pointing_vector(
                        source_id,
                        target_id,
                        candidate["end"],
                    )
                )
            if (
                state.delivered_to_ground_mbit
                >= state.definition.data_size_mbit - self.data_tolerance
            ):
                state.delivered_to_ground_mbit = state.definition.data_size_mbit
                state.status = TaskStatus.COMPLETED
                state.completion_time_s = candidate["end"]
            records.append(self._accepted_record(candidate))
        return records

    @staticmethod
    def _accepted_record(candidate):
        """由已接受候选构造记录，不修改状态。"""
        subaction = candidate["subaction"]
        return TransmissionRecord(
            source_satellite_id=candidate["source_id"],
            subaction_index=candidate["subaction_index"],
            composite_source_id=candidate["source_id"],
            target_id=subaction.target_id,
            task_id=subaction.task_id,
            link_type=candidate["link_type"],
            requested_ratio=candidate["ratio"],
            requested_start_s=candidate["start"],
            accepted=True,
            transmitted_data_mbit=candidate["amount"],
            actual_start_s=candidate["start"],
            actual_end_s=candidate["end"],
            rate_mbps=candidate["rate"],
            violation_codes=(),
            projected=candidate["projected"],
        )

    @staticmethod
    def _rejected(
        common,
        ratio,
        start,
        link_type,
        code,
        projected,
        rate=None,
    ):
        """构造拒绝记录；拒绝的子动作不会修改任何环境状态。"""
        subaction = common["subaction"]
        return TransmissionRecord(
            source_satellite_id=common["source_id"],
            subaction_index=common["subaction_index"],
            composite_source_id=common["source_id"],
            target_id=subaction.target_id,
            task_id=subaction.task_id,
            link_type=link_type,
            requested_ratio=ratio,
            requested_start_s=start,
            accepted=False,
            transmitted_data_mbit=0.0,
            actual_start_s=None,
            actual_end_s=None,
            rate_mbps=rate,
            violation_codes=(code,),
            projected=projected,
        )

    @staticmethod
    def _record_sort_key(record):
        """给调试记录提供稳定顺序，不依赖动作字典顺序。"""
        return (
            record.requested_start_s,
            record.source_satellite_id,
            record.task_id or "",
            record.target_id or "",
            record.subaction_index,
        )

    def _advance_time(self):
        """推进到下一个决策时刻，并同步到达和过期状态。"""
        self.step_index += 1
        self.current_time_s = self.step_index * self.step_seconds
        self.terminated = self.step_index >= self.dataset.step_count
        self._update_task_statuses()

    def _update_task_statuses(self):
        """在当前秒数激活任务，并仅在严格超过expiration后使其过期。"""
        for state in self.tasks.values():
            if (
                state.status == TaskStatus.NOT_ARRIVED
                and state.definition.arrival_time_s <= self.current_time_s
            ):
                state.status = TaskStatus.ACTIVE
            if (
                state.status == TaskStatus.ACTIVE
                and self.current_time_s > state.definition.expiration_time_s
            ):
                state.status = TaskStatus.EXPIRED

    def _check_data_conservation(self):
        """验证每个任务的Mbit数据不复制、不丢失且没有显著负值。"""
        for state in self.tasks.values():
            if np.any(state.data_on_satellites_mbit < -self.data_tolerance):
                raise RuntimeError("任务在卫星上的数据量出现负值")
            if (
                state.delivered_to_ground_mbit
                > state.definition.data_size_mbit + self.data_tolerance
            ):
                raise RuntimeError("地面站送达量超过任务原始数据量")
            difference = (
                state.total_accounted_data_mbit() - state.definition.data_size_mbit
            )
            if abs(difference) > self.data_tolerance:
                raise RuntimeError(
                    "任务数据不守恒：{0}".format(state.definition.task_id)
                )
            if (
                state.status == TaskStatus.COMPLETED
                and state.remaining_data_mbit > self.data_tolerance
            ):
                raise RuntimeError("完成任务仍有未送达数据")

    def check_data_conservation(self):
        """公开执行任务数据守恒检查；本方法只读状态且不返回数值。"""
        self._check_data_conservation()

    def _load_balance_values(self):
        """返回已到达任务的修正负载均衡指标，不修改状态。"""
        arrived_rows = [
            self.task_index[task_id]
            for task_id, state in self.tasks.items()
            if state.definition.arrival_time_s <= self.current_time_s
        ]
        values = self.outgoing_seconds[arrived_rows]
        return load_balance(
            values,
            self.total_window_seconds,
            float(self.config["metrics"]["load_balance"]["b_max"]),
        )

    def _observations(self):
        """生成每颗卫星的只读结构化局部观测，数据单位为Mbit。"""
        balance, _, _ = self._load_balance_values()
        observations = {}
        candidate_limit = int(self.config["action"]["candidate_task_count"])
        for index, satellite_id in enumerate(self.dataset.satellite_ids):
            held_tasks = []
            for state in self.tasks.values():
                held = float(state.data_on_satellites_mbit[index])
                if state.status == TaskStatus.ACTIVE and held > self.data_tolerance:
                    held_tasks.append(
                        {
                            "task_id": state.definition.task_id,
                            "priority": state.definition.priority,
                            "target_ground_station_id": (
                                state.definition.target_ground_station_id
                            ),
                            "held_data_mbit": held,
                            "remaining_total_data_mbit": state.remaining_data_mbit,
                            "remaining_lifetime_s": (
                                state.definition.expiration_time_s
                                - self.current_time_s
                            ),
                            "expiration_time_s": (
                                state.definition.expiration_time_s
                            ),
                        }
                    )
            held_tasks.sort(
                key=lambda item: (
                    item["expiration_time_s"],
                    -item["priority"],
                    -item["held_data_mbit"],
                    item["task_id"],
                )
            )
            candidates = (
                self.get_action_candidates(satellite_id)
                if not self.terminated
                else []
            )
            observations[satellite_id] = {
                "satellite_id": satellite_id,
                "domain_id": self.dataset.satellite_domain_ids[index],
                "time_s": self.current_time_s,
                "step_index": self.step_index,
                "held_tasks": held_tasks,
                "candidate_tasks": held_tasks[:candidate_limit],
                "available_satellite_targets": sorted(
                    {
                        target
                        for _, target in candidates
                        if target in self.dataset.satellite_index
                    }
                ),
                "available_ground_station_targets": sorted(
                    {
                        target
                        for _, target in candidates
                        if target in self.dataset.ground_station_index
                    }
                ),
                "action_limits": {
                    "max_subactions": 4,
                    "max_inter_satellite_subactions": 3,
                    "max_sgl_subactions": 1,
                },
                "timeliness_raw": self.timeliness_raw,
                "delivered_timeliness_raw": self.delivered_timeliness_raw,
                "load_balance_raw": balance,
            }
        return observations

    def _build_info(self, records, submitted, inter_reservations):
        """汇总刚结束时隙和episode累计指标，不写逐步日志。"""
        balance, balance_mean, mean_std = self._load_balance_values()
        accepted = [record for record in records if record.accepted]
        rejected = [record for record in records if not record.accepted]
        return {
            "time_s": self.current_time_s,
            "step_index": self.step_index,
            "transmission_records": records,
            "submitted_subaction_count": submitted,
            "accepted_subaction_count": len(accepted),
            "rejected_subaction_count": len(rejected),
            "accepted_isl_count": sum(
                record.link_type == "ISL" for record in accepted
            ),
            "accepted_idl_count": sum(
                record.link_type == "IDL" for record in accepted
            ),
            "accepted_sgl_count": sum(
                record.link_type == "SGL" for record in accepted
            ),
            "rejected_by_task_uniqueness_count": self._violation_count(
                rejected,
                violation.TASK_ALREADY_SCHEDULED_THIS_SLOT,
            ),
            "rejected_by_physical_link_count": self._violation_count(
                rejected,
                violation.PHYSICAL_LINK_ALREADY_USED_THIS_SLOT,
            ),
            "rejected_by_interface_capacity_count": self._violation_count(
                rejected,
                violation.INTER_SATELLITE_INTERFACE_CAPACITY_EXCEEDED,
            ),
            "rejected_by_sgl_limit_count": self._violation_count(
                rejected,
                violation.SGL_PER_SLOT_LIMIT_EXCEEDED,
            ),
            "max_observed_inter_interface_usage": max(
                (
                    maximum_concurrent_usage(intervals)
                    for intervals in inter_reservations.values()
                ),
                default=0,
            ),
            "same_slot_forwarding_blocked_count": self._violation_count(
                rejected,
                violation.SAME_SLOT_FORWARDING_NOT_ALLOWED,
            ),
            "timeliness_raw": self.timeliness_raw,
            "delivered_timeliness_raw": self.delivered_timeliness_raw,
            "load_balance_raw": balance,
            "load_balance_mean_per_task": balance_mean,
            "mean_utilization_std": mean_std,
            "completed_task_count": self._count(TaskStatus.COMPLETED),
            "expired_task_count": self._count(TaskStatus.EXPIRED),
            "active_task_count": self._count(TaskStatus.ACTIVE),
            "delivered_data_mbit": sum(
                state.delivered_to_ground_mbit for state in self.tasks.values()
            ),
            "accepted_transmission_count": self.accepted_total,
            "rejected_transmission_count": self.rejected_total,
        }

    @staticmethod
    def _violation_count(records, code):
        """统计包含指定稳定违反代码的拒绝记录数量。"""
        return sum(code in record.violation_codes for record in records)

    def _count(self, status):
        """返回指定任务状态的数量，不修改状态。"""
        return sum(state.status == status for state in self.tasks.values())
