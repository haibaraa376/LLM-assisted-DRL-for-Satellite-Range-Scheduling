"""将第二天结构化观测编码为固定维度MAPPO输入并解码复合动作。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from srs_env.models import (
    SatelliteCompositeAction,
    TaskStatus,
    TransmissionSubAction,
)

from .config import validate_runtime_compatibility


@dataclass(frozen=True)
class EncodedAgentObservation:
    """保存一个智能体的213维输入、基础Mask及稳定槽位映射。"""

    observation: np.ndarray
    base_target_mask: np.ndarray
    task_ids: tuple
    target_ids: tuple

    def __post_init__(self):
        """检查固定形状、数据类型和有限性，防止错误传播到策略网络。"""
        if self.observation.shape != (213,) or self.observation.dtype != np.float32:
            raise ValueError("Actor观测必须是shape=(213,)的float32数组")
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("Actor观测包含NaN或Inf")
        if self.base_target_mask.shape != (4, 20):
            raise ValueError("基础目标Mask形状必须为(4,20)")
        if self.base_target_mask.dtype != bool:
            raise ValueError("基础目标Mask必须使用bool类型")
        if len(self.task_ids) != 4 or len(self.target_ids) != 19:
            raise ValueError("任务槽必须为4，目标槽必须为19")
        if not np.all(self.base_target_mask[:, 0]):
            raise ValueError("每个任务槽的IDLE动作必须有效")


class MappoObservationEncoder:
    """构造共享Actor输入和集中式Critic状态。

    卫星ID不会作为连续整数输入，而使用业务域one-hot及真实轨道角的正余弦，
    避免人为引入ID大小关系。Top-4固定槽让共享Actor始终接收213维输入；
    目标顺序固定为数据集的15星再4站，保证训练、解码和Checkpoint语义一致。
    本编码器只读取环境，不改变任务或资源状态。
    """

    def __init__(self, env, config, constellation_path=Path("configs/constellation.yaml")):
        """初始化编码元数据并计算距离参考值；距离单位为km。"""
        validate_runtime_compatibility(config, env)
        self.env = env
        self.config = config
        self.encoding = config["encoding"]
        self.normalization = self.encoding["normalization"]
        self.satellite_ids = tuple(env.dataset.satellite_ids)
        self.ground_station_ids = tuple(env.dataset.ground_station_ids)
        self.target_ids = self.satellite_ids + self.ground_station_ids
        self._satellite_identity = self._load_identity_features(constellation_path)
        finite_distances = np.concatenate(
            [
                env.dataset.sgl_range_km[np.isfinite(env.dataset.sgl_range_km)],
                env.dataset.isl_range_km[np.isfinite(env.dataset.isl_range_km)],
                env.dataset.idl_range_km[np.isfinite(env.dataset.idl_range_km)],
            ]
        )
        if finite_distances.size == 0 or finite_distances.max() <= 0.0:
            raise ValueError("链路数据中没有可用的正有限距离参考值")
        self.distance_reference_km = float(finite_distances.max())

    def _load_identity_features(self, path):
        """从真实RAAN和平近点角构造7维静态身份，不使用未来轨道位置。"""
        with Path(path).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        records = {item["id"]: item for item in document.get("satellites", [])}
        domain_index = {"D1": 0, "D2": 1, "D3": 2}
        features = {}
        for satellite_id in self.satellite_ids:
            if satellite_id not in records:
                raise ValueError("星座配置缺少卫星：{0}".format(satellite_id))
            record = records[satellite_id]
            elements = record.get("orbital_elements", {})
            if "raan_deg" not in elements or "mean_anomaly_deg" not in elements:
                raise ValueError(
                    "卫星{0}缺少RAAN或初始平近点角".format(satellite_id)
                )
            domain = record.get("domain_id")
            if domain not in domain_index:
                raise ValueError("卫星业务域必须是D1、D2或D3")
            identity = np.zeros(7, dtype=np.float32)
            identity[domain_index[domain]] = 1.0
            raan = np.radians(float(elements["raan_deg"]))
            phase = np.radians(float(elements["mean_anomaly_deg"]))
            identity[3:] = (np.sin(raan), np.cos(raan), np.sin(phase), np.cos(phase))
            features[satellite_id] = identity
        return features

    def encode_agent(self, satellite_id, structured_observation):
        """编码一颗卫星的当前局部观测，返回213维float32结果与Mask。"""
        if satellite_id not in self.env.dataset.satellite_index:
            raise ValueError("编码请求包含未知卫星：{0}".format(satellite_id))
        if structured_observation.get("satellite_id") != satellite_id:
            raise ValueError("结构化观测的卫星ID与编码请求不一致")
        if not self.env.tasks:
            raise ValueError("episode任务数为0，无法归一化活跃任务数量")

        task_items = self._select_candidate_tasks(structured_observation)
        task_features, task_ids = self._encode_tasks(task_items)
        target_features = self._encode_targets(satellite_id)
        self_features = self._encode_self_state(satellite_id, structured_observation)
        observation = np.concatenate(
            [
                self._satellite_identity[satellite_id],
                self_features,
                task_features.reshape(-1),
                target_features.reshape(-1),
            ]
        ).astype(np.float32, copy=False)
        if self.normalization["clip_features_to_unit_interval"]:
            # 轨道正余弦允许[-1,1]；其余归一化特征限制在[0,1]。
            observation[7:] = np.clip(observation[7:], 0.0, 1.0)
        base_mask = self._build_base_mask(satellite_id, task_ids)
        return EncodedAgentObservation(
            observation=observation,
            base_target_mask=base_mask,
            task_ids=task_ids,
            target_ids=self.target_ids,
        )

    def _select_candidate_tasks(self, structured_observation):
        """再次稳定排序候选任务并保留前4项，避免依赖字典顺序。"""
        items = list(structured_observation.get("candidate_tasks", []))
        for item in items:
            required = {
                "task_id",
                "priority",
                "held_data_mbit",
                "expiration_time_s",
            }
            if not required.issubset(item):
                raise ValueError("候选任务缺少编码所需字段")
        items.sort(
            key=lambda item: (
                item["expiration_time_s"],
                -item["priority"],
                -item["held_data_mbit"],
                item["task_id"],
            )
        )
        return items[: self.encoding["candidate_task_count"]]

    def _encode_self_state(self, satellite_id, observation):
        """编码当前时间、ACTIVE数据总量和ACTIVE任务数三个比例。"""
        satellite_index = self.env.dataset.satellite_index[satellite_id]
        active_states = [
            state for state in self.env.tasks.values() if state.status == TaskStatus.ACTIVE
        ]
        total_held = sum(
            state.data_on_satellites_mbit[satellite_index] for state in active_states
        )
        active_count = sum(
            state.data_on_satellites_mbit[satellite_index] > self.env.data_tolerance
            for state in active_states
        )
        maximum_held = (
            self.encoding["candidate_task_count"]
            * self.normalization["max_task_size_mbit"]
        )
        values = np.array(
            [
                float(observation["time_s"]) / self.normalization["horizon_s"],
                total_held / maximum_held,
                active_count / len(self.env.tasks),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("卫星自身状态包含NaN或Inf")
        return np.clip(values, 0.0, 1.0)

    def _encode_tasks(self, task_items):
        """编码4×8任务特征；不足4项时用全零槽补齐。"""
        count = self.encoding["candidate_task_count"]
        features = np.zeros((count, self.encoding["task_feature_dim"]), dtype=np.float32)
        task_ids = [None] * count
        station_index = {
            station_id: index
            for index, station_id in enumerate(self.ground_station_ids)
        }
        for slot, item in enumerate(task_items):
            task_id = item["task_id"]
            if task_id not in self.env.tasks:
                raise ValueError("候选任务ID不属于当前episode：{0}".format(task_id))
            state = self.env.tasks[task_id]
            definition = state.definition
            station = definition.target_ground_station_id
            task_ids[slot] = task_id
            features[slot, 0] = item["held_data_mbit"] / definition.data_size_mbit
            features[slot, 1] = state.remaining_data_mbit / definition.data_size_mbit
            features[slot, 2] = definition.priority / self.normalization["max_priority"]
            slack = max(definition.expiration_time_s - self.env.current_time_s, 0.0)
            features[slot, 3] = slack / self.normalization["max_task_lifetime_s"]
            features[slot, 4 + station_index[station]] = 1.0
        if not np.all(np.isfinite(features)):
            raise ValueError("任务特征包含NaN或Inf")
        return np.clip(features, 0.0, 1.0), tuple(task_ids)

    def _encode_targets(self, source_id):
        """按固定19目标顺序编码当前时隙链路特征。"""
        features = np.zeros(
            (self.encoding["target_slot_count"], self.encoding["target_feature_dim"]),
            dtype=np.float32,
        )
        slot_start = self.env.current_time_s
        slot_end = slot_start + self.env.step_seconds
        source_index = self.env.dataset.satellite_index[source_id]
        for target_slot, target_id in enumerate(self.target_ids):
            if target_id == source_id:
                features[target_slot, 8] = 1.0
                continue
            link_type = self.env.dataset.get_link_type(source_id, target_id)
            window = self.env.dataset.windows.find_overlapping_window(
                link_type,
                source_id,
                target_id,
                slot_start,
                slot_end,
            )
            if window is None:
                features[target_slot, 8] = 1.0
            else:
                overlap_start = max(slot_start, window.start_time_s)
                overlap_end = min(slot_end, window.end_time_s)
                overlap_seconds = max(overlap_end - overlap_start, 0.0)
                features[target_slot, 0] = 1.0
                features[target_slot, 1 + ("SGL", "ISL", "IDL").index(link_type)] = 1.0
                capacity = overlap_seconds * window.rate_mbps
                features[target_slot, 4] = (
                    capacity / self.normalization["max_slot_capacity_mbit"]
                )
                features[target_slot, 5] = (
                    max(window.end_time_s - slot_start, 0.0)
                    / self.normalization["horizon_s"]
                )
                features[target_slot, 6] = self._distance_ratio(
                    link_type,
                    source_index,
                    target_id,
                    overlap_start,
                )
                features[target_slot, 8] = (
                    overlap_start - slot_start
                ) / self.env.step_seconds
            if target_id in self.env.dataset.satellite_index:
                target_index = self.env.dataset.satellite_index[target_id]
                target_held = sum(
                    state.data_on_satellites_mbit[target_index]
                    for state in self.env.tasks.values()
                    if state.status == TaskStatus.ACTIVE
                )
                maximum_held = (
                    self.encoding["candidate_task_count"]
                    * self.normalization["max_task_size_mbit"]
                )
                features[target_slot, 7] = target_held / maximum_held
        if not np.all(np.isfinite(features)):
            raise ValueError("目标特征包含NaN或Inf")
        return np.clip(features, 0.0, 1.0)

    def _distance_ratio(self, link_type, source_index, target_id, time_s):
        """读取窗口交集起点处的离散链路距离并按数据集最大值归一化。"""
        time_index = min(
            int(max(time_s, 0.0) // self.env.step_seconds),
            len(self.env.dataset.timestamps_unix_s) - 1,
        )
        if link_type == "SGL":
            target_index = self.env.dataset.ground_station_index[target_id]
            distance = self.env.dataset.sgl_range_km[
                time_index,
                source_index,
                target_index,
            ]
        else:
            target_index = self.env.dataset.satellite_index[target_id]
            matrix = (
                self.env.dataset.isl_range_km
                if link_type == "ISL"
                else self.env.dataset.idl_range_km
            )
            distance = matrix[time_index, source_index, target_index]
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("链路距离必须是非负有限数")
        return float(distance) / self.distance_reference_km

    def _build_base_mask(self, satellite_id, task_ids):
        """构造4×20局部合法Mask，不提前推测其他智能体的全局竞争。"""
        mask = np.zeros((4, 20), dtype=bool)
        mask[:, 0] = True
        environment_candidates = set(self.env.get_action_candidates(satellite_id))
        for task_slot, task_id in enumerate(task_ids):
            if task_id is None:
                continue
            state = self.env.tasks[task_id]
            for target_slot, target_id in enumerate(self.target_ids):
                choice = target_slot + 1
                if target_id == satellite_id:
                    continue
                if (task_id, target_id) not in environment_candidates:
                    continue
                if target_id in self.env.dataset.ground_station_index:
                    if target_id != state.definition.target_ground_station_id:
                        continue
                mask[task_slot, choice] = True
        return mask

    def encode_all_agents(self, observations):
        """按数据集固定卫星顺序编码15个智能体，不修改观测字典。"""
        if set(observations) != set(self.satellite_ids):
            raise ValueError("局部观测必须完整包含15颗卫星")
        return {
            satellite_id: self.encode_agent(
                satellite_id,
                observations[satellite_id],
            )
            for satellite_id in self.satellite_ids
        }

    def encode_critic_state(self, encoded_agents, global_state):
        """拼接15×213局部输入与5个全局比例，返回3200维float32状态。"""
        if not self.env.tasks:
            raise ValueError("episode任务数为0，无法编码集中式状态")
        local = [encoded_agents[item].observation for item in self.satellite_ids]
        task_count = len(self.env.tasks)
        statuses = global_state["task_statuses"].values()
        total_data = sum(state.definition.data_size_mbit for state in self.env.tasks.values())
        delivered = sum(
            item["delivered_to_ground_mbit"]
            for item in global_state["task_statuses"].values()
        )
        extras = np.array(
            [
                global_state["time_s"] / self.normalization["horizon_s"],
                sum(item["status"] == TaskStatus.ACTIVE.value for item in statuses)
                / task_count,
                global_state["completed_task_count"] / task_count,
                global_state["expired_task_count"] / task_count,
                delivered / total_data,
            ],
            dtype=np.float32,
        )
        result = np.concatenate(local + [extras]).astype(np.float32, copy=False)
        expected = self.encoding["expected_critic_state_dim"]
        if result.shape != (expected,) or not np.all(np.isfinite(result)):
            raise ValueError("集中式Critic状态形状或数值无效")
        return result

    def metadata(self):
        """返回Checkpoint所需的固定编码语义元数据。"""
        return {
            "actor_observation_dim": self.encoding[
                "expected_actor_observation_dim"
            ],
            "critic_state_dim": self.encoding["expected_critic_state_dim"],
            "candidate_task_count": self.encoding["candidate_task_count"],
            "target_choice_count": self.encoding[
                "target_choice_count_with_idle"
            ],
            "satellite_ids": list(self.satellite_ids),
            "ground_station_ids": list(self.ground_station_ids),
        }


def decode_composite_action(encoded, target_choices, bounded_continuous_actions):
    """把4个槽的离散目标和有界连续量解码为环境复合动作。

    连续数组最后一维依次是传输比例和30秒槽内开始偏移。IDLE槽不会生成
    子动作；函数保持任务槽顺序，不替代环境的全局仲裁。
    """
    target_choices = np.asarray(target_choices)
    bounded = np.asarray(bounded_continuous_actions, dtype=float)
    if target_choices.shape != (4,) or bounded.shape != (4, 2):
        raise ValueError("动作解码需要shape=(4,)和shape=(4,2)的输入")
    if not np.all(np.isfinite(bounded)) or np.any(bounded < 0.0) or np.any(bounded > 1.0):
        raise ValueError("连续动作必须是[0,1]内的有限数")
    transmissions = []
    for slot, raw_choice in enumerate(target_choices):
        choice = int(raw_choice)
        if choice != raw_choice or not 0 <= choice <= len(encoded.target_ids):
            raise ValueError("离散目标选择超出0到19范围")
        if not encoded.base_target_mask[slot, choice]:
            raise ValueError("动作选择了基础Mask禁止的目标")
        if choice == 0:
            continue
        task_id = encoded.task_ids[slot]
        if task_id is None:
            raise RuntimeError("空任务槽不能解码为非IDLE动作")
        transmissions.append(
            TransmissionSubAction(
                task_id=task_id,
                target_id=encoded.target_ids[choice - 1],
                transmission_ratio=float(bounded[slot, 0]),
                start_offset=float(bounded[slot, 1]),
            )
        )
    return SatelliteCompositeAction(tuple(transmissions))
