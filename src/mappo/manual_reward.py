"""实现可复用的人工奖励特征提取与权重组合。"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from srs_env.models import TaskStatus

from .reward_metadata import RewardLogMetadata


@dataclass(frozen=True)
class RewardFeatures:
    """描述一个环境时隙的八个人工奖励特征。"""

    sgl_progress: float
    relay_progress: float
    completion_score: float
    balance_score: float
    expiration_loss: float
    invalid_action_rate: float
    coordination_conflict_rate: float
    relay_cost: float

    def __post_init__(self):
        """确保全部特征可安全写入JSON且不含NaN或Inf。"""
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("人工奖励特征包含NaN或Inf")


@dataclass(frozen=True)
class RewardBreakdown:
    """保存奖励特征、八个带符号贡献及最终团队奖励。"""

    features: RewardFeatures
    weighted_sgl_progress: float
    weighted_relay_progress: float
    weighted_completion: float
    weighted_balance: float
    weighted_expiration: float
    weighted_invalid_action: float
    weighted_coordination_conflict: float
    weighted_relay_cost: float
    total_reward: float

    def __post_init__(self):
        """拒绝任何非有限加权项。"""
        values = [
            value
            for name, value in asdict(self).items()
            if name != "features"
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("人工奖励加权项包含NaN或Inf")

    def as_dict(self):
        """返回只含Python标量和字典的JSON可序列化结构。"""
        return asdict(self)

    def component_values(self):
        """展开原始特征和加权贡献，供统一训练日志累计。"""
        return {
            **asdict(self.features),
            **{
                name: value
                for name, value in asdict(self).items()
                if name != "features"
            },
        }


@dataclass(frozen=True)
class RewardStateSnapshot:
    """保存时隙执行前任务状态、送达量和发送时长的独立副本。"""

    task_statuses: dict
    delivered_mbit: dict
    outgoing_seconds: np.ndarray


def make_reward_snapshot(environment):
    """从环境创建只读语义快照，不引用可变的发送时长数组。"""
    return RewardStateSnapshot(
        task_statuses={
            task_id: state.status.value
            for task_id, state in environment.tasks.items()
        },
        delivered_mbit={
            task_id: float(state.delivered_to_ground_mbit)
            for task_id, state in environment.tasks.items()
        },
        outgoing_seconds=environment.outgoing_seconds.copy(),
    )


def _progress_value(environment, record, maximum_priority):
    """计算一条成功记录的数据比例、优先级和剩余寿命乘积。"""
    if record.task_id not in environment.tasks:
        raise RuntimeError("成功传输记录引用了未知任务")
    state = environment.tasks[record.task_id]
    task = state.definition
    if record.actual_start_s is None or not math.isfinite(record.actual_start_s):
        raise RuntimeError("成功传输记录缺少有限的实际开始时刻")
    if record.transmitted_data_mbit <= 0 or not math.isfinite(
        record.transmitted_data_mbit
    ):
        raise RuntimeError("成功传输记录的数据量必须是正有限数")
    if record.link_type == "SGL" and record.target_id != task.target_ground_station_id:
        raise RuntimeError("成功SGL记录的地面站与任务目标不一致")
    data_ratio = record.transmitted_data_mbit / task.data_size_mbit
    priority_ratio = task.priority / maximum_priority
    survival_ratio = np.clip(
        (task.expiration_time_s - record.actual_start_s) / task.survival_time_s,
        0.0,
        1.0,
    )
    return float(data_ratio * priority_ratio * survival_ratio)


def _balance_score(environment, accepted_records, snapshot, epsilon):
    """基于执行前任务利用率计算欠载发送塑形，无传输时返回0。"""
    if not accepted_records:
        return 0.0
    scores = []
    for record in accepted_records:
        if record.task_id not in environment.task_index:
            raise RuntimeError("均衡塑形记录引用了未知任务")
        task_row = environment.task_index[record.task_id]
        source_index = environment.dataset.satellite_index.get(
            record.source_satellite_id
        )
        if source_index is None:
            raise RuntimeError("均衡塑形记录引用了未知源卫星")
        utilization = np.divide(
            snapshot.outgoing_seconds[task_row],
            environment.total_window_seconds,
            out=np.zeros_like(snapshot.outgoing_seconds[task_row]),
            where=environment.total_window_seconds > 0.0,
        )
        mean_utilization = float(utilization.mean())
        standard_deviation = float(utilization.std())
        score = (mean_utilization - utilization[source_index]) / (
            standard_deviation + epsilon
        )
        scores.append(float(np.clip(score, -1.0, 1.0)))
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def extract_reward_features(environment, info, previous_snapshot, config):
    """从环境、传输记录和时隙前快照提取八个奖励特征。

    函数只读环境并返回新快照。全局竞争不会混入明显无效动作类别；未知违反
    代码会立即失败，避免奖励日志静默漏计。
    """
    if previous_snapshot is None:
        raise RuntimeError("提取人工奖励前必须先创建Episode快照")
    normalization = config["normalization"]
    tolerance = config["numerical"]["tolerance"]
    records = list(info.get("transmission_records", ()))
    submitted = int(info.get("submitted_subaction_count", len(records)))
    accepted = [record for record in records if record.accepted]
    sgl_sum = 0.0
    relay_sum = 0.0
    relay_cost_sum = 0.0
    for record in accepted:
        if record.link_type not in {"SGL", "ISL", "IDL"}:
            raise RuntimeError("成功传输记录包含未知链路类型")
        progress = _progress_value(
            environment,
            record,
            normalization["max_priority"],
        )
        if record.link_type == "SGL":
            sgl_sum += progress
        else:
            relay_sum += progress
            task_size = environment.tasks[
                record.task_id
            ].definition.data_size_mbit
            relay_cost_sum += record.transmitted_data_mbit / task_size

    completion_sum = 0.0
    expiration_sum = 0.0
    for task_id, state in environment.tasks.items():
        previous_status = previous_snapshot.task_statuses[task_id]
        current_status = state.status.value
        task = state.definition
        if (
            previous_status != TaskStatus.COMPLETED.value
            and current_status == TaskStatus.COMPLETED.value
        ):
            if state.delivered_to_ground_mbit < task.data_size_mbit - tolerance:
                raise RuntimeError("任务标记完成但送达数据量不足")
            completion_sum += task.priority / normalization["max_priority"]
        if (
            previous_status != TaskStatus.EXPIRED.value
            and current_status == TaskStatus.EXPIRED.value
        ):
            undelivered_ratio = np.clip(
                (task.data_size_mbit - state.delivered_to_ground_mbit)
                / task.data_size_mbit,
                0.0,
                1.0,
            )
            expiration_sum += (
                task.priority / normalization["max_priority"]
            ) * undelivered_ratio

    invalid_codes = set(config["invalid_violation_codes"])
    coordination_codes = set(config["coordination_violation_codes"])
    known_codes = invalid_codes | coordination_codes
    invalid_count = 0
    coordination_count = 0
    for record in records:
        if record.accepted:
            continue
        codes = set(record.violation_codes)
        unknown = codes - known_codes
        if unknown:
            raise RuntimeError(
                "人工奖励遇到未知违反代码：{0}".format(sorted(unknown))
            )
        invalid_count += bool(codes & invalid_codes)
        coordination_count += bool(codes & coordination_codes)

    denominator = max(submitted, 1)
    features = RewardFeatures(
        sgl_progress=sgl_sum / normalization["sgl_parallel_reference"],
        relay_progress=relay_sum / normalization["relay_parallel_reference"],
        completion_score=completion_sum
        / normalization["completion_parallel_reference"],
        balance_score=_balance_score(
            environment,
            accepted,
            previous_snapshot,
            normalization["balance_epsilon"],
        ),
        expiration_loss=float(
            np.clip(
                expiration_sum
                / normalization["expiration_parallel_reference"],
                0.0,
                1.0,
            )
        ),
        invalid_action_rate=float(np.clip(invalid_count / denominator, 0.0, 1.0)),
        coordination_conflict_rate=float(
            np.clip(coordination_count / denominator, 0.0, 1.0)
        ),
        relay_cost=relay_cost_sum / normalization["relay_parallel_reference"],
    )
    return features, make_reward_snapshot(environment)


def combine_manual_reward(features, weights, numerical=None):
    """仅根据RewardFeatures与配置权重组合共享团队奖励。"""
    weighted = {
        "weighted_sgl_progress": weights["sgl_progress"] * features.sgl_progress,
        "weighted_relay_progress": weights["relay_progress"]
        * features.relay_progress,
        "weighted_completion": weights["completion"] * features.completion_score,
        "weighted_balance": weights["balance"] * features.balance_score,
        "weighted_expiration": -weights["expiration"] * features.expiration_loss,
        "weighted_invalid_action": -weights["invalid_action"]
        * features.invalid_action_rate,
        "weighted_coordination_conflict": -weights["coordination_conflict"]
        * features.coordination_conflict_rate,
        "weighted_relay_cost": -weights["relay_cost"] * features.relay_cost,
    }
    total = float(sum(weighted.values()))
    weighted_values_are_finite = all(
        math.isfinite(value) for value in weighted.values()
    )
    if not weighted_values_are_finite or not math.isfinite(total):
        raise ValueError("人工奖励贡献或总奖励包含NaN/Inf")
    if numerical is not None and abs(total) > numerical["hard_failure_abs_reward"]:
        raise RuntimeError("人工奖励绝对值超过硬失败阈值")
    return RewardBreakdown(features=features, total_reward=total, **weighted)


class ManualReward:
    """维护Episode快照并计算可供MAPPO使用的共享人工奖励。"""

    def __init__(self, config):
        """接收 ``manual_reward`` 配置，不读取或修改环境。"""
        self.config = config
        self.previous_snapshot = None
        self.warning_count = 0
        self.last_breakdown = None

    def reset(self, environment):
        """在Episode重置后保存初始快照并清零警告计数。"""
        self.previous_snapshot = make_reward_snapshot(environment)
        self.warning_count = 0
        self.last_breakdown = None

    def compute(self, environment, info):
        """在环境step后提取特征、组合权重并推进内部快照。"""
        features, next_snapshot = extract_reward_features(
            environment,
            info,
            self.previous_snapshot,
            self.config,
        )
        breakdown = combine_manual_reward(
            features,
            self.config["weights"],
            self.config["numerical"],
        )
        if abs(breakdown.total_reward) > self.config["numerical"][
            "warning_abs_reward"
        ]:
            self.warning_count += 1
        self.previous_snapshot = next_snapshot
        self.last_breakdown = breakdown
        return breakdown

    @property
    def log_metadata(self):
        """声明人工奖励没有额外塑形项。"""
        return RewardLogMetadata(
            reward_method="manual_reward",
            base_reward_name="manual_reward",
            shaping_reward_name=None,
        )
    environment_aware = True
