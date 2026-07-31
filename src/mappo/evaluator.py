"""提供固定validation场景的确定性MAPPO评估与Best选择规则。"""

import math

import numpy as np
import torch

from .encoding import decode_composite_action


_AGGREGATE_METRICS = (
    "timeliness_raw",
    "load_balance_mean_per_task",
    "completed_task_count",
    "expired_task_count",
    "delivered_data_mbit",
    "accepted_subaction_count",
    "rejected_subaction_count",
)


def is_better_validation_result(candidate, incumbent, primary_tolerance):
    """按及时性主指标和负载均衡次指标判断候选是否更优。"""
    if primary_tolerance < 0 or not math.isfinite(primary_tolerance):
        raise ValueError("Best模型主指标容差必须是非负有限数")
    primary = float(candidate["timeliness_raw_mean"])
    secondary = float(candidate["load_balance_mean_per_task_mean"])
    if not math.isfinite(primary) or not math.isfinite(secondary):
        raise ValueError("候选validation指标包含NaN或Inf")
    if incumbent is None:
        return True
    incumbent_primary = float(incumbent["timeliness_raw_mean"])
    incumbent_secondary = float(
        incumbent["load_balance_mean_per_task_mean"]
    )
    if not math.isfinite(incumbent_primary) or not math.isfinite(
        incumbent_secondary
    ):
        raise ValueError("已有Best validation指标包含NaN或Inf")
    difference = primary - incumbent_primary
    if difference > primary_tolerance:
        return True
    if difference < -primary_tolerance:
        return False
    return secondary > incumbent_secondary


class MappoEvaluator:
    """在固定validation任务上运行确定性完整Episode，不执行梯度更新。"""

    def __init__(self, environment, encoder, actor, device):
        """保存环境、编码器、共享Actor和推理设备，不重置任何状态。"""
        self.environment = environment
        self.encoder = encoder
        self.actor = actor
        self.device = torch.device(device)

    def evaluate(self, seeds, task_count, max_steps=None):
        """评估多个互异validation seed并返回逐场景与聚合原始指标。"""
        seeds = list(seeds)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("validation seeds必须非空且互不重复")
        if not 0 < task_count <= 150:
            raise ValueError("validation任务数必须位于1到150")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("评估最大步数必须为正数")
        was_training = self.actor.training
        self.actor.eval()
        scenarios = []
        try:
            with torch.no_grad():
                for seed in seeds:
                    scenarios.append(
                        self._evaluate_scenario(seed, task_count, max_steps)
                    )
        finally:
            self.actor.train(was_training)
        aggregate = {}
        for metric in _AGGREGATE_METRICS:
            values = np.asarray(
                [scenario[metric] for scenario in scenarios],
                dtype=float,
            )
            aggregate[metric + "_mean"] = float(values.mean())
            aggregate[metric + "_std"] = float(values.std())
            aggregate[metric + "_min"] = float(values.min())
            aggregate[metric + "_max"] = float(values.max())
        if not all(math.isfinite(value) for value in aggregate.values()):
            raise RuntimeError("validation聚合指标包含NaN或Inf")
        return {"scenarios": scenarios, "aggregate": aggregate}

    def _evaluate_scenario(self, seed, task_count, max_steps):
        """运行一个validation场景，默认走完2880个决策步。"""
        observations, reset_info = self.environment.reset(
            seed=int(seed),
            split="validation",
            task_count=int(task_count),
        )
        totals = {
            "accepted_subaction_count": 0,
            "rejected_subaction_count": 0,
            "accepted_isl_count": 0,
            "accepted_idl_count": 0,
            "accepted_sgl_count": 0,
        }
        final_info = None
        steps = 0
        while not self.environment.terminated:
            encoded = self.encoder.encode_all_agents(observations)
            actor_observations = np.stack(
                [
                    encoded[satellite_id].observation
                    for satellite_id in self.encoder.satellite_ids
                ]
            )
            masks = np.stack(
                [
                    encoded[satellite_id].base_target_mask
                    for satellite_id in self.encoder.satellite_ids
                ]
            )
            batch = self.actor.sample_actions(
                torch.as_tensor(
                    actor_observations,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(masks, dtype=torch.bool, device=self.device),
                deterministic=True,
            )
            choices = batch.target_choices.cpu().numpy()
            continuous = batch.bounded_continuous_actions.cpu().numpy()
            actions = {
                satellite_id: decode_composite_action(
                    encoded[satellite_id],
                    choices[index],
                    continuous[index],
                )
                for index, satellite_id in enumerate(self.encoder.satellite_ids)
            }
            observations, _, _, _, final_info = self.environment.step(actions)
            for name in totals:
                totals[name] += int(final_info[name])
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        self.environment.check_data_conservation()
        result = {
            "seed": int(seed),
            "selected_task_ids": list(reset_info["selected_task_ids"]),
            "environment_steps": steps,
            "timeliness_raw": float(final_info["timeliness_raw"]),
            "load_balance_raw": float(final_info["load_balance_raw"]),
            "load_balance_mean_per_task": float(
                final_info["load_balance_mean_per_task"]
            ),
            "mean_utilization_std": float(final_info["mean_utilization_std"]),
            "completed_task_count": int(final_info["completed_task_count"]),
            "expired_task_count": int(final_info["expired_task_count"]),
            "delivered_data_mbit": float(final_info["delivered_data_mbit"]),
            **totals,
            "data_conservation_passed": True,
        }
        numeric = [
            value
            for key, value in result.items()
            if key not in {"selected_task_ids", "data_conservation_passed"}
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("validation场景指标包含NaN或Inf")
        return result
