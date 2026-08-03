"""提供隔离协议上的确定性MAPPO评估和完整业务指标。"""

import math

import numpy as np
import torch

from .encoding import decode_composite_action
from .evaluation_protocol import (
    public_protocol_metadata,
    sample_protocol_task_ids,
)
from .model_selection import is_better_validation_result as _central_is_better


def is_better_validation_result(candidate, incumbent, rule=None):
    """兼容旧调用的公开包装；正式选择仍委托集中规则实现。"""
    if isinstance(rule, (int, float)) and not isinstance(rule, bool):
        rule = (
            ("timeliness_raw_mean", "max", float(rule)),
            ("load_balance_mean_per_task_mean", "max", 0.0),
        )
    return _central_is_better(candidate, incumbent, rule)


_AGGREGATE_METRICS = (
    "timeliness_raw",
    "delivered_timeliness_raw",
    "load_balance_raw",
    "load_balance_mean_per_task",
    "mean_utilization_std",
    "completed_task_count",
    "expired_task_count",
    "active_task_count",
    "delivered_data_mbit",
    "accepted_subaction_count",
    "rejected_subaction_count",
    "accepted_isl_count",
    "accepted_idl_count",
    "accepted_sgl_count",
    "completion_rate",
    "expiration_rate",
    "rejected_subaction_rate",
    "sgl_action_fraction",
)


class MappoEvaluator:
    """在固定任务协议上运行确定性完整Episode，不执行梯度更新。"""

    def __init__(self, environment, encoder, actor, device):
        self.environment = environment
        self.encoder = encoder
        self.actor = actor
        self.device = torch.device(device)

    def evaluate(
        self,
        seeds=None,
        task_count=None,
        max_steps=None,
        protocol=None,
    ):
        """返回逐场景、聚合指标和协议审计信息。

        ``seeds/task_count`` 只为历史调用兼容；正式基线必须传入protocol。
        """
        if protocol is not None:
            seeds = list(protocol["seeds"])
            task_count = int(protocol["task_count"])
        else:
            seeds = list(seeds or ())
            task_count = int(task_count or 0)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("评估seeds必须非空且互不重复")
        maximum = (
            len(protocol["pool_task_ids"])
            if protocol is not None
            else len(self.environment.task_splits["validation"])
        )
        if not 0 < task_count <= maximum:
            raise ValueError("评估任务数超出任务池范围")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("评估最大步数必须为正数")
        was_training = self.actor.training
        self.actor.eval()
        scenarios = []
        try:
            with torch.no_grad():
                for seed in seeds:
                    scenarios.append(
                        self._evaluate_scenario(
                            int(seed),
                            task_count,
                            max_steps,
                            protocol,
                        )
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
            raise RuntimeError("评估聚合指标包含NaN或Inf")
        passed = all(item["data_conservation_passed"] for item in scenarios)
        result = {
            "scenarios": scenarios,
            "aggregate": aggregate,
            "all_scenarios_data_conservation_passed": passed,
        }
        if protocol is not None:
            result["protocol"] = public_protocol_metadata(protocol)
        return result

    def _evaluate_scenario(self, seed, task_count, max_steps, protocol):
        """运行单个场景；协议模式显式注入任务ID以避免跨池污染。"""
        if protocol is None:
            observations, reset_info = self.environment.reset(
                seed=seed,
                split="validation",
                task_count=task_count,
            )
        else:
            selected_ids = sample_protocol_task_ids(protocol, seed)
            observations, reset_info = self.environment.reset(
                seed=seed,
                task_ids=selected_ids,
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
        if final_info is None:
            raise RuntimeError("评估场景没有产生环境step")
        self.environment.check_data_conservation()
        accepted = totals["accepted_subaction_count"]
        rejected = totals["rejected_subaction_count"]
        attempted = accepted + rejected
        result = {
            "seed": seed,
            "selected_task_ids": list(reset_info["selected_task_ids"]),
            "environment_steps": steps,
            "full_episode": bool(self.environment.terminated),
            "timeliness_raw": float(final_info["timeliness_raw"]),
            "delivered_timeliness_raw": float(
                final_info["delivered_timeliness_raw"]
            ),
            "load_balance_raw": float(final_info["load_balance_raw"]),
            "load_balance_mean_per_task": float(
                final_info["load_balance_mean_per_task"]
            ),
            "mean_utilization_std": float(final_info["mean_utilization_std"]),
            "completed_task_count": int(final_info["completed_task_count"]),
            "expired_task_count": int(final_info["expired_task_count"]),
            "active_task_count": int(final_info["active_task_count"]),
            "delivered_data_mbit": float(final_info["delivered_data_mbit"]),
            **totals,
            "completion_rate": final_info["completed_task_count"] / task_count,
            "expiration_rate": final_info["expired_task_count"] / task_count,
            "rejected_subaction_rate": rejected / max(attempted, 1),
            "sgl_action_fraction": totals["accepted_sgl_count"] / max(accepted, 1),
            "data_conservation_passed": True,
        }
        numeric = [
            value
            for key, value in result.items()
            if key not in {
                "selected_task_ids",
                "data_conservation_passed",
                "full_episode",
            }
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("评估场景指标包含NaN或Inf")
        return result
