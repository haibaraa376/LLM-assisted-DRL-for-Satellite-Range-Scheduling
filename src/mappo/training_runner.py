"""调度任意基线奖励的完整Episode、固定验证及边界Checkpoint。"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .checkpoint import capture_rng_state, save_checkpoint
from .evaluation_protocol import (
    build_evaluation_protocol,
    public_protocol_metadata,
)
from .model_selection import is_better_validation_result
from .trainer import parameter_vector
from baselines.log_schema import (
    build_episode_log_record,
    build_update_log_record,
)


def _write_jsonl(path, record):
    """追加一行JSON并立即flush，避免长训练中日志停留在缓冲区。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


class BaselineTrainingRunner:
    """运行统一基线的train采样、更新、validation和边界保存。"""

    def __init__(
        self,
        trainer,
        evaluator,
        config,
        encoder,
        training_config=None,
        method_name="manual_mappo",
    ):
        """保存训练组件和配置；实际Episode由 ``run`` 显式启动。"""
        if trainer.auto_reset_on_done:
            raise ValueError("正式训练Trainer必须关闭Episode终点自动重置")
        self.trainer = trainer
        self.evaluator = evaluator
        self.config = config
        self.training = training_config or config["manual_training"]
        self.encoder = encoder
        self.method_name = method_name

    def _prepare_logs(self, resume):
        """新训练清空三类JSONL，Resume则保留并继续追加。"""
        paths = self.training["logging"]
        if not resume:
            for name in (
                "update_log_path",
                "episode_log_path",
                "validation_log_path",
            ):
                path = Path(paths[name])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

    def run(
        self,
        target_episode_count=None,
        start_episode_index=0,
        update_index=0,
        best_validation_result=None,
        best_episode_index=None,
        best_validation_details=None,
        skip_validation=False,
        max_steps_per_episode=None,
        validation_max_steps=None,
        resume=False,
    ):
        """训练到目标总Episode数；可用步数上限运行明确标记的缩短验收。"""
        target = int(target_episode_count or self.training["episode_count"])
        if target <= start_episode_index:
            raise ValueError("目标总Episode数必须大于恢复起始索引")
        if max_steps_per_episode is not None and max_steps_per_episode <= 0:
            raise ValueError("缩短验收步数必须为正数")
        self._prepare_logs(resume)
        actor_initial = parameter_vector(self.trainer.actor)
        critic_initial = parameter_vector(self.trainer.critic)
        total_steps_before = self.trainer.environment_steps
        episode_records = []
        protocol_name = self.training["validation"]["protocol"]
        selection_protocol = build_evaluation_protocol(
            protocol_name,
            self.training["evaluation_protocols"],
            self.trainer.environment.task_splits,
        )
        protocol_metadata = public_protocol_metadata(selection_protocol)

        for episode_index in range(start_episode_index, target):
            episode_seed = self.training["base_episode_seed"] + episode_index
            reset_info = self.trainer.reset_episode(
                episode_seed,
                "train",
                self.training["task_count"],
            )
            episode_steps = 0
            episode_updates = 0
            episode_reward_sum = 0.0
            episode_reward_min = math.inf
            episode_reward_max = -math.inf
            component_sums = {}
            component_abs_sums = {}
            action_totals = {
                "accepted_subaction_count": 0,
                "rejected_subaction_count": 0,
                "accepted_isl_count": 0,
                "accepted_idl_count": 0,
                "accepted_sgl_count": 0,
                "invalid_masked_action_count": 0,
            }
            final_info = None
            while not self.trainer.environment.terminated:
                rollout_limit = int(self.training["rollout_steps_per_update"])
                if max_steps_per_episode is not None:
                    remaining = max_steps_per_episode - episode_steps
                    if remaining <= 0:
                        break
                    rollout_limit = min(rollout_limit, remaining)
                buffer = self.trainer.collect_rollout(rollout_limit)
                statistics = self.trainer.update(buffer)
                rollout = self.trainer.last_rollout_statistics
                update_index += 1
                episode_updates += 1
                episode_steps += buffer.size
                episode_reward_sum += rollout["sum_shared_reward"]
                episode_reward_min = min(
                    episode_reward_min,
                    rollout["min_shared_reward"],
                )
                episode_reward_max = max(
                    episode_reward_max,
                    rollout["max_shared_reward"],
                )
                for name, value in rollout["reward_component_sums"].items():
                    component_sums[name] = component_sums.get(name, 0.0) + value
                for name, value in rollout[
                    "reward_component_abs_sums"
                ].items():
                    component_abs_sums[name] = (
                        component_abs_sums.get(name, 0.0) + value
                    )
                for name in action_totals:
                    action_totals[name] += int(rollout.get(name, 0))
                final_info = rollout["final_info"]
                update_record = build_update_log_record(
                    {
                        "method": self.method_name,
                        "episode_index": episode_index,
                        "update_index": update_index,
                        "environment_steps": self.trainer.environment_steps,
                        "rollout_size": buffer.size,
                        "mean_shared_reward": rollout["mean_shared_reward"],
                        "sum_shared_reward": rollout["sum_shared_reward"],
                        "reward_component_sums": rollout[
                            "reward_component_sums"
                        ],
                        "reward_component_abs_sums": rollout[
                            "reward_component_abs_sums"
                        ],
                        **statistics.__dict__,
                        "accepted_subaction_count": rollout[
                            "accepted_subaction_count"
                        ],
                        "rejected_subaction_count": rollout[
                            "rejected_subaction_count"
                        ],
                        "invalid_masked_action_count": rollout[
                            "invalid_masked_action_count"
                        ],
                    },
                    self.trainer.reward_model,
                    rollout["sum_shared_reward"],
                    rollout["reward_component_sums"],
                )
                _write_jsonl(
                    self.training["logging"]["update_log_path"],
                    update_record,
                )
                if rollout["episode_done"]:
                    break

            self.trainer.environment.check_data_conservation()
            if final_info is None:
                raise RuntimeError("训练Episode没有产生任何环境step")
            full_episode = self.trainer.environment.terminated
            episode_record = build_episode_log_record(
                {
                    "method": self.method_name,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "selected_task_ids": list(
                        reset_info["selected_task_ids"]
                    ),
                    "environment_steps": episode_steps,
                    "update_count": episode_updates,
                    "full_episode": full_episode,
                    "mean_step_reward": episode_reward_sum / episode_steps,
                    "min_step_reward": episode_reward_min,
                    "max_step_reward": episode_reward_max,
                    "reward_component_sums": component_sums,
                    "reward_component_abs_sums": component_abs_sums,
                    "timeliness_raw": float(final_info["timeliness_raw"]),
                    "delivered_timeliness_raw": float(
                        final_info["delivered_timeliness_raw"]
                    ),
                    "load_balance_raw": float(
                        final_info["load_balance_raw"]
                    ),
                    "load_balance_mean_per_task": float(
                        final_info["load_balance_mean_per_task"]
                    ),
                    "mean_utilization_std": float(
                        final_info["mean_utilization_std"]
                    ),
                    "completed_task_count": int(
                        final_info["completed_task_count"]
                    ),
                    "expired_task_count": int(
                        final_info["expired_task_count"]
                    ),
                    "active_task_count": int(
                        final_info["active_task_count"]
                    ),
                    "delivered_data_mbit": float(
                        final_info["delivered_data_mbit"]
                    ),
                    **action_totals,
                    "data_conservation_passed": True,
                },
                self.trainer.reward_model,
                episode_reward_sum,
                component_sums,
            )
            if (
                self.trainer.reward_model.log_metadata.reward_method
                == "manual_plus_lyapunov"
            ):
                episode_record.update(
                    {
                        "manual_reward_sum": component_sums["manual_reward"],
                        "lyapunov_shaping_sum": component_sums.get(
                            "lyapunov_shaping",
                            0.0,
                        ),
                        "backlog_mean": component_sums.get("backlog", 0.0)
                        / episode_steps,
                        "expiration_risk_mean": component_sums.get(
                            "expiration_risk",
                            0.0,
                        )
                        / episode_steps,
                        "expired_undelivered_mean": component_sums.get(
                            "expired_undelivered",
                            0.0,
                        )
                        / episode_steps,
                        "utilization_imbalance_mean": component_sums.get(
                            "utilization_imbalance",
                            0.0,
                        )
                        / episode_steps,
                        "potential_start": getattr(
                            self.trainer.reward_model,
                            "episode_initial_potential",
                            None,
                        ),
                        "potential_end": getattr(
                            self.trainer.reward_model,
                            "episode_current_potential",
                            None,
                        ),
                    }
                )
            self._check_finite_record(episode_record)
            _write_jsonl(
                self.training["logging"]["episode_log_path"],
                episode_record,
            )
            episode_records.append(episode_record)

            validation_result = None
            is_new_best = False
            should_validate = (
                not skip_validation
                and (episode_index + 1)
                % int(self.training["validation_interval_episodes"])
                == 0
            )
            if should_validate:
                validation_result = self.evaluator.evaluate(
                    max_steps=validation_max_steps,
                    protocol=selection_protocol,
                )
                candidate = validation_result["aggregate"]
                is_new_best = is_better_validation_result(
                    candidate,
                    best_validation_result,
                    self.training["best_model_rule"]["metrics"],
                )
                validation_record = {
                    "episode_index": episode_index,
                    "update_index": update_index,
                    **validation_result,
                    "is_new_best": is_new_best,
                }
                _write_jsonl(
                    self.training["logging"]["validation_log_path"],
                    validation_record,
                )
                if is_new_best:
                    best_validation_result = candidate
                    best_episode_index = episode_index
                    best_validation_details = validation_result

            training_state = {
                "episode_index": episode_index,
                "next_episode_index": episode_index + 1,
                "update_index": update_index,
                "environment_steps": self.trainer.environment_steps,
                "best_validation_result": best_validation_result,
                "best_episode_index": best_episode_index,
                "best_validation_details": best_validation_details,
                "evaluation_protocol": protocol_metadata,
            }
            rng_state = capture_rng_state(self.trainer.rng)
            if is_new_best:
                self._save_boundary_checkpoint(
                    self.training["checkpoint"]["best_path"],
                    update_index,
                    training_state,
                    rng_state,
                    validation_result,
                )
            self._save_boundary_checkpoint(
                self.training["checkpoint"]["last_path"],
                update_index,
                training_state,
                rng_state,
                validation_result,
            )

        actor_change = float(
            torch.linalg.vector_norm(
                parameter_vector(self.trainer.actor) - actor_initial
            )
        )
        critic_change = float(
            torch.linalg.vector_norm(
                parameter_vector(self.trainer.critic) - critic_initial
            )
        )
        summary = {
            "method": self.method_name,
            "device": str(self.trainer.device),
            "start_episode_index": start_episode_index,
            "target_episode_count": target,
            "episodes_run": len(episode_records),
            "environment_steps_this_run": (
                self.trainer.environment_steps - total_steps_before
            ),
            "total_update_index": update_index,
            "actor_parameter_change_norm": actor_change,
            "critic_parameter_change_norm": critic_change,
            "best_episode_index": best_episode_index,
            "best_validation_result": best_validation_result,
            "best_validation_scenarios": (
                best_validation_details.get("scenarios", [])
                if best_validation_details
                else []
            ),
            "best_validation_data_conservation": (
                best_validation_details.get(
                    "all_scenarios_data_conservation_passed",
                    False,
                )
                if best_validation_details
                else False
            ),
            "shortened_acceptance_mode": max_steps_per_episode is not None,
            "data_conservation_passed": True,
            "evaluation_protocol": protocol_metadata,
            "reward_diagnostics": self._reward_diagnostics(episode_records),
            "llm_reward_weight_metadata": getattr(
                self.trainer.reward_model,
                "weight_metadata",
                None,
            ),
        }
        Path(self.training["logging"]["summary_path"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _reward_diagnostics(episode_records):
        """汇总八个带权奖励分量的绝对贡献及最大单项支配比例。"""
        weighted_names = (
            "weighted_sgl_progress",
            "weighted_relay_progress",
            "weighted_completion",
            "weighted_balance",
            "weighted_expiration",
            "weighted_invalid_action",
            "weighted_coordination_conflict",
            "weighted_relay_cost",
        )
        totals = {name: 0.0 for name in weighted_names}
        for record in episode_records:
            absolute = record.get("reward_component_abs_sums", {})
            for name in weighted_names:
                totals[name] += float(absolute.get(name, 0.0))
        absolute_total = sum(totals.values())
        dominance = (
            max(totals.values()) / absolute_total
            if absolute_total > 0.0
            else 0.0
        )
        return {
            "weighted_component_abs_sums": totals,
            "weighted_component_abs_total": absolute_total,
            "maximum_single_component_dominance": dominance,
        }

    def _save_boundary_checkpoint(
        self,
        path,
        update_index,
        training_state,
        rng_state,
        validation_result,
    ):
        """只在Episode边界保存正式训练状态。"""
        save_checkpoint(
            path,
            self.trainer.actor,
            self.trainer.critic,
            self.trainer.actor_optimizer,
            self.trainer.critic_optimizer,
            self.config,
            update_index,
            self.encoder.metadata(),
            training_state=training_state,
            rng_state=rng_state,
            validation_result=validation_result,
        )

    @staticmethod
    def _check_finite_record(record):
        """递归检查Episode统计中的数值均为有限值。"""
        def values(item):
            if isinstance(item, dict):
                for value in item.values():
                    yield from values(value)
            elif isinstance(item, list):
                return
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                yield item

        if not all(math.isfinite(value) for value in values(record)):
            raise RuntimeError("训练Episode统计包含NaN或Inf")
