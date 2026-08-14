"""调度任意基线奖励的完整Episode、固定验证及边界Checkpoint。"""

import csv
import json
import math
from pathlib import Path
import shutil

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
        # 早期手工 MAPPO 测试可显式跳过验证，且其旧配置没有协议字段。
        selection_protocol = None
        protocol_metadata = None
        if not skip_validation:
            protocol_name = self.training["validation"]["protocol"]
            selection_protocol = build_evaluation_protocol(
                protocol_name,
                self.training["evaluation_protocols"],
                self.trainer.environment.task_splits,
            )
            protocol_metadata = public_protocol_metadata(selection_protocol)

        training_seed = self.training.get("training_seed")
        if training_seed is None:
            training_seed = self.training["base_episode_seed"]
        for episode_index in range(start_episode_index, target):
            # 正式基线使用training_seed；旧manual_training配置沿用其同义字段，
            # 二者均固定为同一训练场景，不随Episode递增。
            episode_seed = training_seed
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
                    self.training["best_model_rule"],
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
                # 候选身份和有效权重在 config 中冻结，便于单独恢复阶段状态。
                "candidate_id": self.config.get("candidate_id"),
                "reward_spec_id": self.config.get("reward_spec_id"),
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
            stage_path = None
            # 只有完整 Episode、守恒检查和当次日志完成后才保存正式阶段状态。
            if (
                full_episode
                and self.training.get("save_episode_checkpoints", False)
            ):
                stage_path = Path(self.training["checkpoint"]["stage_directory"]) / (
                    "episode_{0:04d}.pt".format(episode_index + 1)
                )
                if stage_path.exists():
                    raise FileExistsError("阶段Checkpoint已存在，拒绝覆盖：{0}".format(stage_path))
                self._save_boundary_checkpoint(
                    stage_path,
                    update_index,
                    training_state,
                    rng_state,
                    validation_result,
                )
            episode_record["checkpoint_path"] = str(stage_path) if stage_path else None
            episode_record["official_experiment"] = max_steps_per_episode is None
            episode_record["validation"] = (
                validation_result["aggregate"] if validation_result else None
            )
            episode_record["is_new_candidate_best"] = is_new_best

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
            # ORSO分段续训需要保留最佳验证细节，避免后续阶段把守恒状态误判为缺失。
            "best_validation_details": best_validation_details,
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
        curve_paths = self._write_learning_curves(episode_records, resume=resume)
        summary["episode_checkpoint_paths"] = [
            record["checkpoint_path"] for record in episode_records
            if record.get("checkpoint_path")
        ]
        summary["learning_curve_paths"] = curve_paths
        Path(self.training["logging"]["summary_path"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def _write_learning_curves(self, episode_records, resume=False):
        """从已写出的训练/验证记录生成曲线，绝不重新运行环境。"""
        if not self.training.get("write_learning_curves", False):
            return {}
        root = Path(self.training["logging"]["summary_path"]).parent
        json_path = root / "learning_curve.json"
        csv_path, png_path = root / "learning_curve.csv", root / "learning_curve.png"
        points = []
        if resume and json_path.is_file():
            points = json.loads(json_path.read_text(encoding="utf-8"))
        elif resume and csv_path.is_file():
            # 完成Run会删除详细JSON；续训时从保留的精简CSV恢复曲线历史。
            points = self._points_from_compact_curve(csv_path)
        for record in episode_records:
            point = {
                key: record.get(key)
                for key in (
                    "episode_index", "episode_seed", "environment_steps", "update_count",
                    "mean_step_reward", "reward_component_sums", "reward_component_abs_sums",
                    "maximum_single_component_dominance", "accepted_subaction_count",
                    "rejected_subaction_count", "accepted_isl_count", "accepted_idl_count",
                    "accepted_sgl_count", "timeliness_raw", "delivered_timeliness_raw",
                    "completed_task_count", "expired_task_count", "delivered_data_mbit",
                    "load_balance_mean_per_task", "checkpoint_path", "full_episode",
                    "data_conservation_passed", "official_experiment", "is_new_candidate_best",
                )
            }
            validation = record.get("validation") or {}
            # 保留原始每 Episode validation，候选筛选可直接做尾部窗口聚合。
            point["validation"] = validation or None
            for name in (
                "delivered_timeliness_raw", "completion_rate", "expiration_rate",
                "delivered_data_mbit", "rejected_subaction_rate",
                "load_balance_mean_per_task", "sgl_action_fraction", "accepted_sgl_count",
            ):
                point[name + "_mean"] = validation.get(name + "_mean")
                point[name + "_std"] = validation.get(name + "_std")
            points.append(point)
        json_path.write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
        # CSV 将嵌套奖励分量转为 JSON 字符串，以保留完整八项分量且便于表格查看。
        columns = sorted({key for point in points for key in point})
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for point in points:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in point.items()})
        self._plot_learning_curve(points, png_path)
        return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path)}

    @staticmethod
    def _points_from_compact_curve(csv_path):
        """把完成Run保留的简洁曲线还原为可追加的内部点。"""
        metric_map = {
            "completion": "completion_rate",
            "expiration": "expiration_rate",
            "timeliness": "delivered_timeliness_raw",
            "data_mbit": "delivered_data_mbit",
            "reject_rate": "rejected_subaction_rate",
            "balance": "load_balance_mean_per_task",
            "sgl_fraction": "sgl_action_fraction",
        }
        points = []
        with Path(csv_path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                point = {
                    "episode_index": int(row["episode"]) - 1,
                    "episode_seed": int(row["seed"]),
                    "environment_steps": int(row["environment_steps"]),
                    "update_count": int(row["updates"]),
                    "mean_step_reward": float(row["mean_step_reward"]),
                    "accepted_isl_count": int(row["accepted_isl"]),
                    "accepted_idl_count": int(row["accepted_idl"]),
                    "accepted_sgl_count": int(row["accepted_sgl"]),
                    "full_episode": row["full_episode"].lower() == "true",
                    "data_conservation_passed": (
                        row["data_conservation_passed"].lower() == "true"
                    ),
                }
                for prefix, metric in metric_map.items():
                    point[metric + "_mean"] = float(row[prefix + "_mean"])
                    point[metric + "_std"] = float(row[prefix + "_std"])
                points.append(point)
        return points

    @staticmethod
    def compact_completed_artifacts(output_directory):
        """压缩已完成方法的详细日志，保留可读曲线、关键摘要和恢复Checkpoint。"""
        root = Path(output_directory)
        source = root / "learning_curve.json"
        if source.is_file():
            points = json.loads(source.read_text(encoding="utf-8"))
            columns = (
                "episode", "seed", "environment_steps", "updates",
                "mean_step_reward", "completion_mean", "completion_std",
                "expiration_mean", "expiration_std", "timeliness_mean",
                "timeliness_std", "data_mbit_mean", "data_mbit_std",
                "reject_rate_mean", "reject_rate_std", "balance_mean",
                "balance_std", "sgl_fraction_mean", "sgl_fraction_std",
                "accepted_isl", "accepted_idl", "accepted_sgl",
                "full_episode", "data_conservation_passed",
            )
            with (root / "learning_curve.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for point in points:
                    writer.writerow({
                        "episode": int(point["episode_index"]) + 1,
                        "seed": point["episode_seed"],
                        "environment_steps": point["environment_steps"],
                        "updates": point["update_count"],
                        "mean_step_reward": point["mean_step_reward"],
                        "completion_mean": point["completion_rate_mean"],
                        "completion_std": point["completion_rate_std"],
                        "expiration_mean": point["expiration_rate_mean"],
                        "expiration_std": point["expiration_rate_std"],
                        "timeliness_mean": point["delivered_timeliness_raw_mean"],
                        "timeliness_std": point["delivered_timeliness_raw_std"],
                        "data_mbit_mean": point["delivered_data_mbit_mean"],
                        "data_mbit_std": point["delivered_data_mbit_std"],
                        "reject_rate_mean": point["rejected_subaction_rate_mean"],
                        "reject_rate_std": point["rejected_subaction_rate_std"],
                        "balance_mean": point["load_balance_mean_per_task_mean"],
                        "balance_std": point["load_balance_mean_per_task_std"],
                        "sgl_fraction_mean": point["sgl_action_fraction_mean"],
                        "sgl_fraction_std": point["sgl_action_fraction_std"],
                        "accepted_isl": point["accepted_isl_count"],
                        "accepted_idl": point["accepted_idl_count"],
                        "accepted_sgl": point["accepted_sgl_count"],
                        "full_episode": point["full_episode"],
                        "data_conservation_passed": point["data_conservation_passed"],
                    })
        summary_path = root / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            keep = (
                "method", "device", "start_episode_index", "target_episode_count",
                "episodes_run", "environment_steps_this_run", "total_update_index",
                "actor_parameter_change_norm", "critic_parameter_change_norm",
                "best_episode_index", "best_validation_result",
                "best_validation_data_conservation", "shortened_acceptance_mode",
                "data_conservation_passed", "evaluation_protocol", "reward_diagnostics",
                "llm_reward_weight_metadata",
            )
            compact_summary = {name: summary[name] for name in keep if name in summary}
            summary_path.write_text(
                json.dumps(compact_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        # 这些文件包含逐任务、逐更新或完整验证场景，不是完成训练后的必要结果。
        for name in (
            "learning_curve.json", "episodes.jsonl", "train_updates.jsonl",
            "validation.jsonl",
        ):
            path = root / name
            if path.exists():
                path.unlink()
        stage_directory = root / "checkpoints"
        if stage_directory.exists():
            shutil.rmtree(stage_directory)

    @staticmethod
    def _plot_learning_curve(points, path):
        """以无交互后端输出核心验证指标图，适合服务器与 Windows 命令行。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as error:
            raise RuntimeError("学习曲线需要matplotlib，请安装项目依赖") from error
        episodes = [point["episode_index"] + 1 for point in points]
        metrics = ("completion_rate_mean", "expiration_rate_mean", "delivered_timeliness_raw_mean", "delivered_data_mbit_mean")
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for axis, metric in zip(axes.flat, metrics):
            values = [point.get(metric) for point in points]
            axis.plot(episodes, values, marker="o")
            axis.set_title(metric)
            axis.set_xlabel("Episode")
            axis.grid(alpha=0.3)
        figure.savefig(path, dpi=150)
        plt.close(figure)

    @staticmethod
    def _reward_diagnostics(episode_records):
        """汇总基础与LLM塑形的实际绝对贡献，供候选资格检查使用。"""
        base_names = (
            "weighted_sgl_progress",
            "weighted_relay_progress",
            "weighted_completion",
            "weighted_balance",
            "weighted_expiration",
            "weighted_invalid_action",
            "weighted_coordination_conflict",
            "weighted_relay_cost",
        )
        llm_names = tuple("llm_" + name for name in (
            "sgl_progress", "relay_progress", "completion", "balance",
            "expiration", "invalid_action", "coordination_conflict", "relay_cost",
        ))
        # 人工/PPO-Lya历史记录没有LLM分量；一旦出现任一LLM字段则必须完整，
        # 不能把漏记的塑形贡献静默当成零。
        has_llm_components = any(
            name.startswith("llm_")
            for record in episode_records
            for name in record.get("reward_component_abs_sums", {})
        )
        names = base_names + llm_names if has_llm_components else base_names
        totals = {name: 0.0 for name in names}
        for record in episode_records:
            absolute = record.get("reward_component_abs_sums", {})
            for name in totals:
                if name not in absolute:
                    raise ValueError("奖励贡献日志缺少必要字段：{0}".format(name))
                totals[name] += float(absolute[name])
        base_abs_sum = sum(totals[name] for name in base_names)
        llm_abs_sum = sum(totals[name] for name in llm_names if name in totals)
        absolute_total = base_abs_sum + llm_abs_sum
        dominance = (
            max(totals.values()) / absolute_total
            if absolute_total > 0.0
            else 0.0
        )
        component_shares = {
            name: value / absolute_total if absolute_total > 0.0 else 0.0
            for name, value in totals.items()
        }
        dominant_component = (
            max(component_shares, key=component_shares.get)
            if component_shares
            else None
        )
        return {
            "weighted_component_abs_sums": {
                name: totals[name] for name in base_names
            },
            "llm_component_abs_sums": {
                name: totals[name] for name in llm_names if name in totals
            },
            "weighted_component_abs_total": absolute_total,
            "base_abs_sum": base_abs_sum,
            "llm_abs_sum": llm_abs_sum,
            "total_abs_sum": absolute_total,
            "llm_contribution_ratio": (
                llm_abs_sum / absolute_total if absolute_total > 0.0 else 0.0
            ),
            "component_shares": component_shares,
            "dominant_component": dominant_component,
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
