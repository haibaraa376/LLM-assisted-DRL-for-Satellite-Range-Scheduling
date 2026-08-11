"""运行固定任务的人工奖励 MAPPO 诊断实验。

该入口只用于比较人工奖励权重，不改变网络、PPO 超参数或环境。训练过程
复用统一 Runner；结束时将其临时 JSONL 日志压缩为诊断所需的三个 CSV。
"""

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import shutil

from baselines.baseline_runner import (
    build_baseline_components,
    build_training_config,
)
from baselines.config import load_baseline_config
from baselines.methods import BaselineMethod
from mappo.config import load_mappo_config
from mappo.training_runner import BaselineTrainingRunner


REWARD_PROFILES = {
    "no_conflict": {
        "sgl_progress": 1.00,
        "relay_progress": 0.15,
        "completion": 0.50,
        "balance": 0.05,
        "expiration": 0.50,
        "invalid_action": 0.10,
        "coordination_conflict": 0.00,
        "relay_cost": 0.02,
    },
    "balanced": {
        "sgl_progress": 1.00,
        "relay_progress": 0.15,
        "completion": 1.00,
        "balance": 0.05,
        "expiration": 0.50,
        "invalid_action": 0.10,
        "coordination_conflict": 0.004,
        "relay_cost": 0.02,
    },
}

COMPONENT_COLUMNS = (
    ("weighted_sgl_progress", "sgl_progress"),
    ("weighted_relay_progress", "relay_progress"),
    ("weighted_completion", "completion"),
    ("weighted_balance", "balance"),
    ("weighted_expiration", "expiration"),
    ("weighted_invalid_action", "invalid_action"),
    ("weighted_coordination_conflict", "coordination_conflict"),
    ("weighted_relay_cost", "relay_cost"),
)

TEMPORARY_ARTIFACTS = (
    ".episodes.jsonl",
    ".updates.jsonl",
    ".validation.jsonl",
    ".summary.json",
)


def parse_args():
    """解析两个诊断实验共同使用的少量参数。"""
    parser = argparse.ArgumentParser(description="固定任务人工奖励MAPPO诊断")
    parser.add_argument(
        "--reward-profile",
        required=True,
        choices=tuple(REWARD_PROFILES),
        help="no_conflict 或 balanced。",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--task-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--fixed-task-seed", type=int, default=2025)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--validation-max-steps", type=int)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    return parser.parse_args()


def _read_jsonl(path):
    """读取Runner的临时日志；缺行直接报错而不制造诊断数据。"""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    if not records:
        raise RuntimeError("诊断临时日志为空：{0}".format(path))
    return records


def _require(record, field, record_name):
    """读取诊断必需字段；缺失时明确指出来源记录。"""
    if field not in record:
        raise ValueError("{0}缺少必要字段：{1}".format(record_name, field))
    return record[field]


def write_diagnostic_csvs(output_directory, episode_records, update_records, task_count):
    """把临时统一日志转换为实验要求的三个精简 CSV。"""
    root = Path(output_directory)
    if task_count <= 0:
        raise ValueError("task_count必须为正数")

    with (root / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        columns = (
            "episode",
            "completion_rate",
            "expiration_rate",
            "delivered_timeliness_raw",
            "delivered_data_mbit",
            "episode_reward",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in episode_records:
            writer.writerow(
                {
                    "episode": int(_require(record, "episode_index", "Episode日志"))
                    + 1,
                    "completion_rate": float(
                        _require(
                            record,
                            "completed_task_count",
                            "Episode日志",
                        )
                    )
                    / task_count,
                    "expiration_rate": float(
                        _require(
                            record,
                            "expired_task_count",
                            "Episode日志",
                        )
                    )
                    / task_count,
                    "delivered_timeliness_raw": float(
                        _require(
                            record,
                            "delivered_timeliness_raw",
                            "Episode日志",
                        )
                    ),
                    "delivered_data_mbit": float(
                        _require(
                            record,
                            "delivered_data_mbit",
                            "Episode日志",
                        )
                    ),
                    "episode_reward": float(
                        _require(
                            record,
                            "total_training_reward",
                            "Episode日志",
                        )
                    ),
                }
            )

    with (root / "ppo_diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        columns = (
            "episode",
            "update",
            "approximate_kl",
            "clip_fraction",
            "critic_loss",
            "entropy",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in update_records:
            writer.writerow(
                {
                    "episode": int(
                        _require(record, "episode_index", "Update日志")
                    )
                    + 1,
                    "update": int(
                        _require(record, "update_index", "Update日志")
                    ),
                    "approximate_kl": float(
                        _require(record, "approximate_kl", "Update日志")
                    ),
                    "clip_fraction": float(
                        _require(record, "clip_fraction", "Update日志")
                    ),
                    "critic_loss": float(
                        _require(record, "critic_loss", "Update日志")
                    ),
                    "entropy": float(
                        _require(record, "entropy", "Update日志")
                    ),
                }
            )

    with (root / "reward_contributions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        columns = ("episode",) + tuple(
            output_name for _, output_name in COMPONENT_COLUMNS
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in episode_records:
            values = {}
            absolute = _require(
                record,
                "reward_component_abs_sums",
                "Episode日志",
            )
            for source_name, output_name in COMPONENT_COLUMNS:
                if source_name not in absolute:
                    raise ValueError(
                        "Episode奖励贡献缺少必要字段：{0}".format(
                            source_name
                        )
                    )
                value = abs(float(absolute[source_name]))
                if not math.isfinite(value):
                    raise ValueError(
                        "Episode奖励贡献不是有限数：{0}".format(source_name)
                    )
                values[output_name] = value
            total = sum(values.values())
            if total <= 0.0:
                raise ValueError("Episode奖励绝对贡献总和必须大于0")
            writer.writerow(
                {
                    "episode": int(
                        _require(record, "episode_index", "Episode日志")
                    )
                    + 1,
                    **{
                        name: value / total for name, value in values.items()
                    },
                }
            )


def _remove_temporary_artifacts(output_directory):
    """删除转换完成后的临时日志与可能的阶段Checkpoint目录。"""
    root = Path(output_directory)
    for name in TEMPORARY_ARTIFACTS:
        path = root / name
        if path.is_file():
            path.unlink()
    stage_directory = root / "checkpoints"
    if stage_directory.is_dir():
        shutil.rmtree(stage_directory)


def _resolve_output_directory(args):
    """为两个 profile 提供互不覆盖的默认目录。"""
    if args.output_dir:
        return Path(args.output_dir)
    return Path("results/manual_diagnostics") / args.reward_profile


def main():
    """运行一个 profile，并在完成后仅保留诊断所需产物。"""
    args = parse_args()
    if args.episodes <= 0 or args.task_count <= 0:
        raise ValueError("episodes与task-count必须为正数")
    if args.seed < 0 or args.fixed_task_seed < 0:
        raise ValueError("seed必须为非负整数")
    output_directory = _resolve_output_directory(args).resolve()
    if output_directory.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖：{0}".format(output_directory))
    output_directory.mkdir(parents=True)

    baseline_config = deepcopy(load_baseline_config(args.config))
    mappo_config = load_mappo_config(args.mappo_config)
    baseline_config["seed"] = args.seed
    if args.device:
        baseline_config["device"] = args.device
    baseline_config["training"]["task_count"] = args.task_count
    baseline_config["training"]["episode_count"] = args.episodes
    mappo_config["manual_reward"]["weights"] = deepcopy(
        REWARD_PROFILES[args.reward_profile]
    )

    config, encoder, actor, critic, trainer, evaluator = build_baseline_components(
        BaselineMethod.MANUAL_MAPPO,
        baseline_config,
        mappo_config,
    )
    training = build_training_config(baseline_config, output_directory)
    training.update(
        {
            "task_count": args.task_count,
            "episode_count": args.episodes,
            "base_episode_seed": args.fixed_task_seed,
            "fixed_task_seed": args.fixed_task_seed,
            "save_episode_checkpoints": False,
            "write_learning_curves": False,
            "store_validation_details": False,
            "checkpoint": {
                "best_path": str(output_directory / "best.pt"),
                "last_path": str(output_directory / "last.pt"),
                "stage_directory": str(output_directory / "checkpoints"),
                "save_at_episode_boundary_only": True,
            },
            "logging": {
                "update_log_path": str(output_directory / ".updates.jsonl"),
                "episode_log_path": str(output_directory / ".episodes.jsonl"),
                "validation_log_path": str(
                    output_directory / ".validation.jsonl"
                ),
                "summary_path": str(output_directory / ".summary.json"),
            },
        }
    )
    runner = BaselineTrainingRunner(
        trainer,
        evaluator,
        config,
        encoder,
        training_config=training,
        method_name="manual_mappo",
    )
    try:
        runner.run(
            target_episode_count=args.episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            validation_max_steps=args.validation_max_steps,
        )
        episode_records = _read_jsonl(training["logging"]["episode_log_path"])
        update_records = _read_jsonl(training["logging"]["update_log_path"])
        write_diagnostic_csvs(
            output_directory,
            episode_records,
            update_records,
            args.task_count,
        )
    finally:
        _remove_temporary_artifacts(output_directory)

    expected = (
        "metrics.csv",
        "ppo_diagnostics.csv",
        "reward_contributions.csv",
        "best.pt",
        "last.pt",
    )
    missing = [name for name in expected if not (output_directory / name).is_file()]
    if missing:
        raise RuntimeError("诊断输出缺少必要文件：{0}".format(", ".join(missing)))
    print("诊断实验完成：{0}".format(output_directory))
    print("奖励权重：{0}".format(REWARD_PROFILES[args.reward_profile]))


if __name__ == "__main__":
    main()
