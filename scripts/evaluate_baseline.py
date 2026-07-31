"""确定性验证单个基线Checkpoint或重新评估整个Run。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from baselines.baseline_runner import build_baseline_components
from baselines.config import load_baseline_config, validate_baseline_config
from baselines.llm_schema import LlmRewardSpec
from baselines.methods import BaselineMethod, parse_baseline_method
from baselines.run_management import atomic_write_json, load_json
from mappo.checkpoint import load_checkpoint
from mappo.config import load_mappo_config


def parse_args():
    parser = argparse.ArgumentParser(description="验证单个基线或整个Run")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--method")
    target.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--reward-spec")
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _load_spec(path, baseline_config):
    if path is None:
        return None
    limits = baseline_config["methods"]["llm_ppo"]["weight_limits"]
    return LlmRewardSpec.load(path, limits["minimum"], limits["maximum"])


def evaluate_one(
    method,
    checkpoint_path,
    reward_spec,
    baseline_config,
    mappo_config,
    max_steps=None,
):
    """加载并校验Checkpoint后在固定validation场景确定性推理。"""
    method = BaselineMethod(method)
    spec = reward_spec if method == BaselineMethod.LLM_PPO else None
    config, encoder, actor, critic, trainer, evaluator = build_baseline_components(
        method,
        baseline_config,
        mappo_config,
        spec,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        actor,
        critic,
        expected_encoder_metadata=encoder.metadata(),
        map_location=trainer.device,
    )
    saved = checkpoint.get("config", {})
    if saved.get("baseline_method") != method.value:
        raise ValueError("Checkpoint方法与验证方法不一致")
    if method == BaselineMethod.PPO_LYA and saved.get(
        "lyapunov_reward_config"
    ) != trainer.reward_model.config:
        raise ValueError("Checkpoint的Lyapunov奖励配置不一致")
    if saved.get("reward_spec_id") != getattr(spec, "spec_id", None):
        raise ValueError("Checkpoint与冻结奖励规范不一致")
    validation = baseline_config["training"]["validation"]
    return evaluator.evaluate(
        validation["seeds"],
        validation["task_count"],
        max_steps=max_steps,
    )


def _evaluate_run(args, baseline_config, mappo_config):
    root = Path(args.run_dir)
    manifest = load_json(root / "run_manifest.json")
    spec = _load_spec(
        args.reward_spec or manifest.get("reward_spec_path"),
        baseline_config,
    )
    records = []
    for method_value in manifest["methods"]:
        method = BaselineMethod(method_value)
        checkpoint = root / method.value / "best_checkpoint.pt"
        state = manifest["method_states"].get(method.value, {})
        if state.get("status") != "completed" or not checkpoint.is_file():
            records.append(
                {
                    "method": method.value,
                    "status": "not_evaluated",
                    "reason": "方法未完成或不存在Best Checkpoint",
                }
            )
            continue
        result = evaluate_one(
            method,
            checkpoint,
            spec,
            baseline_config,
            mappo_config,
            args.max_steps,
        )
        atomic_write_json(root / method.value / "evaluation.json", result)
        records.append(
            {
                "method": method.value,
                "status": "evaluated",
                **result["aggregate"],
                "best_checkpoint": str(checkpoint),
                "reward_spec_id": (
                    spec.spec_id
                    if method == BaselineMethod.LLM_PPO
                    else None
                ),
            }
        )
    evaluated = [
        item for item in records if item["status"] == "evaluated"
    ]
    ranking = sorted(
        evaluated,
        key=lambda item: (
            -item["timeliness_raw_mean"],
            -item["load_balance_mean_per_task_mean"],
        ),
    )
    comparison = {
        "schema_version": "1.0",
        "methods": records,
        "validation_ranking": [item["method"] for item in ranking],
    }
    atomic_write_json(root / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False))


def main():
    args = parse_args()
    baseline_config = deepcopy(load_baseline_config(args.config))
    mappo_config = load_mappo_config(args.mappo_config)
    if args.device:
        baseline_config["device"] = args.device
    validate_baseline_config(baseline_config, mappo_config)
    if args.run_dir:
        _evaluate_run(args, baseline_config, mappo_config)
        return
    if not args.checkpoint:
        raise ValueError("单方法验证必须提供--checkpoint")
    method = parse_baseline_method(args.method)
    if method == BaselineMethod.LLM_PPO and not args.reward_spec:
        raise ValueError("LLM-PPO验证必须提供--reward-spec")
    spec = _load_spec(args.reward_spec, baseline_config)
    result = evaluate_one(
        method,
        args.checkpoint,
        spec,
        baseline_config,
        mappo_config,
        args.max_steps,
    )
    output = Path(args.output or "evaluation.json")
    atomic_write_json(output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
