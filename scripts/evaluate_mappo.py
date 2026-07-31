"""独立加载人工奖励MAPPO Checkpoint并在validation上确定性评估。"""

import argparse
import json
from pathlib import Path

from mappo.checkpoint import load_checkpoint
from mappo.config import load_mappo_config, resolve_device, validate_mappo_config
from mappo.encoding import MappoObservationEncoder
from mappo.evaluator import MappoEvaluator
from mappo.networks import CentralizedCritic, SharedActor
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.tasks import load_task_database, load_task_splits


def parse_args():
    """解析Checkpoint、validation seed和输出路径。"""
    parser = argparse.ArgumentParser(description="评估人工奖励MAPPO")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/mappo.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--output",
        default="results/day3/manual_reward_mappo/evaluation.json",
    )
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main():
    """拒绝test划分，加载模型并输出validation逐场景及聚合指标。"""
    args = parse_args()
    if args.split != "validation":
        raise ValueError("第三天独立评估只允许validation，test保留给最终实验")
    config = load_mappo_config(args.config)
    if args.device:
        config["device"] = args.device
    validate_mappo_config(config)
    set_global_seed(config["seed"])
    dataset = load_skyfield_dataset()
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        task_database=load_task_database(),
        task_splits=load_task_splits(),
    )
    encoder = MappoObservationEncoder(environment, config)
    device = resolve_device(config["device"])
    actor = SharedActor(config).to(device)
    critic = CentralizedCritic(config).to(device)
    load_checkpoint(
        args.checkpoint,
        actor,
        critic,
        expected_encoder_metadata=encoder.metadata(),
        map_location=device,
    )
    validation = config["manual_training"]["validation"]
    result = MappoEvaluator(environment, encoder, actor, device).evaluate(
        args.seeds or validation["seeds"],
        args.task_count or validation["task_count"],
        max_steps=args.max_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
