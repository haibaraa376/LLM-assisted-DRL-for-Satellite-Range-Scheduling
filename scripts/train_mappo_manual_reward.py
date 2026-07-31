"""运行人工奖励MAPPO的完整Episode训练、validation和断点恢复。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from mappo.checkpoint import load_checkpoint, restore_rng_state
from mappo.config import load_mappo_config, validate_mappo_config
from mappo.encoding import MappoObservationEncoder
from mappo.evaluator import MappoEvaluator
from mappo.manual_reward import ManualReward
from mappo.training_runner import BaselineTrainingRunner
from mappo.networks import CentralizedCritic, SharedActor
from mappo.trainer import MappoTrainer
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.tasks import load_task_database, load_task_splits


def parse_args():
    """解析正式训练参数；episodes表示目标总Episode数。"""
    parser = argparse.ArgumentParser(description="训练人工奖励MAPPO基线")
    parser.add_argument("--config", default="configs/mappo.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--validation-seeds", nargs="+", type=int)
    parser.add_argument("--validation-task-count", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output-dir")
    # 仅用于CI或本地流程验收；省略时严格运行完整2880步Episode。
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--validation-max-steps", type=int)
    return parser.parse_args()


def _apply_overrides(config, args):
    """应用CLI覆盖并保持train/validation/test隔离。"""
    config = deepcopy(config)
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("训练种子必须是非负整数")
        config["seed"] = args.seed
        config["manual_training"]["base_episode_seed"] = args.seed
    if args.validation_seeds:
        config["manual_training"]["validation"]["seeds"] = args.validation_seeds
    if args.validation_task_count is not None:
        config["manual_training"]["validation"]["task_count"] = (
            args.validation_task_count
        )
    if args.output_dir:
        root = Path(args.output_dir)
        training = config["manual_training"]
        training["checkpoint"]["best_path"] = str(root / "best_checkpoint.pt")
        training["checkpoint"]["last_path"] = str(root / "last_checkpoint.pt")
        training["logging"]["update_log_path"] = str(root / "train_updates.jsonl")
        training["logging"]["episode_log_path"] = str(root / "episodes.jsonl")
        training["logging"]["validation_log_path"] = str(root / "validation.jsonl")
        training["logging"]["summary_path"] = str(root / "summary.json")
    return config


def _clean_new_training_outputs(config):
    """新训练仅删除人工奖励结果目录中的七个约定文件。"""
    training = config["manual_training"]
    paths = [
        training["checkpoint"]["best_path"],
        training["checkpoint"]["last_path"],
        training["logging"]["update_log_path"],
        training["logging"]["episode_log_path"],
        training["logging"]["validation_log_path"],
        training["logging"]["summary_path"],
    ]
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            path.unlink()


def main():
    """按冻结顺序构建网络，可选恢复，并训练到目标总Episode数。"""
    args = parse_args()
    config = _apply_overrides(load_mappo_config(args.config), args)
    validate_mappo_config(config)
    target_episodes = (
        args.episodes
        if args.episodes is not None
        else config["manual_training"]["episode_count"]
    )
    if target_episodes <= 0:
        raise ValueError("目标总Episode数必须为正数")
    set_global_seed(config["seed"])
    dataset = load_skyfield_dataset()
    database = load_task_database()
    splits = load_task_splits()
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        task_database=database,
        task_splits=splits,
    )
    encoder = MappoObservationEncoder(environment, config)
    actor = SharedActor(config)
    critic = CentralizedCritic(config)
    reward_model = ManualReward(config["manual_reward"])
    trainer = MappoTrainer(
        environment,
        encoder,
        actor,
        critic,
        config,
        reward_model=reward_model,
        auto_reset_on_done=False,
    )
    evaluator = MappoEvaluator(environment, encoder, actor, trainer.device)
    runner = BaselineTrainingRunner(trainer, evaluator, config, encoder)

    start_episode = 0
    update_index = 0
    best_result = None
    best_episode = None
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            actor,
            critic,
            trainer.actor_optimizer,
            trainer.critic_optimizer,
            encoder.metadata(),
            trainer.device,
        )
        state = checkpoint.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("Resume需要正式训练Checkpoint，旧smoke文件不包含训练状态")
        restore_rng_state(checkpoint.get("rng_state"), trainer.rng)
        start_episode = int(state["next_episode_index"])
        update_index = int(state["update_index"])
        trainer.environment_steps = int(state["environment_steps"])
        best_result = state.get("best_validation_result")
        best_episode = state.get("best_episode_index")
    else:
        _clean_new_training_outputs(config)

    summary = runner.run(
        target_episode_count=target_episodes,
        start_episode_index=start_episode,
        update_index=update_index,
        best_validation_result=best_result,
        best_episode_index=best_episode,
        skip_validation=args.skip_validation,
        max_steps_per_episode=args.max_steps_per_episode,
        validation_max_steps=args.validation_max_steps,
        resume=bool(args.resume),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
