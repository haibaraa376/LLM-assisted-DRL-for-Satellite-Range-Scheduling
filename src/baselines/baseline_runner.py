"""组装共享MAPPO基础设施，供三种奖励基线复用。"""

from copy import deepcopy
from pathlib import Path

from mappo.checkpoint import load_checkpoint, restore_rng_state
from mappo.encoding import MappoObservationEncoder
from mappo.evaluator import MappoEvaluator
from mappo.evaluation_protocol import (
    build_evaluation_protocol,
    public_protocol_metadata,
)
from mappo.manual_reward import ManualReward
from mappo.networks import CentralizedCritic, SharedActor
from mappo.trainer import MappoTrainer
from mappo.training_runner import BaselineTrainingRunner
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.tasks import load_task_database, load_task_splits

from .llm_reward import LlmWeightReward
from .lyapunov_reward import PpoLyaReward
from .methods import BaselineMethod


def build_training_config(baseline_config, output_dir):
    """把公平的基线配置映射到通用Runner格式。"""
    training = baseline_config["training"]
    root = Path(output_dir)
    return {
        "split": "train",
        "task_count": int(training["task_count"]),
        "base_episode_seed": int(baseline_config["seed"]),
        "episode_count": int(training["episode_count"]),
        "rollout_steps_per_update": int(training["rollout_steps"]),
        "validation_interval_episodes": int(
            training["validation_interval_episodes"]
        ),
        "checkpoint_interval_episodes": 1,
        "validation": deepcopy(training["validation"]),
        "evaluation_protocols": deepcopy(
            baseline_config["evaluation_protocols"]
        ),
        "checkpoint": {
            "best_path": str(root / "best_checkpoint.pt"),
            "last_path": str(root / "last_checkpoint.pt"),
            "stage_directory": str(root / "checkpoints"),
            "save_at_episode_boundary_only": True,
        },
        "save_episode_checkpoints": bool(
            baseline_config["artifact_management"]["save_episode_checkpoints"]
        ),
        "write_learning_curves": bool(
            baseline_config["artifact_management"]["write_learning_curves"]
        ),
        "logging": {
            "update_log_path": str(root / "train_updates.jsonl"),
            "episode_log_path": str(root / "episodes.jsonl"),
            "validation_log_path": str(root / "validation.jsonl"),
            "summary_path": str(root / "summary.json"),
        },
        "best_model_rule": deepcopy(baseline_config["best_model_rule"]),
    }


def build_baseline_components(
    method,
    baseline_config,
    mappo_config,
    reward_spec=None,
):
    """按统一种子构建环境、网络、优化器、奖励和验证器。"""
    method = BaselineMethod(method)
    config = deepcopy(mappo_config)
    config["seed"] = int(baseline_config["seed"])
    config["device"] = baseline_config["device"]
    config["baseline_method"] = method.value
    config["baseline_evaluation_protocols"] = deepcopy(
        baseline_config["evaluation_protocols"]
    )
    set_global_seed(config["seed"])
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        load_skyfield_dataset(),
        load_environment_config(),
        task_database=load_task_database(),
        task_splits=load_task_splits(),
    )
    encoder = MappoObservationEncoder(environment, config)
    actor = SharedActor(config)
    critic = CentralizedCritic(config)
    manual = ManualReward(config["manual_reward"])
    if method == BaselineMethod.MANUAL_MAPPO:
        if reward_spec is not None:
            raise ValueError("Manual-MAPPO不接受奖励规范")
        reward_model = manual
    elif method == BaselineMethod.PPO_LYA:
        if reward_spec is not None:
            raise ValueError("PPO-Lya不接受奖励规范")
        config["lyapunov_reward_config"] = deepcopy(
            baseline_config["methods"]["ppo_lya"]["lyapunov"]
        )
        reward_model = PpoLyaReward(
            manual,
            config["lyapunov_reward_config"],
        )
    else:
        if reward_spec is None:
            raise ValueError("LLM-PPO必须提供冻结reward spec")
        reward_model = LlmWeightReward(
            manual,
            reward_spec,
            baseline_config.get("reward_composition"),
        )
        config["reward_spec_id"] = reward_spec.spec_id
        config["llm_reward_weight_metadata"] = deepcopy(
            reward_model.weight_metadata
        )
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
    return config, encoder, actor, critic, trainer, evaluator


def restore_baseline_checkpoint(
    path,
    actor,
    critic,
    trainer,
    encoder,
    method,
    protocol_name="checkpoint_selection",
):
    """恢复边界Checkpoint并校验方法与冻结奖励标识。"""
    checkpoint = load_checkpoint(
        path,
        actor,
        critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        encoder.metadata(),
        trainer.device,
    )
    saved_config = checkpoint.get("config", {})
    method = BaselineMethod(method)
    if saved_config.get("baseline_method") != method.value:
        raise ValueError("Checkpoint所属基线方法与当前方法不一致")
    if method == BaselineMethod.PPO_LYA and saved_config.get(
        "lyapunov_reward_config"
    ) != trainer.reward_model.config:
        raise ValueError("Checkpoint的PPO-Lya奖励配置与当前配置不一致")
    expected_spec = getattr(trainer.reward_model, "reward_spec_id", None)
    if saved_config.get("reward_spec_id") != expected_spec:
        raise ValueError("Checkpoint冻结奖励Spec与当前文件不一致")
    expected_weight_metadata = getattr(
        trainer.reward_model,
        "weight_metadata",
        None,
    )
    if saved_config.get("llm_reward_weight_metadata") != expected_weight_metadata:
        raise ValueError("Checkpoint的LLM有效权重元数据与当前配置不一致")
    state = checkpoint.get("training_state")
    if not isinstance(state, dict):
        raise ValueError("基线Resume需要Episode边界Checkpoint")
    protocol = build_evaluation_protocol(
        protocol_name,
        trainer.config["baseline_evaluation_protocols"],
        trainer.environment.task_splits,
    )
    if state.get("evaluation_protocol") != public_protocol_metadata(protocol):
        raise ValueError("Checkpoint评估协议与当前运行协议不一致")
    restore_rng_state(checkpoint.get("rng_state"), trainer.rng)
    trainer.environment_steps = int(state["environment_steps"])
    return state


def make_runner(
    method,
    trainer,
    evaluator,
    config,
    encoder,
    baseline_config,
    output_dir,
):
    """创建复用MAPPO实现的统一外层Runner。"""
    method = BaselineMethod(method)
    training = build_training_config(baseline_config, output_dir)
    return BaselineTrainingRunner(
        trainer,
        evaluator,
        config,
        encoder,
        training_config=training,
        method_name=method.value,
    )
