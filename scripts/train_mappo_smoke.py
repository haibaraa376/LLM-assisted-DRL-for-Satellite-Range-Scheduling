"""运行2×64步MAPPO冒烟训练并验证Checkpoint确定性恢复。"""

import json
import platform
from pathlib import Path

import numpy as np
import torch

from mappo.checkpoint import load_checkpoint, save_checkpoint
from mappo.config import load_mappo_config
from mappo.encoding import MappoObservationEncoder, decode_composite_action
from mappo.networks import CentralizedCritic, SharedActor
from mappo.trainer import MappoTrainer, parameter_vector
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.tasks import load_task_database, load_task_splits, sample_episode_tasks


def build_environment(config):
    """按validation划分、20任务和固定种子构建冒烟环境。"""
    dataset = load_skyfield_dataset()
    database = load_task_database()
    split_ids = load_task_splits()[config["smoke_training"]["split"]]
    tasks = sample_episode_tasks(
        database,
        split_ids,
        config["smoke_training"]["task_count"],
        config["seed"],
    )
    return CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        tasks,
    )


def count_parameters(module):
    """返回网络中所有可训练参数数量。"""
    return sum(parameter.numel() for parameter in module.parameters())


def deterministic_reload_check(trainer, loaded_actor, steps):
    """在相同的连续16个真实状态上比较加载前后确定性动作。"""
    all_match = True
    all_bounded = True
    for _ in range(steps):
        encoded = trainer.encoder.encode_all_agents(trainer.observations)
        observations = np.stack(
            [encoded[item].observation for item in trainer.encoder.satellite_ids]
        )
        masks = np.stack(
            [encoded[item].base_target_mask for item in trainer.encoder.satellite_ids]
        )
        observation_tensor = trainer._tensor(observations, torch.float32)
        mask_tensor = trainer._tensor(masks, torch.bool)
        with torch.no_grad():
            original = trainer.actor.sample_actions(
                observation_tensor,
                mask_tensor,
                deterministic=True,
            )
            restored = loaded_actor.sample_actions(
                observation_tensor,
                mask_tensor,
                deterministic=True,
            )
        choices_match = torch.equal(
            original.target_choices,
            restored.target_choices,
        )
        continuous_match = torch.equal(
            original.bounded_continuous_actions,
            restored.bounded_continuous_actions,
        )
        all_match &= choices_match and continuous_match
        bounded = original.bounded_continuous_actions
        all_bounded &= bool(torch.all((bounded >= 0.0) & (bounded <= 1.0)))
        choices = original.target_choices.cpu().numpy()
        continuous = original.bounded_continuous_actions.cpu().numpy()
        actions = {
            satellite_id: decode_composite_action(
                encoded[satellite_id],
                choices[index],
                continuous[index],
            )
            for index, satellite_id in enumerate(trainer.encoder.satellite_ids)
        }
        next_observations, _, done, _, _ = trainer.environment.step(actions)
        if done:
            trainer.observations, _ = trainer.environment.reset(
                seed=trainer.config["seed"]
            )
        else:
            trainer.observations = next_observations
    trainer.environment._check_data_conservation()
    return all_match, all_bounded


def main():
    """执行冒烟训练、JSONL日志、单Checkpoint往返和JSON摘要。"""
    config = load_mappo_config()
    # 网络正交初始化会消耗PyTorch随机数，因此必须先设种子再创建任何网络。
    set_global_seed(config["seed"])
    environment = build_environment(config)
    encoder = MappoObservationEncoder(environment, config)
    actor = SharedActor(config)
    critic = CentralizedCritic(config)
    trainer = MappoTrainer(environment, encoder, actor, critic, config)
    actor_before = parameter_vector(actor)
    critic_before = parameter_vector(critic)
    actor_initial_norm = float(torch.linalg.vector_norm(actor_before))
    critic_initial_norm = float(torch.linalg.vector_norm(critic_before))
    logs = []
    all_losses_finite = True
    all_gradients_finite = True
    invalid_masked_action_count = 0
    rollout_steps = int(config["rollout"]["steps_per_update"])
    for update_index in range(1, config["smoke_training"]["update_count"] + 1):
        buffer = trainer.collect_rollout(rollout_steps)
        statistics = trainer.update(buffer)
        rollout_statistics = trainer.last_rollout_statistics
        invalid_masked_action_count += rollout_statistics[
            "invalid_masked_action_count"
        ]
        numeric = list(statistics.__dict__.values())
        all_losses_finite &= bool(np.all(np.isfinite(numeric[:6])))
        all_gradients_finite &= bool(np.all(np.isfinite(numeric[6:])))
        logs.append(
            {
                "update": update_index,
                "environment_steps": trainer.environment_steps,
                "mean_shared_reward": rollout_statistics["mean_shared_reward"],
                "sum_shared_reward": rollout_statistics["sum_shared_reward"],
                "actor_loss": statistics.actor_loss,
                "critic_loss": statistics.critic_loss,
                "entropy": statistics.entropy,
                "approximate_kl": statistics.approximate_kl,
                "clip_fraction": statistics.clip_fraction,
                "actor_gradient_norm": statistics.actor_gradient_norm,
                "critic_gradient_norm": statistics.critic_gradient_norm,
                "accepted_subaction_count": rollout_statistics[
                    "accepted_subaction_count"
                ],
                "rejected_subaction_count": rollout_statistics[
                    "rejected_subaction_count"
                ],
                "reward_mode": "diagnostic_timeliness_delta",
                "official_experiment": False,
            }
        )

    actor_change = float(torch.linalg.vector_norm(parameter_vector(actor) - actor_before))
    critic_change = float(torch.linalg.vector_norm(parameter_vector(critic) - critic_before))
    if actor_change <= 0.0 or critic_change <= 0.0:
        raise RuntimeError("冒烟更新后Actor或Critic参数没有变化")

    smoke = config["smoke_training"]
    log_path = Path(smoke["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in logs),
        encoding="utf-8",
    )
    checkpoint_path = Path(smoke["checkpoint_path"])
    save_checkpoint(
        checkpoint_path,
        actor,
        critic,
        trainer.actor_optimizer,
        trainer.critic_optimizer,
        config,
        smoke["update_count"],
        encoder.metadata(),
    )
    # 恢复网络的随机初值随后会被Checkpoint覆盖；仍显式设置种子，使所有
    # 网络构造入口都遵守统一顺序。
    set_global_seed(config["seed"])
    loaded_actor = SharedActor(config).to(trainer.device)
    loaded_critic = CentralizedCritic(config).to(trainer.device)
    loaded_actor_optimizer = torch.optim.Adam(loaded_actor.parameters())
    loaded_critic_optimizer = torch.optim.Adam(loaded_critic.parameters())
    load_checkpoint(
        checkpoint_path,
        loaded_actor,
        loaded_critic,
        loaded_actor_optimizer,
        loaded_critic_optimizer,
        encoder.metadata(),
        trainer.device,
    )
    deterministic_match, all_actions_bounded = deterministic_reload_check(
        trainer,
        loaded_actor,
        int(smoke["deterministic_evaluation_steps"]),
    )
    last = logs[-1]
    summary = {
        "seed": config["seed"],
        "device": str(trainer.device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "actor_observation_dim": config["encoding"][
            "expected_actor_observation_dim"
        ],
        "critic_state_dim": config["encoding"]["expected_critic_state_dim"],
        "agent_count": config["agents"]["count"],
        "candidate_task_count": config["encoding"]["candidate_task_count"],
        "target_choice_count": config["encoding"][
            "target_choice_count_with_idle"
        ],
        "updates": smoke["update_count"],
        "rollout_steps_per_update": rollout_steps,
        "total_environment_steps": trainer.environment_steps,
        "actor_parameters": count_parameters(actor),
        "critic_parameters": count_parameters(critic),
        "actor_initial_parameter_norm": actor_initial_norm,
        "critic_initial_parameter_norm": critic_initial_norm,
        "actor_parameter_change_norm": actor_change,
        "critic_parameter_change_norm": critic_change,
        "actor_loss": last["actor_loss"],
        "critic_loss": last["critic_loss"],
        "entropy": last["entropy"],
        "approximate_kl": last["approximate_kl"],
        "all_losses_finite": all_losses_finite,
        "all_gradients_finite": all_gradients_finite,
        "all_actions_within_bounds": all_actions_bounded,
        "invalid_masked_action_count": invalid_masked_action_count,
        "environment_data_conservation_passed": True,
        "checkpoint_saved": checkpoint_path.exists(),
        "checkpoint_reloaded": True,
        "deterministic_actions_match_after_reload": deterministic_match,
        "reward_mode": "diagnostic_timeliness_delta",
        "official_experiment": False,
    }
    Path(smoke["summary_path"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
