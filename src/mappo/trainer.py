"""实现共享Actor、集中式Critic的基础MAPPO采集与更新。"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .buffer import RolloutBuffer
from .config import resolve_device
from .encoding import decode_composite_action
from .reward import TimelinessDeltaReward


@dataclass
class UpdateStatistics:
    """记录一次MAPPO更新的有限标量诊断值。"""

    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    actor_gradient_norm: float
    critic_gradient_norm: float


class MappoTrainer:
    """管理CTDE Rollout、GAE及PPO更新。

    15颗卫星共用一个Actor并各自产生局部动作，因此Actor每次更新使用
    ``T×15`` 样本；完全合作任务只使用一个共享全局奖励和价值，所以Critic
    只使用T个集中状态。诊断奖励仅验证训练链路，不代表正式实验目标。
    """

    def __init__(
        self,
        environment,
        encoder,
        actor,
        critic,
        config,
        reward_model=None,
        auto_reset_on_done=True,
    ):
        """设置设备、私有采样器、优化器并重置环境，不修改任务数据库。

        调用方必须先用 ``set_global_seed`` 控制网络初始化；Trainer不会隐式
        重置Python、NumPy或PyTorch的全局随机状态。
        """
        self.environment = environment
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.config = config
        self.auto_reset_on_done = bool(auto_reset_on_done)
        self.device = resolve_device(config["device"])
        self.actor.to(self.device)
        self.critic.to(self.device)
        seed = int(config["seed"])
        # 这里只创建Trainer私有的minibatch随机生成器。网络初始化所需的
        # 全局种子必须由入口在构造Actor和Critic之前显式设置。
        self.rng = np.random.default_rng(seed)
        algorithm = config["algorithm"]
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(algorithm["actor_learning_rate"]),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(algorithm["critic_learning_rate"]),
        )
        reward_config = config["diagnostic_reward"]
        self.reward_enabled = bool(reward_config["enabled"])
        self.reward_model = reward_model or TimelinessDeltaReward(
            scale=float(reward_config["timeliness_scale"])
        )
        self.observations, _ = self.environment.reset(seed=seed)
        self._reset_reward_model()
        self.environment_steps = 0
        self.last_rollout_statistics = {}

    def _reset_reward_model(self):
        """按诊断奖励或环境感知奖励的公开接口重置Episode状态。"""
        if getattr(self.reward_model, "environment_aware", False):
            self.reward_model.reset(self.environment)
        else:
            self.reward_model.reset(
                {"timeliness_raw": self.environment.timeliness_raw}
            )

    def set_observations(self, observations):
        """设置外层调度器刚重置得到的15星观测，不修改环境。"""
        if set(observations) != set(self.encoder.satellite_ids):
            raise ValueError("Episode观测必须完整包含15颗卫星")
        self.observations = observations

    def reset_episode(self, seed, split=None, task_count=None, task_ids=None):
        """按划分或指定任务显式开始新Episode并重置奖励快照。

        ``task_ids`` 仅供固定任务诊断使用；未提供时保持原有按
        ``split + task_count`` 采样的训练行为。
        """
        observations, reset_info = self.environment.reset(
            seed=seed,
            split=split,
            task_count=task_count,
            task_ids=task_ids,
        )
        self.set_observations(observations)
        self._reset_reward_model()
        return reset_info

    def _tensor(self, value, dtype):
        """将NumPy数据转换到训练设备，不复制无关Python结构。"""
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def collect_rollout(self, steps):
        """采集指定环境步数，计算bootstrap与GAE并返回固定Buffer。"""
        if steps <= 0:
            raise ValueError("Rollout步数必须为正数")
        encoding = self.config["encoding"]
        buffer = RolloutBuffer(
            capacity=steps,
            agent_count=self.config["agents"]["count"],
            observation_dim=encoding["expected_actor_observation_dim"],
            critic_state_dim=encoding["expected_critic_state_dim"],
            task_slots=encoding["candidate_task_count"],
            target_choices=encoding["target_choice_count_with_idle"],
        )
        accepted_total = 0
        rejected_total = 0
        accepted_link_totals = {"ISL": 0, "IDL": 0, "SGL": 0}
        invalid_masked_actions = 0
        reward_values = []
        last_done = False
        final_info = None
        reward_component_sums = {}
        reward_component_abs_sums = {}

        for _ in range(steps):
            encoded = self.encoder.encode_all_agents(self.observations)
            satellite_ids = self.encoder.satellite_ids
            actor_observations = np.stack(
                [encoded[item].observation for item in satellite_ids]
            ).astype(np.float32)
            base_masks = np.stack(
                [encoded[item].base_target_mask for item in satellite_ids]
            )
            critic_state = self.encoder.encode_critic_state(
                encoded,
                self.environment.get_global_state(),
            )
            observation_tensor = self._tensor(actor_observations, torch.float32)
            mask_tensor = self._tensor(base_masks, torch.bool)
            critic_tensor = self._tensor(critic_state[None, :], torch.float32)
            with torch.no_grad():
                actor_batch = self.actor.sample_actions(
                    observation_tensor,
                    mask_tensor,
                    deterministic=False,
                )
                value = float(self.critic(critic_tensor).item())
            target_choices = actor_batch.target_choices.cpu().numpy()
            raw_actions = actor_batch.raw_continuous_actions.cpu().numpy()
            bounded_actions = actor_batch.bounded_continuous_actions.cpu().numpy()
            old_log_probs = actor_batch.log_prob.cpu().numpy()
            actions = {}
            for agent_index, satellite_id in enumerate(satellite_ids):
                choices = target_choices[agent_index]
                if not np.all(
                    encoded[satellite_id].base_target_mask[
                        np.arange(4),
                        choices,
                    ]
                ):
                    invalid_masked_actions += 1
                    raise RuntimeError("Actor采样了基础Mask禁止的动作")
                actions[satellite_id] = decode_composite_action(
                    encoded[satellite_id],
                    choices,
                    bounded_actions[agent_index],
                )
            next_observations, rewards, done, _, info = self.environment.step(actions)
            if getattr(self.reward_model, "environment_aware", False):
                breakdown = self.reward_model.compute(self.environment, info)
                shared_reward = breakdown.total_reward
                component_values = breakdown.component_values()
            elif self.reward_enabled:
                shared_reward = self.reward_model.compute(info)
                component_values = {
                    "diagnostic_timeliness_delta": shared_reward
                }
            else:
                shared_reward = float(np.mean(list(rewards.values())))
                component_values = {"environment_reward": shared_reward}
            if not math.isfinite(shared_reward):
                raise RuntimeError("共享诊断奖励包含NaN或Inf")
            buffer.add(
                observations=actor_observations,
                critic_state=critic_state,
                base_target_masks=base_masks,
                target_choices=target_choices,
                raw_continuous_actions=raw_actions,
                old_log_probs=old_log_probs,
                shared_reward=shared_reward,
                done=done,
                old_value=value,
            )
            accepted_total += info["accepted_subaction_count"]
            rejected_total += info["rejected_subaction_count"]
            for link_type in accepted_link_totals:
                accepted_link_totals[link_type] += int(
                    info["accepted_{0}_count".format(link_type.lower())]
                )
            reward_values.append(shared_reward)
            for name, value in component_values.items():
                reward_component_sums[name] = (
                    reward_component_sums.get(name, 0.0) + float(value)
                )
                reward_component_abs_sums[name] = (
                    reward_component_abs_sums.get(name, 0.0)
                    + abs(float(value))
                )
            self.environment_steps += 1
            last_done = done
            final_info = info
            if done:
                if not self.auto_reset_on_done:
                    self.observations = next_observations
                    break
                self.observations, _ = self.environment.reset(seed=self.config["seed"])
                self._reset_reward_model()
            else:
                self.observations = next_observations

        if last_done:
            bootstrap_value = 0.0
        else:
            encoded = self.encoder.encode_all_agents(self.observations)
            critic_state = self.encoder.encode_critic_state(
                encoded,
                self.environment.get_global_state(),
            )
            with torch.no_grad():
                bootstrap_value = float(
                    self.critic(
                        self._tensor(critic_state[None, :], torch.float32)
                    ).item()
                )
        algorithm = self.config["algorithm"]
        buffer.compute_gae(
            bootstrap_value,
            float(algorithm["gamma"]),
            float(algorithm["gae_lambda"]),
        )
        self.environment.check_data_conservation()
        self.last_rollout_statistics = {
            "mean_shared_reward": float(np.mean(reward_values)),
            "sum_shared_reward": float(np.sum(reward_values)),
            "min_shared_reward": float(np.min(reward_values)),
            "max_shared_reward": float(np.max(reward_values)),
            "accepted_subaction_count": int(accepted_total),
            "rejected_subaction_count": int(rejected_total),
            "invalid_masked_action_count": int(invalid_masked_actions),
            "episode_done": bool(last_done),
            "final_info": final_info,
            "reward_component_sums": reward_component_sums,
            "reward_component_abs_sums": reward_component_abs_sums,
            "reward_warning_count": int(
                getattr(self.reward_model, "warning_count", 0)
            ),
            "accepted_isl_count": accepted_link_totals["ISL"],
            "accepted_idl_count": accepted_link_totals["IDL"],
            "accepted_sgl_count": accepted_link_totals["SGL"],
            "reward_spec_id": getattr(
                self.reward_model,
                "reward_spec_id",
                None,
            ),
        }
        return buffer

    @staticmethod
    def _check_gradients(module, label):
        """确认指定网络所有已生成梯度均为有限数。"""
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(item).all() for item in gradients):
            raise RuntimeError("{0}梯度缺失或包含NaN/Inf".format(label))

    def update(self, buffer):
        """执行PPO clip Actor更新与value clip Critic更新并返回统计。"""
        if buffer.size == 0:
            raise ValueError("空Rollout不能执行MAPPO更新")
        algorithm = self.config["algorithm"]
        if algorithm["normalize_advantages"]:
            advantages = buffer.advantages[: buffer.size]
            mean = float(advantages.mean())
            standard_deviation = float(advantages.std())
            buffer.advantages[: buffer.size] = (
                advantages - mean
            ) / (standard_deviation + 1.0e-8)

        actor_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []
        actor_gradient_norms = []
        stop_actor = False
        for _ in range(int(algorithm["actor_epochs"])):
            epoch_kls = []
            for batch in buffer.iter_actor_minibatches(
                int(algorithm["actor_minibatch_size"]),
                rng=self.rng,
            ):
                observations = self._tensor(batch["observations"], torch.float32)
                masks = self._tensor(batch["base_target_masks"], torch.bool)
                choices = self._tensor(batch["target_choices"], torch.long)
                raw_actions = self._tensor(
                    batch["raw_continuous_actions"],
                    torch.float32,
                )
                old_log_probs = self._tensor(batch["old_log_probs"], torch.float32)
                advantages = self._tensor(batch["advantages"], torch.float32)
                new_log_probs, entropy = self.actor.evaluate_actions(
                    observations,
                    masks,
                    choices,
                    raw_actions,
                )
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                if not torch.isfinite(ratio).all():
                    raise RuntimeError("PPO概率比包含NaN或Inf")
                clipped_ratio = ratio.clamp(
                    1.0 - float(algorithm["clip_ratio"]),
                    1.0 + float(algorithm["clip_ratio"]),
                )
                policy_loss = -torch.minimum(
                    ratio * advantages,
                    clipped_ratio * advantages,
                ).mean()
                actor_loss = policy_loss - float(
                    algorithm["entropy_coefficient"]
                ) * entropy.mean()
                if not torch.isfinite(actor_loss):
                    raise RuntimeError("Actor loss包含NaN或Inf")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self._check_gradients(self.actor, "Actor")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    float(algorithm["max_gradient_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Actor梯度范数不是有限数")
                self.actor_optimizer.step()
                approximate_kl = float((old_log_probs - new_log_probs).mean().item())
                clip_fraction = float(
                    ((ratio - 1.0).abs() > float(algorithm["clip_ratio"]))
                    .float()
                    .mean()
                    .item()
                )
                actor_losses.append(float(actor_loss.item()))
                entropies.append(float(entropy.mean().item()))
                approximate_kls.append(approximate_kl)
                epoch_kls.append(approximate_kl)
                clip_fractions.append(clip_fraction)
                actor_gradient_norms.append(float(gradient_norm.item()))
            if epoch_kls and np.mean(epoch_kls) > float(algorithm["target_kl"]):
                stop_actor = True
            if stop_actor:
                break

        critic_losses = []
        critic_gradient_norms = []
        for _ in range(int(algorithm["critic_epochs"])):
            for batch in buffer.iter_critic_minibatches(
                int(algorithm["critic_minibatch_size"]),
                rng=self.rng,
            ):
                states = self._tensor(batch["critic_states"], torch.float32)
                old_values = self._tensor(batch["old_values"], torch.float32)
                returns = self._tensor(batch["returns"], torch.float32)
                values = self.critic(states)
                clipped_values = old_values + (values - old_values).clamp(
                    -float(algorithm["value_clip"]),
                    float(algorithm["value_clip"]),
                )
                loss_unclipped = (values - returns).square()
                loss_clipped = (clipped_values - returns).square()
                critic_loss = 0.5 * torch.maximum(
                    loss_unclipped,
                    loss_clipped,
                ).mean() * float(algorithm["value_loss_coefficient"])
                if not torch.isfinite(critic_loss):
                    raise RuntimeError("Critic loss包含NaN或Inf")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                self._check_gradients(self.critic, "Critic")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    float(algorithm["max_gradient_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Critic梯度范数不是有限数")
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))
                critic_gradient_norms.append(float(gradient_norm.item()))

        returns = buffer.returns[: buffer.size]
        old_values = buffer.old_values[: buffer.size]
        return_variance = float(np.var(returns))
        explained_variance = (
            0.0
            if return_variance <= 1.0e-12
            else 1.0 - float(np.var(returns - old_values)) / return_variance
        )
        statistics = UpdateStatistics(
            actor_loss=float(np.mean(actor_losses)),
            critic_loss=float(np.mean(critic_losses)),
            entropy=float(np.mean(entropies)),
            approximate_kl=float(np.mean(approximate_kls)),
            clip_fraction=float(np.mean(clip_fractions)),
            explained_variance=explained_variance,
            actor_gradient_norm=float(np.mean(actor_gradient_norms)),
            critic_gradient_norm=float(np.mean(critic_gradient_norms)),
        )
        if not all(math.isfinite(value) for value in statistics.__dict__.values()):
            raise RuntimeError("MAPPO更新统计包含NaN或Inf")
        return statistics


def parameter_vector(module):
    """把网络参数拼接为CPU向量，用于冒烟训练前后变化检查。"""
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    )
