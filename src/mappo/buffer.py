"""实现固定长度多智能体Rollout Buffer与GAE。"""

import numpy as np


class RolloutBuffer:
    """保存T步、15智能体的Actor数据和T份集中式Critic数据。

    Actor训练将时间与智能体维展平为 ``T×15``；同一时间步的共享优势复制给
    15个Actor样本。Critic只保留 ``T`` 个全局状态，避免把相同状态重复15次。
    Buffer保存变换前raw连续动作，以便PPO精确重算Tanh变换后的概率密度。
    """

    def __init__(
        self,
        capacity,
        agent_count=15,
        observation_dim=213,
        critic_state_dim=3200,
        task_slots=4,
        target_choices=20,
    ):
        """预分配固定形状数组；容量为环境step数。"""
        if capacity <= 0:
            raise ValueError("Rollout容量必须为正整数")
        self.capacity = int(capacity)
        self.agent_count = int(agent_count)
        self.observation_dim = int(observation_dim)
        self.critic_state_dim = int(critic_state_dim)
        self.task_slots = int(task_slots)
        self.target_choices = int(target_choices)
        self.observations = np.zeros(
            (capacity, agent_count, observation_dim),
            dtype=np.float32,
        )
        self.critic_states = np.zeros((capacity, critic_state_dim), dtype=np.float32)
        self.base_target_masks = np.zeros(
            (capacity, agent_count, task_slots, target_choices),
            dtype=bool,
        )
        self.target_actions = np.zeros(
            (capacity, agent_count, task_slots),
            dtype=np.int64,
        )
        self.raw_continuous_actions = np.zeros(
            (capacity, agent_count, task_slots, 2),
            dtype=np.float32,
        )
        self.old_log_probs = np.zeros((capacity, agent_count), dtype=np.float32)
        self.shared_rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.old_values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.bootstrap_value = 0.0
        self.position = 0

    @property
    def size(self):
        """返回当前已保存的环境step数。"""
        return self.position

    def add(
        self,
        observations,
        critic_state,
        base_target_masks,
        target_choices,
        raw_continuous_actions,
        old_log_probs,
        shared_reward,
        done,
        old_value,
    ):
        """追加一个环境step；所有Actor数组第一维必须是15个智能体。"""
        if self.position >= self.capacity:
            raise RuntimeError("Rollout Buffer已经写满")
        expected_shapes = {
            "observations": (
                np.asarray(observations).shape,
                (self.agent_count, self.observation_dim),
            ),
            "critic_state": (
                np.asarray(critic_state).shape,
                (self.critic_state_dim,),
            ),
            "base_target_masks": (
                np.asarray(base_target_masks).shape,
                (self.agent_count, self.task_slots, self.target_choices),
            ),
            "target_choices": (
                np.asarray(target_choices).shape,
                (self.agent_count, self.task_slots),
            ),
            "raw_continuous_actions": (
                np.asarray(raw_continuous_actions).shape,
                (self.agent_count, self.task_slots, 2),
            ),
            "old_log_probs": (
                np.asarray(old_log_probs).shape,
                (self.agent_count,),
            ),
        }
        for name, (actual, expected) in expected_shapes.items():
            if actual != expected:
                raise ValueError("{0}形状应为{1}".format(name, expected))
        numeric_values = (
            observations,
            critic_state,
            raw_continuous_actions,
            old_log_probs,
            [shared_reward, old_value],
        )
        if not all(np.all(np.isfinite(value)) for value in numeric_values):
            raise ValueError("Rollout数据包含NaN或Inf")
        index = self.position
        self.observations[index] = observations
        self.critic_states[index] = critic_state
        self.base_target_masks[index] = base_target_masks
        self.target_actions[index] = target_choices
        self.raw_continuous_actions[index] = raw_continuous_actions
        self.old_log_probs[index] = old_log_probs
        self.shared_rewards[index] = shared_reward
        self.dones[index] = done
        self.old_values[index] = old_value
        self.position += 1

    def compute_gae(self, bootstrap_value, gamma, gae_lambda):
        """按共享奖励计算GAE和return；done后的状态不进行bootstrap。"""
        if self.position == 0:
            raise RuntimeError("空Rollout不能计算GAE")
        if not np.isfinite(bootstrap_value):
            raise ValueError("bootstrap value必须是有限数")
        if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma或gae_lambda范围非法")
        self.bootstrap_value = float(bootstrap_value)
        next_advantage = 0.0
        for time_index in reversed(range(self.position)):
            nonterminal = 1.0 - float(self.dones[time_index])
            next_value = (
                self.bootstrap_value
                if time_index == self.position - 1
                else self.old_values[time_index + 1]
            )
            delta = (
                self.shared_rewards[time_index]
                + gamma * nonterminal * next_value
                - self.old_values[time_index]
            )
            next_advantage = (
                delta
                + gamma * gae_lambda * nonterminal * next_advantage
            )
            self.advantages[time_index] = next_advantage
        self.returns[: self.position] = (
            self.advantages[: self.position] + self.old_values[: self.position]
        )
        if not np.all(np.isfinite(self.advantages[: self.position])):
            raise RuntimeError("GAE结果包含NaN或Inf")

    def iter_actor_minibatches(self, batch_size, shuffle=True, rng=None):
        """迭代 ``T×15`` Actor样本；共享优势按智能体数复制。"""
        if batch_size <= 0:
            raise ValueError("Actor minibatch大小必须为正数")
        time_count = self.position
        sample_count = time_count * self.agent_count
        indices = np.arange(sample_count)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(indices)
        observations = self.observations[:time_count].reshape(
            sample_count,
            self.observation_dim,
        )
        masks = self.base_target_masks[:time_count].reshape(
            sample_count,
            self.task_slots,
            self.target_choices,
        )
        choices = self.target_actions[:time_count].reshape(sample_count, self.task_slots)
        raw_actions = self.raw_continuous_actions[:time_count].reshape(
            sample_count,
            self.task_slots,
            2,
        )
        log_probs = self.old_log_probs[:time_count].reshape(sample_count)
        advantages = np.repeat(self.advantages[:time_count], self.agent_count)
        for start in range(0, sample_count, batch_size):
            selected = indices[start : start + batch_size]
            yield {
                "observations": observations[selected],
                "base_target_masks": masks[selected],
                "target_choices": choices[selected],
                "raw_continuous_actions": raw_actions[selected],
                "old_log_probs": log_probs[selected],
                "advantages": advantages[selected],
            }

    def iter_critic_minibatches(self, batch_size, shuffle=True, rng=None):
        """迭代T个集中式Critic样本，不复制到智能体维。"""
        if batch_size <= 0:
            raise ValueError("Critic minibatch大小必须为正数")
        indices = np.arange(self.position)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(indices)
        for start in range(0, self.position, batch_size):
            selected = indices[start : start + batch_size]
            yield {
                "critic_states": self.critic_states[selected],
                "old_values": self.old_values[selected],
                "returns": self.returns[selected],
            }
