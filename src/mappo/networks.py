"""定义15颗卫星共享的Actor与单值集中式Critic。"""

from dataclasses import dataclass

import torch
from torch import nn

from .distributions import SquashedNormal01


@dataclass
class ActorActionBatch:
    """保存一批自回归复合动作及其策略统计。"""

    target_choices: torch.Tensor
    raw_continuous_actions: torch.Tensor
    bounded_continuous_actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor


def _orthogonal_linear(layer, gain):
    """对Linear层应用正交权重和零偏置并返回原层。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class SharedActor(nn.Module):
    """由15颗卫星共享参数的复合动作策略网络。

    共享Actor利用相同决策规则处理不同卫星，身份差异由7维物理身份特征表达，
    无需维护15套网络。4个任务槽按顺序自回归采样，使局部可预知的3条星间、
    1条SGL和卫星目标不重复约束直接满足；其他智能体竞争仍交给环境仲裁。
    """

    def __init__(self, config):
        """按配置创建213维输入、4×20离散头和4×2连续头。"""
        super().__init__()
        encoding = config["encoding"]
        actor = config["actor"]
        self.observation_dim = encoding["expected_actor_observation_dim"]
        self.task_slots = encoding["candidate_task_count"]
        self.target_choices = encoding["target_choice_count_with_idle"]
        self.satellite_target_count = encoding["satellite_target_count"]
        self.log_std_min = float(actor["continuous_log_std_min"])
        self.log_std_max = float(actor["continuous_log_std_max"])
        hidden_sizes = actor["hidden_sizes"]
        if hidden_sizes != [256, 256] or actor["activation"] != "tanh":
            raise ValueError("共享Actor网络必须使用[256,256] Tanh结构")
        tanh_gain = nn.init.calculate_gain("tanh")
        self.backbone = nn.Sequential(
            _orthogonal_linear(nn.Linear(self.observation_dim, 256), tanh_gain),
            nn.Tanh(),
            _orthogonal_linear(nn.Linear(256, 256), tanh_gain),
            nn.Tanh(),
        )
        output_gain = float(actor["output_gain"])
        self.target_head = _orthogonal_linear(
            nn.Linear(256, self.task_slots * self.target_choices),
            output_gain,
        )
        self.continuous_head = _orthogonal_linear(
            nn.Linear(256, self.task_slots * 2),
            output_gain,
        )
        self.continuous_log_std = nn.Parameter(
            torch.full(
                (self.task_slots, 2),
                float(actor["continuous_log_std_initial"]),
            )
        )

    def forward(self, observations):
        """返回 ``[B,4,20]`` logits、``[B,4,2]`` mean和``[4,2]`` log_std。"""
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("Actor输入形状必须为[B,213]")
        if not torch.isfinite(observations).all():
            raise ValueError("Actor输入包含NaN或Inf")
        hidden = self.backbone(observations)
        logits = self.target_head(hidden).view(-1, self.task_slots, self.target_choices)
        mean = self.continuous_head(hidden).view(-1, self.task_slots, 2)
        log_std = self.continuous_log_std.clamp(self.log_std_min, self.log_std_max)
        if not torch.isfinite(log_std).all():
            raise RuntimeError("Actor连续log_std包含NaN或Inf")
        return logits, mean, log_std

    def _validate_masks(self, masks, batch_size):
        """验证基础Mask形状、类型及IDLE不变量。"""
        expected = (batch_size, self.task_slots, self.target_choices)
        if masks.shape != expected or masks.dtype != torch.bool:
            raise ValueError("Actor基础Mask必须是[B,4,20]的bool张量")
        if not torch.all(masks[:, :, 0]):
            raise ValueError("Actor每个任务槽的IDLE必须有效")

    def _mask_for_slot(
        self,
        base_slot_mask,
        selected_satellite_targets,
        inter_count,
        sgl_selected,
    ):
        """根据前序选择重建当前槽动态Mask。"""
        mask = base_slot_mask.clone()
        satellite_choices = mask[:, 1 : 1 + self.satellite_target_count]
        satellite_choices &= ~selected_satellite_targets
        satellite_choices &= (inter_count < 3).unsqueeze(1)
        mask[:, 1 : 1 + self.satellite_target_count] = satellite_choices
        ground_start = 1 + self.satellite_target_count
        mask[:, ground_start:] &= ~sgl_selected.unsqueeze(1)
        mask[:, 0] = True
        if not torch.all(mask.any(dim=1)):
            raise RuntimeError("动态Mask出现没有合法动作的任务槽")
        return mask

    def sample_actions(self, observations, base_target_masks, deterministic=False):
        """按4槽自回归采样并返回离散、raw、有界动作及总概率。"""
        logits, mean, log_std = self.forward(observations)
        batch_size = observations.shape[0]
        self._validate_masks(base_target_masks, batch_size)
        expanded_log_std = log_std.unsqueeze(0).expand_as(mean)
        continuous_distribution = SquashedNormal01(mean, expanded_log_std)
        if deterministic:
            raw_all, bounded_all = continuous_distribution.deterministic()
        else:
            raw_all, bounded_all = continuous_distribution.sample()

        selected_satellite_targets = torch.zeros(
            (batch_size, self.satellite_target_count),
            dtype=torch.bool,
            device=observations.device,
        )
        inter_count = torch.zeros(batch_size, dtype=torch.long, device=observations.device)
        sgl_selected = torch.zeros(batch_size, dtype=torch.bool, device=observations.device)
        choices = []
        total_log_prob = torch.zeros(batch_size, device=observations.device)
        total_entropy = torch.zeros(batch_size, device=observations.device)

        for slot in range(self.task_slots):
            dynamic_mask = self._mask_for_slot(
                base_target_masks[:, slot],
                selected_satellite_targets,
                inter_count,
                sgl_selected,
            )
            masked_logits = logits[:, slot].masked_fill(~dynamic_mask, -1.0e9)
            categorical = torch.distributions.Categorical(logits=masked_logits)
            choice = masked_logits.argmax(dim=1) if deterministic else categorical.sample()
            choices.append(choice)
            total_log_prob += categorical.log_prob(choice)
            total_entropy += categorical.entropy()
            active = choice != 0
            continuous_log_prob = continuous_distribution.log_prob(raw_all)[:, slot].sum(dim=1)
            continuous_entropy = continuous_distribution.base_entropy()[:, slot].sum(dim=1)
            total_log_prob += continuous_log_prob * active
            total_entropy += continuous_entropy * active

            satellite_selected = (choice >= 1) & (
                choice <= self.satellite_target_count
            )
            if satellite_selected.any():
                rows = torch.nonzero(satellite_selected, as_tuple=False).squeeze(1)
                columns = choice[rows] - 1
                selected_satellite_targets[rows, columns] = True
                inter_count += satellite_selected.long()
            sgl_selected |= choice > self.satellite_target_count

        target_choices = torch.stack(choices, dim=1)
        active_slots = target_choices != 0
        raw_actions = raw_all * active_slots.unsqueeze(-1)
        bounded_actions = bounded_all * active_slots.unsqueeze(-1)
        if not torch.isfinite(total_log_prob).all():
            raise RuntimeError("Actor采样log probability出现NaN或Inf")
        return ActorActionBatch(
            target_choices=target_choices,
            raw_continuous_actions=raw_actions,
            bounded_continuous_actions=bounded_actions,
            log_prob=total_log_prob,
            entropy=total_entropy,
        )

    def evaluate_actions(
        self,
        observations,
        base_target_masks,
        target_choices,
        raw_continuous_actions,
    ):
        """重建自回归动态Mask并返回指定动作的新总log probability与熵。"""
        logits, mean, log_std = self.forward(observations)
        batch_size = observations.shape[0]
        self._validate_masks(base_target_masks, batch_size)
        if target_choices.shape != (batch_size, self.task_slots):
            raise ValueError("离散动作形状必须为[B,4]")
        if raw_continuous_actions.shape != (batch_size, self.task_slots, 2):
            raise ValueError("raw连续动作形状必须为[B,4,2]")
        distribution = SquashedNormal01(
            mean,
            log_std.unsqueeze(0).expand_as(mean),
        )
        selected_satellite_targets = torch.zeros(
            (batch_size, self.satellite_target_count),
            dtype=torch.bool,
            device=observations.device,
        )
        inter_count = torch.zeros(batch_size, dtype=torch.long, device=observations.device)
        sgl_selected = torch.zeros(batch_size, dtype=torch.bool, device=observations.device)
        total_log_prob = torch.zeros(batch_size, device=observations.device)
        total_entropy = torch.zeros(batch_size, device=observations.device)
        for slot in range(self.task_slots):
            mask = self._mask_for_slot(
                base_target_masks[:, slot],
                selected_satellite_targets,
                inter_count,
                sgl_selected,
            )
            choice = target_choices[:, slot]
            if torch.any(choice < 0) or torch.any(choice >= self.target_choices):
                raise ValueError("离散动作超出0到19范围")
            if not torch.all(mask.gather(1, choice.unsqueeze(1)).squeeze(1)):
                raise ValueError("评估动作违反其自回归动态Mask")
            categorical = torch.distributions.Categorical(
                logits=logits[:, slot].masked_fill(~mask, -1.0e9)
            )
            active = choice != 0
            total_log_prob += categorical.log_prob(choice)
            total_entropy += categorical.entropy()
            total_log_prob += (
                distribution.log_prob(raw_continuous_actions)[:, slot].sum(dim=1)
                * active
            )
            total_entropy += distribution.base_entropy()[:, slot].sum(dim=1) * active
            satellite_selected = (choice >= 1) & (
                choice <= self.satellite_target_count
            )
            if satellite_selected.any():
                rows = torch.nonzero(satellite_selected, as_tuple=False).squeeze(1)
                selected_satellite_targets[rows, choice[rows] - 1] = True
                inter_count += satellite_selected.long()
            sgl_selected |= choice > self.satellite_target_count
        if not torch.isfinite(total_log_prob).all():
            raise RuntimeError("Actor评估log probability出现NaN或Inf")
        return total_log_prob, total_entropy


class CentralizedCritic(nn.Module):
    """将3200维全局状态映射为一个共享合作价值 ``V(s_t)``。"""

    def __init__(self, config):
        """创建3200→512→256→1的Tanh网络并进行正交初始化。"""
        super().__init__()
        critic = config["critic"]
        self.state_dim = config["encoding"]["expected_critic_state_dim"]
        if critic["hidden_sizes"] != [512, 256] or critic["activation"] != "tanh":
            raise ValueError("集中式Critic必须使用[512,256] Tanh结构")
        tanh_gain = nn.init.calculate_gain("tanh")
        self.network = nn.Sequential(
            _orthogonal_linear(nn.Linear(self.state_dim, 512), tanh_gain),
            nn.Tanh(),
            _orthogonal_linear(nn.Linear(512, 256), tanh_gain),
            nn.Tanh(),
            _orthogonal_linear(nn.Linear(256, 1), float(critic["output_gain"])),
        )

    def forward(self, critic_states):
        """接收 ``[B,3200]`` 状态并返回有限的 ``[B]`` 共享价值。"""
        if critic_states.ndim != 2 or critic_states.shape[1] != self.state_dim:
            raise ValueError("Critic输入形状必须为[B,3200]")
        if not torch.isfinite(critic_states).all():
            raise ValueError("Critic输入包含NaN或Inf")
        values = self.network(critic_states).squeeze(-1)
        if not torch.isfinite(values).all():
            raise RuntimeError("Critic输出包含NaN或Inf")
        return values
