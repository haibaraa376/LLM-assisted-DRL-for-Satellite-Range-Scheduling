"""保存和恢复单个第三天MAPPO Checkpoint。"""

from pathlib import Path
import random

import numpy as np
import torch


_METADATA_KEYS = (
    "actor_observation_dim",
    "critic_state_dim",
    "candidate_task_count",
    "target_choice_count",
    "satellite_ids",
    "ground_station_ids",
)


def save_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    config,
    update_index,
    encoder_metadata,
    training_state=None,
    rng_state=None,
    validation_result=None,
):
    """保存网络、优化器、配置和固定编码顺序，只生成调用方指定的一个文件。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "config": config,
        "update_index": int(update_index),
        "encoder_metadata": encoder_metadata,
        "torch_version": torch.__version__,
        "training_state": training_state,
        "rng_state": rng_state,
        "validation_result": validation_result,
    }
    torch.save(payload, output)


def load_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer=None,
    critic_optimizer=None,
    expected_encoder_metadata=None,
    map_location="cpu",
):
    """加载Checkpoint并验证维度、任务槽及卫星/地面站顺序。"""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    metadata = checkpoint.get("encoder_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint缺少编码元数据")
    fixed_expected = {
        "actor_observation_dim": 213,
        "critic_state_dim": 3200,
        "candidate_task_count": 4,
        "target_choice_count": 20,
    }
    for key, expected in fixed_expected.items():
        if metadata.get(key) != expected:
            raise ValueError("Checkpoint编码维度不符合第三天固定设计：{0}".format(key))
    if expected_encoder_metadata is not None:
        for key in _METADATA_KEYS:
            if metadata.get(key) != expected_encoder_metadata.get(key):
                raise ValueError("Checkpoint编码元数据不匹配：{0}".format(key))
    try:
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("Checkpoint网络结构与当前模型不一致") from error
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    return checkpoint


def capture_rng_state(trainer_rng):
    """捕获Python、NumPy、Trainer、PyTorch CPU及可用CUDA随机状态。"""
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "trainer_generator_state": trainer_rng.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(rng_state, trainer_rng):
    """恢复Checkpoint中的全部随机状态；旧Checkpoint无状态时明确拒绝。"""
    if not isinstance(rng_state, dict):
        raise ValueError("正式训练Checkpoint缺少随机状态")
    required = {
        "python_random_state",
        "numpy_random_state",
        "trainer_generator_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    if not required.issubset(rng_state):
        raise ValueError("Checkpoint随机状态字段不完整")
    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])
    trainer_rng.bit_generator.state = rng_state["trainer_generator_state"]
    torch.set_rng_state(rng_state["torch_cpu_rng_state"].cpu())
    cuda_states = rng_state["torch_cuda_rng_states"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint包含CUDA随机状态，但当前CUDA不可用")
        # 使用CUDA map_location加载Checkpoint时，状态张量也可能被迁移到GPU；
        # PyTorch的CUDA RNG恢复接口仍要求传入CPU ByteTensor。
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
