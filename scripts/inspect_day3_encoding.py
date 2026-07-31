"""检查第三天固定维度编码、目标顺序和基础动作Mask。"""

import json
from pathlib import Path

import numpy as np

from mappo.config import load_mappo_config
from mappo.encoding import MappoObservationEncoder
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.tasks import load_task_database, load_task_splits, sample_episode_tasks


def build_environment(config):
    """按validation划分、20任务和种子2025构建只用于检查的环境。"""
    dataset = load_skyfield_dataset()
    database = load_task_database()
    split_ids = load_task_splits()[config["smoke_training"]["split"]]
    tasks = sample_episode_tasks(
        database,
        split_ids,
        config["smoke_training"]["task_count"],
        config["seed"],
    )
    environment = CrossDomainSatelliteRangeSchedulingEnv(
        dataset,
        load_environment_config(),
        tasks,
    )
    observations, _ = environment.reset(seed=config["seed"])
    # 随机validation任务通常不在0秒到达；推进空动作直到首个任务可观测，
    # 这样人工摘要能展示真实任务槽和Mask，而不是只展示四个空槽。
    while not any(
        observation["candidate_tasks"] for observation in observations.values()
    ):
        observations, _, terminated, _, _ = environment.step({})
        if terminated:
            raise RuntimeError("validation样本在整个episode内都没有可编码任务")
    return environment, observations


def main():
    """执行编码检查，打印一个样本并写入紧凑JSON摘要。"""
    config = load_mappo_config()
    environment, observations = build_environment(config)
    encoder = MappoObservationEncoder(environment, config)
    encoded = encoder.encode_all_agents(observations)
    critic_state = encoder.encode_critic_state(
        encoded,
        environment.get_global_state(),
    )
    sample_id = next(
        (
            satellite_id
            for satellite_id in environment.dataset.satellite_ids
            if any(task_id is not None for task_id in encoded[satellite_id].task_ids)
        ),
        environment.dataset.satellite_ids[0],
    )
    sample = encoded[sample_id]
    available = [
        [
            "IDLE" if choice == 0 else sample.target_ids[choice - 1]
            for choice in np.flatnonzero(sample.base_target_mask[slot])
        ]
        for slot in range(4)
    ]
    print("样本卫星：", sample_id)
    print("身份特征：", sample.observation[:7].tolist())
    print("自身状态：", sample.observation[7:10].tolist())
    print("任务槽：", list(sample.task_ids))
    print("目标顺序：", list(sample.target_ids))
    print("各槽可选目标：", available)

    summary = {
        "actor_observation_dim": int(sample.observation.shape[0]),
        "critic_state_dim": int(critic_state.shape[0]),
        "agent_count": len(encoded),
        "candidate_task_count": 4,
        "target_slot_count": len(sample.target_ids),
        "target_choice_count": sample.base_target_mask.shape[1],
        "all_features_finite": bool(
            all(np.all(np.isfinite(item.observation)) for item in encoded.values())
            and np.all(np.isfinite(critic_state))
        ),
        "all_masks_have_idle": bool(
            all(np.all(item.base_target_mask[:, 0]) for item in encoded.values())
        ),
        "satellite_ids": list(environment.dataset.satellite_ids),
        "ground_station_ids": list(environment.dataset.ground_station_ids),
        "distance_reference_km": encoder.distance_reference_km,
        "sample_satellite_id": sample_id,
        "sample_task_ids": list(sample.task_ids),
        "sample_valid_choice_counts": sample.base_target_mask.sum(axis=1).tolist(),
    }
    output = Path("results/day3/encoding_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
