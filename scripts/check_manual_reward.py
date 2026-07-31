"""检查人工奖励方向，并可选运行三个完整validation策略Episode。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from mappo.config import load_mappo_config
from mappo.encoding import MappoObservationEncoder, decode_composite_action
from mappo.manual_reward import ManualReward
from mappo.networks import SharedActor
from mappo.utils import set_global_seed
from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.models import (
    SatelliteCompositeAction,
    TaskStatus,
    TransmissionRecord,
    TransmissionSubAction,
)
from srs_env.tasks import load_task_database, load_task_splits


def _environment():
    """构造可按现有划分重置的只读数据环境。"""
    return CrossDomainSatelliteRangeSchedulingEnv(
        load_skyfield_dataset(),
        load_environment_config(),
        task_database=load_task_database(),
        task_splits=load_task_splits(),
    )


def _record(task, source, target, link_type, amount, accepted=True, code=()):
    """创建奖励方向检查使用的最小传输记录。"""
    return TransmissionRecord(
        source_satellite_id=source,
        subaction_index=0,
        composite_source_id=source,
        target_id=target,
        task_id=task.task_id,
        link_type=link_type,
        requested_ratio=amount / task.data_size_mbit,
        requested_start_s=task.arrival_time_s,
        accepted=accepted,
        transmitted_data_mbit=amount if accepted else 0.0,
        actual_start_s=task.arrival_time_s if accepted else None,
        actual_end_s=task.arrival_time_s + amount / 60.0 if accepted else None,
        rate_mbps=60.0 if link_type == "SGL" else 80.0,
        violation_codes=tuple(code),
        projected=False,
    )


def _info(records):
    """构造奖励提取所需的时隙info。"""
    return {
        "transmission_records": records,
        "submitted_subaction_count": len(records),
    }


def quick_diagnostics(config):
    """执行空闲、SGL、中继、完成、过期、冲突与循环中继方向检查。"""
    environment = _environment()
    database = load_task_database()
    task_id = max(
        load_task_splits()["validation"],
        key=lambda item: (database[item].priority, item),
    )
    environment.reset(task_ids=[task_id])
    state = environment.tasks[task_id]
    state.status = TaskStatus.ACTIVE
    task = state.definition
    reward = ManualReward(config["manual_reward"])

    reward.reset(environment)
    idle = reward.compute(environment, _info([]))
    amount = task.data_size_mbit * 0.1

    reward.reset(environment)
    sgl = reward.compute(
        environment,
        _info(
            [
                _record(
                    task,
                    task.source_satellite_id,
                    task.target_ground_station_id,
                    "SGL",
                    amount,
                )
            ]
        ),
    )
    reward.reset(environment)
    relay = reward.compute(
        environment,
        _info([_record(task, task.source_satellite_id, "cs01", "IDL", amount)]),
    )

    state.status = TaskStatus.ACTIVE
    state.delivered_to_ground_mbit = 0.0
    reward.reset(environment)
    state.delivered_to_ground_mbit = task.data_size_mbit
    state.status = TaskStatus.COMPLETED
    completion = reward.compute(environment, _info([]))
    repeated_completion = reward.compute(environment, _info([]))

    state.status = TaskStatus.ACTIVE
    state.delivered_to_ground_mbit = task.data_size_mbit * 0.25
    reward.reset(environment)
    state.status = TaskStatus.EXPIRED
    expiration = reward.compute(environment, _info([]))
    repeated_expiration = reward.compute(environment, _info([]))

    state.status = TaskStatus.ACTIVE
    state.delivered_to_ground_mbit = 0.0
    reward.reset(environment)
    conflict = reward.compute(
        environment,
        _info(
            [
                _record(
                    task,
                    task.source_satellite_id,
                    "cs01",
                    "IDL",
                    amount,
                    accepted=False,
                    code=("TASK_ALREADY_SCHEDULED_THIS_SLOT",),
                )
            ]
        ),
    )
    loop_net = 2.0 * relay.total_reward
    checks = {
        "idle_reward_zero": idle.total_reward == 0.0,
        "idle_balance_zero": idle.features.balance_score == 0.0,
        "sgl_positive": sgl.features.sgl_progress > 0 and sgl.total_reward > 0,
        "relay_weak_positive_with_cost": (
            relay.features.relay_progress > 0
            and relay.features.relay_cost > 0
            and relay.total_reward > 0
            and relay.total_reward < sgl.total_reward
        ),
        "completion_once": (
            completion.features.completion_score > 0
            and repeated_completion.features.completion_score == 0
        ),
        "expiration_once": (
            expiration.features.expiration_loss > 0
            and expiration.total_reward < 0
            and repeated_expiration.features.expiration_loss == 0
        ),
        "conflict_negative": (
            conflict.features.coordination_conflict_rate > 0
            and conflict.total_reward < 0
        ),
        "relay_loop_weaker_than_sgl": loop_net < sgl.total_reward,
        "all_rewards_finite": bool(
            np.all(
                np.isfinite(
                    [
                        idle.total_reward,
                        sgl.total_reward,
                        relay.total_reward,
                        completion.total_reward,
                        expiration.total_reward,
                        conflict.total_reward,
                    ]
                )
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("人工奖励快速方向检查失败：{0}".format(checks))
    return {
        "checks": checks,
        "samples": {
            "idle": idle.as_dict(),
            "sgl": sgl.as_dict(),
            "relay": relay.as_dict(),
            "completion": completion.as_dict(),
            "expiration": expiration.as_dict(),
            "conflict": conflict.as_dict(),
        },
    }


def _heuristic_actions(environment, observations):
    """构造保守且确定的Day2风格启发式动作。

    诊断的目的不是追求吞吐量，而是验证“有效下传优于空闲”的奖励方向。
    因此每个时隙只选择一个全局最优SGL候选，避免把多星独立贪心产生的
    地面站冲突误当作奖励设计问题。
    """
    ranked_candidates = []
    for satellite_id in environment.dataset.satellite_ids:
        candidates = environment.get_action_candidates(satellite_id)
        for task in observations[satellite_id]["candidate_tasks"]:
            ground = task["target_ground_station_id"]
            if (task["task_id"], ground) not in candidates:
                continue
            ranked_candidates.append(
                (
                    -float(task["priority"]),
                    float(task["remaining_lifetime_s"]),
                    task["task_id"],
                    satellite_id,
                    ground,
                )
            )
    if not ranked_candidates:
        return {}

    _, _, task_id, satellite_id, ground = min(ranked_candidates)
    transmission = TransmissionSubAction(task_id, ground, 1.0, 0.0)
    return {
        satellite_id: SatelliteCompositeAction((transmission,)),
    }


def full_episode_diagnostics(config):
    """比较全空闲、Masked随机和启发式三个完整validation Episode。"""
    results = {}
    for policy_name in ("idle", "masked_random", "heuristic"):
        environment = _environment()
        observations, _ = environment.reset(
            seed=config["seed"],
            split="validation",
            task_count=20,
        )
        reward = ManualReward(config["manual_reward"])
        reward.reset(environment)
        actor = None
        encoder = None
        if policy_name == "masked_random":
            set_global_seed(config["seed"])
            encoder = MappoObservationEncoder(environment, config)
            actor = SharedActor(config).cpu()
        total_reward = 0.0
        component_sums = {}
        final_info = None
        while not environment.terminated:
            if policy_name == "idle":
                actions = {}
            elif policy_name == "heuristic":
                actions = _heuristic_actions(environment, observations)
            else:
                encoded = encoder.encode_all_agents(observations)
                observation_array = np.stack(
                    [encoded[item].observation for item in encoder.satellite_ids]
                )
                masks = np.stack(
                    [encoded[item].base_target_mask for item in encoder.satellite_ids]
                )
                with torch.no_grad():
                    batch = actor.sample_actions(
                        torch.as_tensor(observation_array),
                        torch.as_tensor(masks),
                    )
                choices = batch.target_choices.numpy()
                bounded = batch.bounded_continuous_actions.numpy()
                actions = {
                    satellite_id: decode_composite_action(
                        encoded[satellite_id],
                        choices[index],
                        bounded[index],
                    )
                    for index, satellite_id in enumerate(encoder.satellite_ids)
                }
            observations, _, _, _, final_info = environment.step(actions)
            breakdown = reward.compute(environment, final_info)
            total_reward += breakdown.total_reward
            for name, value in {
                **asdict(breakdown.features),
                "total_reward": breakdown.total_reward,
            }.items():
                component_sums[name] = component_sums.get(name, 0.0) + value
        environment.check_data_conservation()
        results[policy_name] = {
            "total_manual_reward": total_reward,
            "reward_component_sums": component_sums,
            "timeliness_raw": final_info["timeliness_raw"],
            "load_balance_mean_per_task": final_info[
                "load_balance_mean_per_task"
            ],
            "completed_task_count": final_info["completed_task_count"],
            "expired_task_count": final_info["expired_task_count"],
            "accepted_subaction_count": final_info[
                "accepted_transmission_count"
            ],
            "rejected_subaction_count": final_info[
                "rejected_transmission_count"
            ],
            "data_conservation_passed": True,
        }
    if results["idle"]["total_manual_reward"] > 1.0e-9:
        raise RuntimeError("全空闲完整Episode累计人工奖励必须非正")
    if results["heuristic"]["total_manual_reward"] <= results["idle"][
        "total_manual_reward"
    ]:
        raise RuntimeError("启发式策略人工奖励未优于全空闲策略")
    return results


def main():
    """运行快速方向检查，并在指定时附加完整Episode策略比较。"""
    parser = argparse.ArgumentParser(description="检查人工奖励方向")
    parser.add_argument("--full-episode", action="store_true")
    args = parser.parse_args()
    config = load_mappo_config()
    result = quick_diagnostics(config)
    if args.full_episode:
        result["full_episode"] = full_episode_diagnostics(config)
    output = Path(
        "results/day3/manual_reward_mappo/reward_diagnostics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["checks"], ensure_ascii=False))


if __name__ == "__main__":
    main()
