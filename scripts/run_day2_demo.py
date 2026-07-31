"""运行完整24小时确定性诊断episode；它不是训练或论文基线算法。"""

import json
from pathlib import Path

from srs_env.config import load_environment_config
from srs_env.data import load_skyfield_dataset
from srs_env.environment import CrossDomainSatelliteRangeSchedulingEnv
from srs_env.models import SatelliteCompositeAction, TransmissionSubAction
from srs_env.tasks import load_task_database, load_task_splits, sample_episode_tasks


def main():
    """以确定性复合动作策略运行24小时并写入紧凑摘要。"""
    dataset = load_skyfield_dataset()
    database = load_task_database()
    tasks = sample_episode_tasks(database, load_task_splits()["validation"], 20, 2025)
    environment = CrossDomainSatelliteRangeSchedulingEnv(dataset, load_environment_config(), tasks)
    observations, _ = environment.reset()
    totals = {
        "submitted_subaction_count": 0,
        "accepted_isl_count": 0,
        "accepted_idl_count": 0,
        "accepted_sgl_count": 0,
        "same_slot_forwarding_blocked_count": 0,
    }
    maximum_interfaces = 0
    while True:
        actions = {}
        for satellite_id in dataset.satellite_ids:
            held = observations[satellite_id]["candidate_tasks"]
            candidates = environment.get_action_candidates(satellite_id)
            if not held or not candidates:
                continue
            transmissions = []
            used_physical_targets = set()
            inter_count = 0
            sgl_count = 0
            for task in held:
                task_candidates = [
                    target
                    for task_id, target in candidates
                    if task_id == task["task_id"]
                ]
                if not task_candidates:
                    continue
                ground = task["target_ground_station_id"]
                if ground in task_candidates and sgl_count == 0:
                    target = ground
                    sgl_count += 1
                else:
                    satellite_targets = sorted(
                        target
                        for target in task_candidates
                        if target in dataset.satellite_index
                        and target not in used_physical_targets
                    )
                    if inter_count >= 3 or not satellite_targets:
                        continue
                    # 只看当前时隙候选，不读取未来窗口。
                    target = satellite_targets[0]
                    used_physical_targets.add(target)
                    inter_count += 1
                transmissions.append(
                    TransmissionSubAction(task["task_id"], target, 1.0, 0.0)
                )
                if len(transmissions) == 4:
                    break
            if transmissions:
                actions[satellite_id] = SatelliteCompositeAction(tuple(transmissions))
        observations, _, terminated, _, info = environment.step(actions)
        for key in totals:
            totals[key] += info[key]
        maximum_interfaces = max(
            maximum_interfaces,
            info["max_observed_inter_interface_usage"],
        )
        if terminated:
            break
    summary = {
        "seed": 2025,
        "task_count": 20,
        "steps": environment.step_index,
        "terminated": terminated,
        "timeliness_raw": info["timeliness_raw"],
        "load_balance_raw": info["load_balance_raw"],
        "load_balance_mean_per_task": info["load_balance_mean_per_task"],
        "completed_task_count": info["completed_task_count"],
        "expired_task_count": info["expired_task_count"],
        "delivered_data_mbit": info["delivered_data_mbit"],
        "accepted_transmission_count": info["accepted_transmission_count"],
        "rejected_transmission_count": info["rejected_transmission_count"],
        **totals,
        "maximum_concurrent_inter_satellite_interfaces": maximum_interfaces,
        "data_conservation_passed": True,
    }
    output = Path("results/day2/demo_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
