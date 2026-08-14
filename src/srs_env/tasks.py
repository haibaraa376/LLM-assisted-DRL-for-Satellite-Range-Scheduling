"""生成、保存、读取和确定性采样第二天任务数据库。"""

import json
from pathlib import Path

import numpy as np

from .models import TaskDefinition


def survival_time(priority, config):
    """按优先级返回T1至T4生存时间，单位为秒。"""
    values = config["task_database"]["survival_time_seconds"]
    if priority <= 3:
        return values["priority_1_to_3"]
    if priority <= 6:
        return values["priority_4_to_6"]
    if priority <= 9:
        return values["priority_7_to_9"]
    return values["priority_10"]


def generate_task_database(config, dataset):
    """使用固定随机种子生成任务定义列表，不写入文件。"""
    settings = config["task_database"]
    rng = np.random.RandomState(settings["generation_seed"])
    sources = [satellite_id for satellite_id, domain_id in zip(dataset.satellite_ids, dataset.satellite_domain_ids) if domain_id == "D2"]
    tasks = []
    for number in range(settings["database_size"]):
        priority = int(rng.randint(1, 11))
        arrival_slot = rng.randint(settings["arrival"]["minimum_time_s"] // 30, settings["arrival"]["maximum_time_s"] // 30 + 1)
        arrival_time_s = float(arrival_slot * 30)
        size_settings = settings["data_size_mbit"]
        data_size_mbit = float(np.exp(rng.uniform(np.log(size_settings["minimum"]), np.log(size_settings["maximum"]))))
        lifetime = float(survival_time(priority, config))
        tasks.append(TaskDefinition("task_{:06d}".format(number + 1), sources[int(rng.randint(len(sources)))], dataset.ground_station_ids[int(rng.randint(4))], priority, data_size_mbit, lifetime, arrival_time_s, arrival_time_s + lifetime, data_size_mbit / 60.0))
    return tasks


def write_task_database(tasks, config, root=Path("data/tasks")):
    """写入JSONL、互斥划分和简洁摘要，返回划分字典。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "task_database.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for task in tasks:
            stream.write(json.dumps(task.__dict__, ensure_ascii=False, sort_keys=True) + "\n")
    ids = np.array([task.task_id for task in tasks])
    rng = np.random.RandomState(config["task_database"]["split_seed"])
    rng.shuffle(ids)
    split_sizes = config["task_database"]["split"]
    required_names = ("train", "reward_search", "checkpoint_selection", "test")
    if set(split_sizes) != set(required_names):
        raise ValueError("任务划分必须恰好包含train、reward_search、checkpoint_selection、test")
    if any(int(split_sizes[name]) != split_sizes[name] or int(split_sizes[name]) <= 0 for name in required_names):
        raise ValueError("任务划分数量必须为正整数")
    if sum(int(split_sizes[name]) for name in required_names) != len(ids):
        raise ValueError("任务划分数量之和必须等于任务数据库大小")
    splits = {}
    offset = 0
    for name in required_names:
        next_offset = offset + int(split_sizes[name])
        splits[name] = ids[offset:next_offset].tolist()
        offset = next_offset
    (root / "task_splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    data_sizes = np.asarray([task.data_size_mbit for task in tasks])
    arrivals = np.asarray([task.arrival_time_s for task in tasks])
    summary = {
        "task_count": len(tasks),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "priority_counts": {
            str(priority): sum(task.priority == priority for task in tasks)
            for priority in range(1, 11)
        },
        "source_satellite_counts": {
            source: sum(task.source_satellite_id == source for task in tasks)
            for source in sorted({task.source_satellite_id for task in tasks})
        },
        "target_ground_station_counts": {
            target: sum(task.target_ground_station_id == target for task in tasks)
            for target in sorted({task.target_ground_station_id for task in tasks})
        },
        "data_size_mbit": {
            "minimum": float(data_sizes.min()),
            "mean": float(data_sizes.mean()),
            "maximum": float(data_sizes.max()),
        },
        "arrival_time_s": {
            "minimum": float(arrivals.min()),
            "maximum": float(arrivals.max()),
        },
        "survival_time_seconds": config["task_database"]["survival_time_seconds"],
        "generation_seed": config["task_database"]["generation_seed"],
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


def load_task_database(path=Path("data/tasks/task_database.jsonl")):
    """读取JSONL任务库并按任务ID返回定义字典。"""
    return {record["task_id"]: TaskDefinition(**record) for record in (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())}


def load_task_splits(path=Path("data/tasks/task_splits.json")):
    """读取train、validation、test任务ID划分。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_episode_tasks(database, split_ids, task_count, seed):
    """无放回采样一个episode，并按到达/优先级/ID稳定排序。"""
    rng = np.random.RandomState(seed)
    selected = rng.choice(split_ids, size=task_count, replace=False)
    return sorted((database[task_id] for task_id in selected), key=lambda task: (task.arrival_time_s, -task.priority, task.task_id))
