"""构建互相隔离、可审计的奖励搜索、Checkpoint和测试协议。"""

import hashlib
import json

import numpy as np


def _sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_evaluation_protocol(name, protocol_config, task_splits):
    """从运行时任务划分生成指定协议及其不可变审计元数据。"""
    if name not in {"reward_search", "checkpoint_selection", "test"}:
        raise ValueError("未知评估协议：{0}".format(name))
    split_seed = int(protocol_config["split_seed"])
    validation_ids = list(task_splits["validation"])
    if len(validation_ids) != 150:
        raise ValueError("validation划分必须恰好包含150个任务")
    rng = np.random.RandomState(split_seed)
    shuffled = np.asarray(validation_ids, dtype=object)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    pools = {
        "reward_search": shuffled[:midpoint].tolist(),
        "checkpoint_selection": shuffled[midpoint:].tolist(),
        "test": list(task_splits["test"]),
    }
    definition = dict(protocol_config[name])
    pool_ids = pools[name]
    task_count = int(definition["task_count"])
    seeds = [int(seed) for seed in definition["seeds"]]
    if task_count <= 0 or task_count > len(pool_ids):
        raise ValueError("协议任务数超过对应任务池")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("协议seeds必须非空且互不重复")
    source = "validation" if name != "test" else "test"
    metadata = {
        "protocol_name": name,
        "source_split": source,
        "pool_size": len(pool_ids),
        "pool_sha256": _sha256_json(pool_ids),
        "protocol_config_sha256": _sha256_json(
            {
                "name": name,
                "split_seed": split_seed,
                "definition": definition,
            }
        ),
        "task_count": task_count,
        "seeds": seeds,
        "deterministic": True,
    }
    return {**metadata, "pool_task_ids": pool_ids}


def sample_protocol_task_ids(protocol, seed):
    """从协议私有池无放回采样，并保持返回ID稳定排序。"""
    pool = list(protocol["pool_task_ids"])
    rng = np.random.RandomState(int(seed))
    selected = rng.choice(pool, size=int(protocol["task_count"]), replace=False)
    return sorted(str(task_id) for task_id in selected)


def public_protocol_metadata(protocol):
    """移除完整任务ID池，仅暴露复现实验所需哈希和参数。"""
    return {
        key: value
        for key, value in protocol.items()
        if key != "pool_task_ids"
    }
