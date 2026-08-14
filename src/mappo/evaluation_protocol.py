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
    definition = dict(protocol_config[name])
    task_count = int(definition["task_count"])
    validation_ids = list(task_splits["validation"])
    if name == "checkpoint_selection" and task_count > len(validation_ids):
        required = task_count - len(validation_ids)
        train_ids = list(task_splits["train"])
        if required > len(train_ids):
            raise ValueError("train划分不足以扩展validation任务池")
        # 仅在正式validation容量不足时，确定性地把部分任务从train划分转入
        # validation划分；环境随后也只能从更新后的train划分采样，避免泄漏。
        transferred = sorted(train_ids)[-required:]
        transferred_set = set(transferred)
        task_splits["train"] = [
            task_id for task_id in train_ids if task_id not in transferred_set
        ]
        task_splits["validation"] = validation_ids + transferred
        validation_ids = list(task_splits["validation"])
    if not validation_ids:
        raise ValueError("validation划分不能为空")
    rng = np.random.RandomState(split_seed)
    shuffled = np.asarray(validation_ids, dtype=object)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    pools = {
        "reward_search": shuffled[:midpoint].tolist(),
        # 正式模型选择可使用完整validation划分，因此可与训练任务数同步。
        "checkpoint_selection": shuffled.tolist(),
        "test": list(task_splits["test"]),
    }
    pool_ids = pools[name]
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
