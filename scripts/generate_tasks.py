"""生成固定任务数据库；默认拒绝覆盖已有结果。"""

import argparse
from pathlib import Path

from srs_env.config import load_task_config
from srs_env.data import load_skyfield_dataset
from srs_env.tasks import (
    generate_task_database,
    load_task_database,
    load_task_splits,
    write_task_database,
)


def main():
    """读取配置和第一天数据，生成可复现的任务库。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    root = Path("data/tasks")
    if (root / "task_database.jsonl").exists() and not arguments.overwrite:
        raise SystemExit("任务数据库已存在，请使用 --overwrite")
    config = load_task_config()
    tasks = generate_task_database(config, load_skyfield_dataset())
    splits = write_task_database(tasks, config, root)
    reloaded = load_task_database(root / "task_database.jsonl")
    reloaded_splits = load_task_splits(root / "task_splits.json")
    if len(reloaded) != len(tasks) or reloaded_splits != splits:
        raise RuntimeError("任务数据库写入后重新读取不一致")
    print("任务数据库：{} 条，划分 {}".format(len(tasks), {name: len(ids) for name, ids in splits.items()}))


if __name__ == "__main__":
    main()
