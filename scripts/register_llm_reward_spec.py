"""注册已有LLM奖励规范到统一基线路径，全程不调用API。"""

import argparse
import json

from baselines.config import load_baseline_config
from baselines.reward_spec_registry import register_reward_spec


def parse_args():
    parser = argparse.ArgumentParser(description="注册已有冻结LLM奖励规范")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", default="configs/baselines.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_baseline_config(args.config)
    llm = config["methods"]["llm_ppo"]
    limits = llm["weight_limits"]
    destination = args.destination or llm["selected_reward_spec_path"]
    metadata = register_reward_spec(
        args.source,
        destination,
        limits["minimum"],
        limits["maximum"],
        force=args.force,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
