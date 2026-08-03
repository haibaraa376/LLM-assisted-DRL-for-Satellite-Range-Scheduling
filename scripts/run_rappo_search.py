"""RAPPO安全入口：默认Mock，先检索再构建奖励Prompt。"""

import argparse
from pathlib import Path
import yaml

from mappo.config import load_mappo_config
from rappo.search_workflow import build_retrieved_reward_prompt


def main():
    parser = argparse.ArgumentParser(description="RAPPO检索增强奖励搜索入口")
    parser.add_argument("--provider", default="mock", choices=("mock", "deepseek"))
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--rappo-config", default="configs/rappo.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--candidate-episodes", type=int, default=1)
    args = parser.parse_args()
    if args.provider != "mock" or args.live_api:
        raise ValueError("本阶段入口默认且仅允许Mock；真实API必须人工另行批准")
    config = yaml.safe_load(Path(args.rappo_config).read_text(encoding="utf-8"))["rappo"]
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("RAPPO输出目录已存在，拒绝覆盖")
    output.mkdir(parents=True)
    prompt, retrieval = build_retrieved_reward_prompt(
        config,
        load_mappo_config(args.mappo_config)["manual_reward"]["weights"],
        "如何避免策略只做星间中继而不进行最终下传？",
        output / "retrievals.jsonl",
    )
    (output / "prompt_preview.txt").write_text(prompt, encoding="utf-8")
    print({"provider": "mock", "knowledge_base_version": retrieval["knowledge_base_version"], "retrieval_count": len(retrieval["results"]), "smoke": args.smoke, "rounds": args.rounds, "candidates_per_round": args.candidates_per_round})


if __name__ == "__main__":
    main()
