"""运行安全奖励生成、候选诊断、公平训练和validation选择。"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from baselines.config import load_baseline_config, validate_baseline_config
from baselines.live_api_confirmation import LiveApiPlan, confirm_live_api_call
from baselines.search_workflow import run_reward_search
from mappo.config import load_mappo_config


def parse_args():
    parser = argparse.ArgumentParser(description="搜索并冻结LLM-PPO奖励权重")
    parser.add_argument("--provider", choices=("mock", "deepseek"), default="deepseek")
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--candidate-episodes", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    parser.add_argument("--output-dir", default="results/baselines/llm_search/llmppo_5r8c_formal_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_config = deepcopy(load_baseline_config(args.config))
    mappo_config = load_mappo_config(args.mappo_config)
    validate_baseline_config(baseline_config, mappo_config)
    llm = baseline_config["methods"]["llm_ppo"]
    search = llm["search"]
    rounds = args.rounds or int(search["rounds"])
    candidates = args.candidates_per_round or int(search["candidates_per_round"])
    candidate_episodes = args.candidate_episodes or int(
        search["candidate_training_episodes"]
    )
    output = Path(
        args.output_dir
        or Path(baseline_config["output"]["llm_search_directory"])
        / datetime.now().strftime("%Y%m%d_%H%M%S_search")
    )
    if output.exists():
        raise FileExistsError("LLM搜索目录已存在，拒绝覆盖")
    approval = None
    if args.provider == "deepseek":
        if not args.live_api:
            raise ValueError("真实DeepSeek搜索必须显式传入--live-api")
        approval = confirm_live_api_call(
            LiveApiPlan(
                model=llm["provider"]["model"],
                rounds=rounds,
                candidates_per_round=candidates,
                candidate_training_episodes=candidate_episodes,
                maximum_api_calls=search["maximum_api_calls"],
                maximum_input_tokens=search["maximum_total_input_tokens"],
                maximum_output_tokens=search["maximum_total_output_tokens"],
                output_directory=str(output),
            )
        )
    selected, summary = run_reward_search(
        args.provider,
        baseline_config,
        mappo_config,
        output,
        rounds,
        candidates,
        candidate_episodes,
        smoke=args.smoke,
        live_api_approval=approval,
    )
    print("已选择奖励规范：{0}".format(selected))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
