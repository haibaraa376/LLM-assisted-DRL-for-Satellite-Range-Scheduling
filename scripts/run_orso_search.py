"""运行ORSO：一次生成固定奖励集合，再以D3RB分配MAPPO Episode预算。"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from baselines.config import load_baseline_config, validate_baseline_config
from baselines.live_api_confirmation import LiveApiPlan, confirm_live_api_call
from mappo.config import load_mappo_config
from orso.config import load_orso_config
from orso.search import run_orso_search


def parse_args():
    parser = argparse.ArgumentParser(description="执行ORSO D3RB奖励搜索")
    parser.add_argument("--provider", choices=("mock", "deepseek"), default="deepseek")
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--config", default="configs/orso.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    orso_config = load_orso_config(args.config)
    baseline_config = deepcopy(load_baseline_config(args.baseline_config))
    if args.device:
        baseline_config["device"] = args.device
    mappo_config = load_mappo_config(args.mappo_config)
    validate_baseline_config(baseline_config, mappo_config)
    output = Path(
        args.output_dir
        or Path(orso_config["output"]["root"])
        / ("orso_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    if output.exists():
        raise FileExistsError("ORSO搜索目录已存在，拒绝覆盖")
    approval = None
    if args.provider == "deepseek":
        if not args.live_api:
            raise ValueError("真实DeepSeek搜索必须显式传入--live-api")
        llm = baseline_config["methods"]["llm_ppo"]
        approval = confirm_live_api_call(
            LiveApiPlan(
                model=llm["provider"]["model"],
                rounds=orso_config["generation"]["rounds"],
                candidates_per_round=orso_config["generation"]["candidates"],
                candidate_training_episodes=orso_config["training"]["max_episodes_per_candidate"],
                maximum_api_calls=llm["search"]["maximum_api_calls"],
                maximum_input_tokens=llm["search"]["maximum_total_input_tokens"],
                maximum_output_tokens=llm["search"]["maximum_total_output_tokens"],
                output_directory=str(output),
            )
        )
    reward_path, summary = run_orso_search(
        args.provider,
        orso_config,
        baseline_config,
        mappo_config,
        output,
        smoke=args.smoke,
        live_api_approval=approval,
    )
    print("已选择ORSO奖励规范：{0}".format(reward_path))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
