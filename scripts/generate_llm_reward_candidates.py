"""安全生成严格JSON奖励候选，不启动基线训练。"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from baselines.candidate_search import CachedRewardGenerator, GenerationBudget
from baselines.config import load_baseline_config, validate_baseline_config
from baselines.live_api_confirmation import LiveApiPlan, confirm_live_api_call
from baselines.llm_prompt import build_initial_reward_prompt
from baselines.llm_provider import (
    DeepSeekRewardGenerationProvider,
    MockRewardGenerationProvider,
)
from baselines.llm_schema import default_mock_specs
from baselines.run_management import atomic_write_json
from mappo.config import load_mappo_config


def parse_args():
    parser = argparse.ArgumentParser(description="生成严格JSON奖励候选")
    parser.add_argument("--provider", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_config = deepcopy(load_baseline_config(args.config))
    mappo_config = load_mappo_config(args.mappo_config)
    validate_baseline_config(baseline_config, mappo_config)
    llm = baseline_config["methods"]["llm_ppo"]
    search = llm["search"]
    output = Path(
        args.output_dir
        or Path(baseline_config["output"]["llm_search_directory"])
        / datetime.now().strftime("%Y%m%d_%H%M%S_generate")
    )
    if output.exists():
        raise FileExistsError("候选生成目录已存在，拒绝覆盖")
    approval = None
    if args.provider == "deepseek":
        if not args.live_api:
            raise ValueError("真实DeepSeek候选生成必须显式传入--live-api")
        approval = confirm_live_api_call(
            LiveApiPlan(
                model=llm["provider"]["model"],
                rounds=args.rounds,
                candidates_per_round=args.candidates,
                candidate_training_episodes=0,
                maximum_api_calls=search["maximum_api_calls"],
                maximum_input_tokens=search["maximum_total_input_tokens"],
                maximum_output_tokens=search["maximum_total_output_tokens"],
                output_directory=str(output),
            )
        )
    output.mkdir(parents=True)
    llm["cache"]["directory"] = str(output / "api_calls")
    mock_specs = default_mock_specs()
    responses = [
        mock_specs[index % len(mock_specs)].to_dict()
        for index in range(args.rounds * args.candidates)
    ]
    provider = (
        MockRewardGenerationProvider(responses)
        if args.provider == "mock"
        else DeepSeekRewardGenerationProvider(
            llm["provider"],
            approval=approval,
        )
    )
    budget = GenerationBudget(
        search["maximum_api_calls"],
        search["maximum_total_input_tokens"],
        search["maximum_total_output_tokens"],
    )
    generator = CachedRewardGenerator(
        provider,
        llm["provider"],
        llm,
        budget,
    )
    generated = []
    index = 0
    for round_index in range(1, args.rounds + 1):
        for candidate_index in range(1, args.candidates + 1):
            index += 1
            candidate_id = "round_{0:02d}_candidate_{1:02d}".format(
                round_index,
                candidate_index,
            )
            prompt = build_initial_reward_prompt(
                mappo_config["manual_reward"]["weights"],
                {"candidate_id": candidate_id},
            )
            spec, audit = generator.generate(prompt, {"candidate_id": candidate_id})
            spec.save(output / "candidates" / candidate_id / "reward_spec.json")
            generated.append(
                {"candidate_id": candidate_id, "spec_id": spec.spec_id, **audit}
            )
    summary = {
        "provider": args.provider,
        "generated": generated,
        "budget": {
            "api_calls": budget.api_calls,
            "input_tokens": budget.input_tokens,
            "output_tokens": budget.output_tokens,
        },
    }
    atomic_write_json(output / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
