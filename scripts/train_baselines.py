"""按用户指定顺序串行训练一个、多个或全部基线方法。"""

import argparse
from pathlib import Path

from baselines.config import load_baseline_config
from baselines.live_api_confirmation import LiveApiPlan, confirm_live_api_call
from baselines.methods import BaselineMethod, parse_baseline_methods
from baselines.orchestrator import BaselineOrchestrator
from baselines.run_management import generate_run_id, load_json
from baselines.search_workflow import run_reward_search
from mappo.config import load_mappo_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一训练多个基线；--episodes是每种方法的目标总Episode数"
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reward-spec")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--validation-max-steps", type=int)
    parser.add_argument("--prepare-llm-reward", action="store_true")
    parser.add_argument("--refresh-llm-reward", action="store_true")
    parser.add_argument("--llm-provider", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--llm-rounds", type=int, default=1)
    parser.add_argument("--llm-candidates", type=int, default=2)
    parser.add_argument("--candidate-episodes", type=int, default=1)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    return parser.parse_args()


def _resume_arguments(args):
    manifest = load_json(Path(args.resume_run) / "run_manifest.json")
    methods = parse_baseline_methods(manifest["methods"])
    reward_spec = args.reward_spec or manifest.get("reward_spec_path")
    return methods, reward_spec


def _prepare_reward(args, baseline_config, mappo_config):
    if args.prepare_llm_reward and args.refresh_llm_reward:
        raise ValueError("--prepare-llm-reward与--refresh-llm-reward不能同时使用")
    search_id = "prepare_{0}".format(generate_run_id())
    output = Path(baseline_config["output"]["llm_search_directory"]) / search_id
    approval = None
    if args.llm_provider == "deepseek":
        search = baseline_config["methods"]["llm_ppo"]["search"]
        provider = baseline_config["methods"]["llm_ppo"]["provider"]
        approval = confirm_live_api_call(
            LiveApiPlan(
                model=provider["model"],
                rounds=args.llm_rounds,
                candidates_per_round=args.llm_candidates,
                candidate_training_episodes=args.candidate_episodes,
                maximum_api_calls=search["maximum_api_calls"],
                maximum_input_tokens=search["maximum_total_input_tokens"],
                maximum_output_tokens=search["maximum_total_output_tokens"],
                output_directory=str(output),
            )
        )
    selected, _ = run_reward_search(
        args.llm_provider,
        baseline_config,
        mappo_config,
        output,
        args.llm_rounds,
        args.llm_candidates,
        args.candidate_episodes,
        smoke=False,
        live_api_approval=approval,
    )
    return str(selected)


def main():
    args = parse_args()
    if args.resume_run:
        if args.methods:
            raise ValueError("Resume从Manifest读取方法顺序，不得再次传--methods")
        if args.run_name:
            raise ValueError("Resume不得创建新的run-name")
        methods, reward_spec = _resume_arguments(args)
    else:
        if not args.methods:
            raise ValueError("新Run必须通过--methods选择方法")
        methods = parse_baseline_methods(args.methods)
        reward_spec = args.reward_spec
    prepare = args.prepare_llm_reward or args.refresh_llm_reward
    if prepare:
        if BaselineMethod.LLM_PPO not in methods:
            raise ValueError("准备LLM奖励时必须选择llm_ppo方法")
        baseline_config = load_baseline_config(args.config)
        mappo_config = load_mappo_config(args.mappo_config)
        reward_spec = _prepare_reward(args, baseline_config, mappo_config)
    orchestrator = BaselineOrchestrator(
        args.config,
        args.mappo_config,
        device=args.device,
        seed=args.seed,
    )
    run_directory, summary, _ = orchestrator.run(
        methods,
        args.episodes,
        reward_spec_path=reward_spec,
        run_name=args.run_name,
        resume_run=args.resume_run,
        continue_on_error=args.continue_on_error,
        skip_validation=args.skip_validation,
        max_steps_per_episode=args.max_steps_per_episode,
        validation_max_steps=args.validation_max_steps,
    )
    print("Run结果：{0}".format(run_directory))
    print("总体状态：{0}".format(summary["status"]))


if __name__ == "__main__":
    main()
