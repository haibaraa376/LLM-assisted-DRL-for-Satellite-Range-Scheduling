"""通过总体编排器训练单个基线方法。"""

import argparse

from baselines.methods import parse_baseline_method
from baselines.orchestrator import BaselineOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="训练一个统一基线方法")
    parser.add_argument("--method", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--reward-spec")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps-per-episode", type=int)
    parser.add_argument("--validation-max-steps", type=int)
    parser.add_argument("--config", default="configs/baselines.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    method = parse_baseline_method(args.method)
    if args.resume and args.output_dir:
        raise ValueError("Resume使用--resume指定Run目录，不能再传--output-dir")
    orchestrator = BaselineOrchestrator(
        args.config,
        args.mappo_config,
        device=args.device,
        seed=args.seed,
    )
    run_directory, summary, _ = orchestrator.run(
        [method],
        args.episodes,
        reward_spec_path=args.reward_spec,
        run_name=args.run_name,
        resume_run=args.resume,
        skip_validation=args.skip_validation,
        max_steps_per_episode=args.max_steps_per_episode,
        validation_max_steps=args.validation_max_steps,
        explicit_run_directory=args.output_dir,
    )
    print("Run结果：{0}".format(run_directory))
    print("总体状态：{0}".format(summary["status"]))


if __name__ == "__main__":
    main()
