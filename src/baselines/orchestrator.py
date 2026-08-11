"""顺序编排单个、多个或全部基线方法的统一训练Run。"""

import gc
from pathlib import Path
import time

import torch

from mappo.config import load_mappo_config
from mappo.training_runner import BaselineTrainingRunner

from .baseline_runner import (
    build_baseline_components,
    make_runner,
    restore_baseline_checkpoint,
)
from .comparison import build_comparison
from .config import load_baseline_config, validate_baseline_config
from .llm_schema import LlmRewardSpec
from .methods import BaselineMethod, display_name
from .reward_spec_registry import register_reward_spec
from .run_management import (
    TrainingLock,
    atomic_write_json,
    generate_run_id,
    load_json,
    sha256_file,
    utc_now,
    validate_run_name,
)


class BaselineOrchestrator:
    """维护Manifest、锁、方法状态、公平初始化、失败策略和Resume。"""

    def __init__(
        self,
        baseline_config_path="configs/baselines.yaml",
        mappo_config_path="configs/mappo.yaml",
        method_executor=None,
        device=None,
        seed=None,
    ):
        self.baseline_config_path = Path(baseline_config_path)
        self.mappo_config_path = Path(mappo_config_path)
        self.baseline_config = load_baseline_config(self.baseline_config_path)
        self.mappo_config = load_mappo_config(self.mappo_config_path)
        if device is not None:
            self.baseline_config["device"] = device
        if seed is not None:
            if seed < 0:
                raise ValueError("共同seed必须是非负整数")
            self.baseline_config["seed"] = seed
        validate_baseline_config(self.baseline_config, self.mappo_config)
        self.method_executor = method_executor or self._execute_method

    def run(
        self,
        methods,
        episodes,
        reward_spec_path=None,
        run_name=None,
        resume_run=None,
        continue_on_error=False,
        skip_validation=False,
        max_steps_per_episode=None,
        validation_max_steps=None,
        explicit_run_directory=None,
    ):
        """按输入顺序串行训练，并在每次状态变化后原子更新Manifest。"""
        methods = [BaselineMethod(method) for method in methods]
        if episodes <= 0:
            raise ValueError("每种方法目标Episode数必须为正数")
        reward_spec, resolved_spec_path = self._resolve_reward_spec(
            methods,
            reward_spec_path,
        )
        if resume_run:
            run_directory = Path(resume_run).resolve()
            manifest = self._load_resume_manifest(
                run_directory,
                methods,
                reward_spec,
            )
            methods = [BaselineMethod(item) for item in manifest["methods"]]
            manifest["episodes_per_method"] = episodes
        else:
            run_id = validate_run_name(run_name) if run_name else generate_run_id()
            run_directory = (
                Path(explicit_run_directory).resolve()
                if explicit_run_directory
                else (
                    Path(self.baseline_config["output"]["runs_directory"])
                    / run_id
                ).resolve()
            )
            if run_directory.exists():
                raise FileExistsError(
                    "Run目录已存在，拒绝覆盖：{0}".format(run_directory)
                )
            run_directory.mkdir(parents=True)
            if reward_spec is not None:
                registered_spec_path = (
                    run_directory
                    / "reward_specs"
                    / "selected_reward_spec.json"
                )
                register_reward_spec(
                    resolved_spec_path,
                    registered_spec_path,
                )
                resolved_spec_path = str(registered_spec_path.resolve())
            manifest = self._new_manifest(
                run_id,
                methods,
                episodes,
                resolved_spec_path,
                reward_spec,
            )
            atomic_write_json(run_directory / "run_manifest.json", manifest)

        self._print_plan(
            run_directory,
            methods,
            episodes,
            reward_spec,
        )
        manifest_path = run_directory / "run_manifest.json"
        failures = []
        with TrainingLock(run_directory):
            for index, method in enumerate(methods, start=1):
                state = manifest["method_states"][method.value]
                if (
                    state.get("status") == "completed"
                    and int(state.get("completed_episode_count", 0)) >= episodes
                ):
                    print(
                        "[{0}/{1}] 跳过已达到目标的 {2}".format(
                            index,
                            len(methods),
                            display_name(method),
                        )
                    )
                    continue
                print(
                    "[{0}/{1}] 开始 {2}".format(
                        index,
                        len(methods),
                        display_name(method),
                    )
                )
                method_directory = run_directory / method.value
                method_directory.mkdir(parents=True, exist_ok=True)
                last_checkpoint = method_directory / "last_checkpoint.pt"
                resume_checkpoint = (
                    last_checkpoint
                    if resume_run and last_checkpoint.exists()
                    else None
                )
                state.update(
                    {
                        "status": "running",
                        "started_at": utc_now(),
                        "target_episode_count": episodes,
                    }
                )
                atomic_write_json(manifest_path, manifest)
                started = time.perf_counter()
                try:
                    result = self.method_executor(
                        method=method,
                        episodes=episodes,
                        output_directory=method_directory,
                        reward_spec=reward_spec,
                        resume_checkpoint=resume_checkpoint,
                        skip_validation=skip_validation,
                        max_steps_per_episode=max_steps_per_episode,
                        validation_max_steps=validation_max_steps,
                    )
                    duration = time.perf_counter() - started
                    state.update(
                        {
                            "status": "completed",
                            "completed_at": utc_now(),
                            "completed_episode_count": episodes,
                            "total_environment_steps": result[
                                "total_environment_steps"
                            ],
                            "best_checkpoint": self._existing_path_or_none(
                                method_directory / "best_checkpoint.pt"
                            ),
                            "last_checkpoint": self._existing_path_or_none(
                                last_checkpoint
                            ),
                            "training_duration_seconds": duration,
                            "reward_method": result["reward_method"],
                            "reward_spec_id": result.get("reward_spec_id"),
                        }
                    )
                    method_summary = result.get("summary", {})
                    validation = (
                        method_summary.get("best_validation_result") or {}
                    )
                    print(
                        "完成 {0}：状态=completed，Episode={1}，"
                        "环境步={2}，耗时={3:.2f}s".format(
                            display_name(method),
                            episodes,
                            result["total_environment_steps"],
                            duration,
                        )
                    )
                    print(
                        "Best Checkpoint：{0}；timeliness={1}；"
                        "load_balance={2}".format(
                            state["best_checkpoint"],
                            validation.get("timeliness_raw_mean"),
                            validation.get(
                                "load_balance_mean_per_task_mean"
                            ),
                        )
                    )
                except Exception as error:
                    state.update(
                        {
                            "status": "failed",
                            "failed_at": utc_now(),
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    failures.append(method.value)
                    manifest["status"] = "failed"
                    atomic_write_json(manifest_path, manifest)
                    self._release_training_objects()
                    if not continue_on_error:
                        raise
                    continue
                finally:
                    self._release_training_objects()
                atomic_write_json(manifest_path, manifest)

            manifest["status"] = (
                "completed_with_errors" if failures else "completed"
            )
            manifest["completed_at"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            comparison = build_comparison(
                run_directory,
                methods,
                manifest["method_states"],
                self.baseline_config["best_model_rule"]["metrics"],
                "checkpoint_selection",
            )
            if self.baseline_config["artifact_management"]["compact_completed_runs"]:
                for method in methods:
                    state = manifest["method_states"][method.value]
                    if state.get("status") == "completed":
                        BaselineTrainingRunner.compact_completed_artifacts(
                            run_directory / method.value
                        )
            summary = {
                "schema_version": "1.0",
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "methods": [method.value for method in methods],
                "episodes_per_method": episodes,
                "failed_methods": failures,
                "comparison_path": str(run_directory / "comparison.json"),
            }
            atomic_write_json(run_directory / "run_summary.json", summary)
        return run_directory, summary, comparison

    def _execute_method(
        self,
        method,
        episodes,
        output_directory,
        reward_spec,
        resume_checkpoint,
        skip_validation,
        max_steps_per_episode,
        validation_max_steps,
    ):
        """重新创建全部组件，执行一种方法并返回语义化摘要。"""
        spec = reward_spec if method == BaselineMethod.LLM_PPO else None
        config, encoder, actor, critic, trainer, evaluator = (
            build_baseline_components(
                method,
                self.baseline_config,
                self.mappo_config,
                spec,
            )
        )
        runner = make_runner(
            method,
            trainer,
            evaluator,
            config,
            encoder,
            self.baseline_config,
            output_directory,
        )
        checkpoint_state = None
        if resume_checkpoint is not None:
            checkpoint_state = restore_baseline_checkpoint(
                resume_checkpoint,
                actor,
                critic,
                trainer,
                encoder,
                method,
            )
        summary = runner.run(
            target_episode_count=episodes,
            start_episode_index=(
                int(checkpoint_state["next_episode_index"])
                if checkpoint_state
                else 0
            ),
            update_index=(
                int(checkpoint_state["update_index"])
                if checkpoint_state
                else 0
            ),
            best_validation_result=(
                checkpoint_state.get("best_validation_result")
                if checkpoint_state
                else None
            ),
            best_episode_index=(
                checkpoint_state.get("best_episode_index")
                if checkpoint_state
                else None
            ),
            best_validation_details=(
                checkpoint_state.get("best_validation_details")
                if checkpoint_state
                else None
            ),
            skip_validation=skip_validation,
            max_steps_per_episode=max_steps_per_episode,
            validation_max_steps=validation_max_steps,
            resume=checkpoint_state is not None,
        )
        total_steps = trainer.environment_steps
        return {
            "summary": summary,
            "total_environment_steps": total_steps,
            "reward_method": trainer.reward_model.log_metadata.reward_method,
            "reward_spec_id": getattr(trainer.reward_model, "reward_spec_id", None),
        }

    def _resolve_reward_spec(self, methods, reward_spec_path):
        if BaselineMethod.LLM_PPO not in methods:
            return None, None
        configured = self.baseline_config["methods"]["llm_ppo"].get(
            "selected_reward_spec_path"
        )
        path = Path(reward_spec_path or configured or "")
        if not path.is_file():
            raise FileNotFoundError(
                "请先注册已有奖励规范，或显式使用--prepare-llm-reward生成。"
            )
        limits = self.baseline_config["methods"]["llm_ppo"]["weight_limits"]
        spec = LlmRewardSpec.load(path, limits["minimum"], limits["maximum"])
        print("检测到冻结奖励规范，本次LLM-PPO训练不会调用DeepSeek API。")
        return spec, str(path.resolve())

    def _new_manifest(
        self,
        run_id,
        methods,
        episodes,
        reward_spec_path,
        reward_spec,
    ):
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "running",
            "methods": [method.value for method in methods],
            "episodes_per_method": episodes,
            "seed": self.baseline_config["seed"],
            "device": self.baseline_config["device"],
            "config_path": str(self.baseline_config_path),
            "mappo_config_path": str(self.mappo_config_path),
            "config_sha256": sha256_file(self.baseline_config_path),
            "mappo_config_sha256": sha256_file(self.mappo_config_path),
            "reward_spec_path": reward_spec_path,
            "reward_spec_id": reward_spec.spec_id if reward_spec else None,
            "started_at": utc_now(),
            "method_states": {
                method.value: {"status": "pending"} for method in methods
            },
        }

    def _load_resume_manifest(self, run_directory, methods, reward_spec):
        manifest_path = run_directory / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Resume Run缺少run_manifest.json")
        manifest = load_json(manifest_path)
        if methods and [item.value for item in methods] != manifest["methods"]:
            raise ValueError("Resume方法顺序与原Run不一致")
        if manifest["config_sha256"] != sha256_file(self.baseline_config_path):
            raise ValueError("基线配置已变化，拒绝Resume")
        if manifest["mappo_config_sha256"] != sha256_file(self.mappo_config_path):
            raise ValueError("MAPPO配置已变化，拒绝Resume")
        if manifest["seed"] != self.baseline_config["seed"]:
            raise ValueError("共同seed已变化，拒绝Resume")
        if manifest["device"] != self.baseline_config["device"]:
            raise ValueError("设备配置已变化，拒绝Resume")
        expected_spec = reward_spec.spec_id if reward_spec else None
        if manifest.get("reward_spec_id") != expected_spec:
            raise ValueError("奖励规范已变化，拒绝Resume")
        return manifest

    def _print_plan(self, run_directory, methods, episodes, reward_spec):
        print("Run ID：{0}".format(run_directory.name))
        print("方法顺序：{0}".format(" -> ".join(item.value for item in methods)))
        print("每种方法目标Episode数：{0}".format(episodes))
        print("共同seed：{0}".format(self.baseline_config["seed"]))
        print("设备：{0}".format(self.baseline_config["device"]))
        print(
            "validation：{0}".format(
                self.baseline_config["training"]["validation"]
            )
        )
        print("奖励Spec ID：{0}".format(reward_spec.spec_id if reward_spec else None))
        print("结果目录：{0}".format(run_directory))
        print("是否会调用API：否")

    @staticmethod
    def _release_training_objects():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _existing_path_or_none(path):
        """仅在产物真实存在时把路径写入Manifest。"""
        candidate = Path(path)
        return str(candidate) if candidate.is_file() else None
