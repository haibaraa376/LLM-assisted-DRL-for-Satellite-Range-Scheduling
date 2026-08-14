"""提供LLM奖励候选诊断、缓存、预算和公平基线搜索。"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from mappo.model_selection import rank_validation_results
from mappo.manual_reward import RewardFeatures, combine_manual_reward
from mappo.trainer import parameter_vector

from .baseline_runner import (
    build_baseline_components,
    make_runner,
    restore_baseline_checkpoint,
)
from .llm_prompt import system_prompt
from .llm_provider import FatalProviderError, RetryableProviderError
from .llm_reward import reward_spec_weights
from .llm_schema import LlmRewardSpec
from .methods import BaselineMethod


# HERON 排名指标之外，资格检查仍需要该字段，名称保持旧接口完全一致。
_ELIGIBILITY_VALIDATION_METRICS = ("accepted_sgl_count",)


def build_staged_plan(candidate_count, staged_config):
    """按候选数缩放固定阶段保留数量，返回各阶段目标Episode和候选数。"""
    if int(candidate_count) <= 0:
        raise ValueError("候选数量必须为正整数")
    initial = int(staged_config["initial_episodes"])
    keep_1 = min(
        int(staged_config["keep_after_stage_1"]),
        max(1, int(candidate_count) // 2),
    )
    keep_2 = min(
        int(staged_config["keep_after_stage_2"]),
        max(1, keep_1 // 2),
    )
    keep_3 = min(int(staged_config["keep_after_stage_3"]), keep_2)
    target_2 = initial + int(staged_config["extra_episodes_stage_2"])
    target_3 = target_2 + int(staged_config["extra_episodes_stage_3"])
    target_4 = target_3 + int(staged_config["confirmation_episodes"])
    if target_4 != int(staged_config["max_episodes_per_candidate"]):
        raise ValueError("分阶段计划的最终Episode数与配置不一致")
    return (
        {"stage": 1, "candidates": int(candidate_count), "episodes": initial},
        {"stage": 2, "candidates": keep_1, "episodes": target_2},
        {"stage": 3, "candidates": keep_2, "episodes": target_3},
        {"stage": 4, "candidates": keep_3, "episodes": target_4},
    )


@dataclass
class GenerationBudget:
    """跟踪真实调用次数和输入/输出Token，不计缓存命中。"""

    maximum_api_calls: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def ensure_request_allowed(self, estimated_input_tokens):
        if self.api_calls >= self.maximum_api_calls:
            raise RuntimeError("已达到LLM API调用预算")
        if self.input_tokens + estimated_input_tokens > self.maximum_input_tokens:
            raise RuntimeError("已达到LLM输入Token预算")
        if self.output_tokens >= self.maximum_output_tokens:
            raise RuntimeError("已达到LLM输出Token预算")

    def record_call(self):
        """在每次真实Provider尝试前计数，失败重试也消耗调用预算。"""
        self.api_calls += 1

    def record_tokens(self, input_tokens, output_tokens):
        """成功响应后累计平台返回的Token用量。"""
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        if self.input_tokens > self.maximum_input_tokens:
            raise RuntimeError("LLM实际输入Token超过预算")
        if self.output_tokens > self.maximum_output_tokens:
            raise RuntimeError("LLM实际输出Token超过预算")


def candidate_cache_key(provider_name, model, thinking_config, prompt):
    """按Provider、模型、思考配置、Prompt和Schema版本计算缓存键。"""
    payload = json.dumps(
        {
            "provider": provider_name,
            "model": model,
            "thinking": thinking_config,
            "prompt": prompt,
            "schema_version": "1.0",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedRewardGenerator:
    """统一执行缓存、有限重试、Schema校验和非敏感审计。"""

    def __init__(self, provider, provider_config, llm_config, budget, sleep=time.sleep):
        self.provider = provider
        self.provider_config = provider_config
        self.llm_config = llm_config
        self.budget = budget
        self.sleep = sleep

    def generate(self, prompt, metadata=None):
        """生成一个合法Spec；缓存命中不调用Provider也不增加预算。"""
        self._reject_sensitive_prompt(prompt)
        thinking = {
            "enabled": self.provider_config["thinking_enabled"],
            "reasoning_effort": self.provider_config["reasoning_effort"],
        }
        key = candidate_cache_key(
            self.provider.name,
            getattr(
                self.provider,
                "model",
                self.provider_config.get("model", "unknown"),
            ),
            thinking,
            prompt,
        )
        call_dir = Path(self.llm_config["cache"]["directory"]) / key
        cached_spec = call_dir / "reward_spec.json"
        if self.llm_config["cache"]["enabled"] and cached_spec.exists():
            spec = LlmRewardSpec.load(
                cached_spec,
                self.llm_config["weight_limits"]["minimum"],
                self.llm_config["weight_limits"]["maximum"],
            )
            return spec, {
                "call_id": key,
                "cache_hit": True,
                "retry_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        estimated = max(1, len(prompt) // 4)
        max_retries = int(self.provider_config["max_retries"])
        delays = list(self.provider_config["retry_delays_seconds"])
        last_error = None
        for attempt in range(max_retries + 1):
            self.budget.ensure_request_allowed(estimated)
            try:
                self.budget.record_call()
                result = self.provider.generate_reward_spec(
                    prompt,
                    {**(metadata or {}), "system_prompt": system_prompt()},
                )
                self.budget.record_tokens(
                    result.input_tokens,
                    result.output_tokens,
                )
                spec = LlmRewardSpec.from_json(
                    result.content,
                    self.llm_config["weight_limits"]["minimum"],
                    self.llm_config["weight_limits"]["maximum"],
                )
                self._write_audit(
                    call_dir,
                    key,
                    prompt,
                    result,
                    spec,
                    attempt,
                    success=True,
                )
                return spec, {
                    "call_id": key,
                    "cache_hit": False,
                    "retry_count": attempt,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
            except FatalProviderError:
                raise
            except (RetryableProviderError, ValueError) as error:
                last_error = error
                if attempt >= max_retries:
                    break
                self.sleep(delays[min(attempt, len(delays) - 1)] if delays else 0)
        raise RuntimeError("LLM候选生成在有限重试后失败") from last_error

    @staticmethod
    def _reject_sensitive_prompt(prompt):
        key = os.getenv("DEEPSEEK_API_KEY")
        if key and key in prompt:
            raise ValueError("Prompt不得包含API Key")
        if "DEEPSEEK_API_KEY=" in prompt:
            raise ValueError("Prompt不得包含密钥赋值")

    def _write_audit(
        self,
        call_dir,
        key,
        prompt,
        result,
        spec,
        retry_count,
        success,
    ):
        """仅保存最终JSON和非敏感元数据，不读取推理过程。"""
        call_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "call_id": key,
            "provider": self.provider.name,
            "model": result.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "cache_hit": False,
            "retry_count": retry_count,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
            "response_id": result.response_id,
            "success": success,
            "error_type": None,
            "api_key_present": bool(os.getenv("DEEPSEEK_API_KEY")),
        }
        (call_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (call_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (call_dir / "response.json").write_text(
            json.dumps(json.loads(result.content), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        spec.save(call_dir / "reward_spec.json")


def diagnose_reward_spec(spec, numerical, manual_weights=None):
    """在训练前检查固定奖励方向、有限性和循环中继约束。"""
    # 诊断必须使用训练时实际生效的归一化权重，而不是LLM原始尺度。
    manual_weights = manual_weights or {
        "sgl_progress": 1.0,
        "relay_progress": 0.15,
        "completion": 0.5,
        "balance": 0.05,
        "expiration": 0.5,
        "invalid_action": 0.1,
        "coordination_conflict": 0.03,
        "relay_cost": 0.02,
    }
    weights = reward_spec_weights(spec, manual_weights)
    zero = RewardFeatures(*(0.0 for _ in range(8)))
    idle = combine_manual_reward(zero, weights, numerical).total_reward
    sgl = combine_manual_reward(
        RewardFeatures(0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        weights,
        numerical,
    ).total_reward
    relay = combine_manual_reward(
        RewardFeatures(0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10),
        weights,
        numerical,
    ).total_reward
    completion = combine_manual_reward(
        RewardFeatures(0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
        weights,
        numerical,
    ).total_reward
    expiration = combine_manual_reward(
        RewardFeatures(0.0, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0),
        weights,
        numerical,
    ).total_reward
    invalid = combine_manual_reward(
        RewardFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0),
        weights,
        numerical,
    ).total_reward
    conflict = combine_manual_reward(
        RewardFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0),
        weights,
        numerical,
    ).total_reward
    checks = {
        "sgl_above_idle": sgl > idle,
        "completion_above_relay": completion > relay,
        "expiration_negative": expiration < 0.0,
        "invalid_negative": invalid < 0.0,
        "conflict_negative": conflict < 0.0,
        "idle_balance_zero": idle == 0.0,
        "relay_loop_below_sgl": relay < sgl,
        "all_finite": all(
            math.isfinite(value)
            for value in (idle, sgl, relay, completion, expiration, invalid, conflict)
        ),
    }
    return {
        "status": (
            "accepted_for_training"
            if all(checks.values())
            else "rejected_before_training"
        ),
        "checks": checks,
        "samples": {
            "idle": idle,
            "sgl": sgl,
            "relay": relay,
            "completion": completion,
            "expiration": expiration,
            "invalid": invalid,
            "conflict": conflict,
        },
    }


def aggregate_tail_validation(
    learning_curve_path,
    hierarchy,
    tail_episodes,
    candidate_id="unknown_candidate",
    include_shortened=False,
):
    """聚合最后若干完整 Episode 的 validation 均值。

    曲线由训练器在每个 Episode 后直接写出，因此这里不重新运行环境；每项标准差
    是 Episode 级验证均值的样本标准差，供 HERON 使用均值标准误比较。
    """
    tail_episodes = int(tail_episodes)
    if tail_episodes <= 0:
        raise ValueError("候选尾部Episode数必须为正整数")
    points = json.loads(Path(learning_curve_path).read_text(encoding="utf-8"))
    complete_points = [
        point
        for point in points
        if (point.get("full_episode") or include_shortened)
        and point.get("data_conservation_passed")
        and point.get("validation") is not None
    ]
    if len(complete_points) < tail_episodes:
        return {
            "eligible": False,
            "reason": "insufficient_complete_tail_episodes",
            "available_complete_episodes": len(complete_points),
            "required_tail_episodes": tail_episodes,
        }
    window = complete_points[-tail_episodes:]
    aggregate = {}
    required_metrics = [entry["metric"] for entry in hierarchy]
    required_metrics.extend(
        metric
        for metric in _ELIGIBILITY_VALIDATION_METRICS
        if metric not in required_metrics
    )
    for metric in required_metrics:
        values = []
        for point in window:
            field_name = metric + "_mean"
            value = point.get(field_name)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(
                    "候选{0}的尾部Episode缺少有限validation字段：{1}".format(
                        candidate_id,
                        field_name,
                    )
                )
            values.append(float(value))
        mean = sum(values) / len(values)
        variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        aggregate[metric + "_mean"] = mean
        aggregate[metric + "_std"] = math.sqrt(variance)
        aggregate[metric + "_sample_count"] = len(values)
    diagnostics = _aggregate_tail_diagnostics(window, candidate_id)
    return {
        "eligible": True,
        "selection_window_start_episode": int(window[0]["episode_index"]) + 1,
        "selection_window_end_episode": int(window[-1]["episode_index"]) + 1,
        "selection_window_size": len(window),
        "aggregated_validation": aggregate,
        "aggregated_diagnostics": diagnostics,
        "tail_episode_records": window,
    }


def _aggregate_tail_diagnostics(window, candidate_id):
    """按尾部Episode平均实际奖励贡献；旧曲线没有诊断字段时保持可读。"""
    component_records = [point.get("reward_component_abs_sums") for point in window]
    if not any(component_records):
        return None
    if not all(isinstance(values, dict) for values in component_records):
        raise ValueError("候选{0}的尾部Episode缺少奖励贡献诊断".format(candidate_id))
    base_names = (
        "weighted_sgl_progress", "weighted_relay_progress", "weighted_completion",
        "weighted_balance", "weighted_expiration", "weighted_invalid_action",
        "weighted_coordination_conflict", "weighted_relay_cost",
    )
    llm_names = tuple("llm_" + name for name in (
        "sgl_progress", "relay_progress", "completion", "balance", "expiration",
        "invalid_action", "coordination_conflict", "relay_cost",
    ))
    required = base_names + llm_names
    totals = {name: 0.0 for name in required}
    for episode_index, values in enumerate(component_records, start=1):
        for name in required:
            if name not in values:
                raise ValueError(
                    "候选{0}的尾部Episode {1}缺少奖励贡献字段：{2}".format(
                        candidate_id, episode_index, name
                    )
                )
            value = float(values[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "候选{0}的尾部Episode {1}奖励贡献字段无效：{2}".format(
                        candidate_id, episode_index, name
                    )
                )
            totals[name] += value
    averages = {name: value / len(window) for name, value in totals.items()}
    base_abs_sum = sum(averages[name] for name in base_names)
    llm_abs_sum = sum(averages[name] for name in llm_names)
    total_abs_sum = base_abs_sum + llm_abs_sum
    shares = {
        name: value / total_abs_sum if total_abs_sum > 0.0 else 0.0
        for name, value in averages.items()
    }
    dominant = max(shares, key=shares.get) if shares else None
    return {
        "base_abs_sum": base_abs_sum,
        "llm_abs_sum": llm_abs_sum,
        "total_abs_sum": total_abs_sum,
        "llm_contribution_ratio": (
            llm_abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
        ),
        "component_shares": shares,
        "dominant_component": dominant,
        "max_component_share": shares[dominant] if dominant else 0.0,
    }


def assess_candidate_eligibility(
    training_summary,
    validation,
    config,
    selection_window=None,
    smoke=False,
):
    """仅以完整性、守恒、有限性和最低SGL作为候选硬资格条件。"""
    selection_window = selection_window or {"eligible": True}
    reasons, warnings = [], []
    if not selection_window.get("eligible"):
        reasons.append(selection_window.get("reason", "invalid_tail_window"))
    if int(training_summary.get("episodes_run", 0)) < 2:
        reasons.append("insufficient_complete_episodes")
    tail_records = selection_window.get("tail_episode_records", [])
    if not tail_records or not all(record.get("full_episode", False) for record in tail_records):
        reasons.append("incomplete_episode")
    if not tail_records or not all(record.get("data_conservation_passed", False) for record in tail_records):
        reasons.append("data_conservation_failed")
    if not bool(training_summary.get("best_validation_data_conservation", False)):
        reasons.append("validation_data_conservation_failed")
    try:
        validation_is_finite = all(
            math.isfinite(float(value)) for value in validation.values()
        )
    except (TypeError, ValueError):
        validation_is_finite = False
    if not validation_is_finite:
        reasons.append("non_finite_validation_metric")
    try:
        accepted_sgl = float(validation["accepted_sgl_count_mean"])
    except (KeyError, TypeError, ValueError):
        reasons.append("missing_accepted_sgl_count_mean")
    else:
        if accepted_sgl < float(config["minimum_accepted_sgl_mean"]):
            reasons.append("accepted_sgl_below_minimum")
    diagnostics = selection_window.get("aggregated_diagnostics")
    if diagnostics:
        if float(diagnostics["llm_contribution_ratio"]) > float(config["maximum_llm_contribution_ratio"]):
            warnings.append("llm_contribution_ratio_above_threshold")
        if float(diagnostics["max_component_share"]) > float(config["maximum_single_component_dominance"]):
            warnings.append("max_component_share_above_threshold")
    checks = {"hard_requirements": not reasons}
    passed = not reasons
    return {
        "passed": passed,
        "reasons": reasons,
        "warnings": warnings,
        # smoke仅验证阶段控制流，缩短Episode不冒充正式通过。
        "eligible": passed or (smoke and selection_window.get("eligible")),
        "checks": checks,
        "formal_eligible": passed,
        "smoke_selection_override": smoke and not passed,
    }


def train_and_rank_candidates(
    candidates,
    baseline_config,
    mappo_config,
    output_root,
    smoke=False,
):
    """以固定的四阶段追加训练和HERON排序训练独立候选。"""
    results = []
    initial_actor = None
    initial_critic = None
    search = baseline_config["methods"]["llm_ppo"]["search"]
    runtimes = []
    staged = baseline_config["staged_search"]
    max_episodes = int(staged["max_episodes_per_candidate"])
    stage_plan = build_staged_plan(len(candidates), staged)
    for candidate_id, spec in candidates:
        diagnosis = diagnose_reward_spec(
            spec,
            mappo_config["manual_reward"]["numerical"],
            mappo_config["manual_reward"]["weights"],
        )
        candidate_dir = Path(output_root) / "cands" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        spec.save(candidate_dir / "spec.json")
        if diagnosis["status"] != "accepted_for_training":
            results.append(
                {
                    "candidate_id": candidate_id,
                    "reward_spec_id": spec.spec_id,
                    "diagnosis": diagnosis,
                    "status": "rejected_before_training",
                }
            )
            continue
        config, encoder, actor, critic, trainer, evaluator = (
            build_baseline_components(
                BaselineMethod.LLM_PPO,
                baseline_config,
                mappo_config,
                spec,
            )
        )
        # 冻结可追溯身份；每一个阶段Checkpoint都会携带这些元数据。
        config["candidate_id"] = candidate_id
        actor_vector = parameter_vector(actor)
        critic_vector = parameter_vector(critic)
        if initial_actor is None:
            initial_actor = actor_vector
            initial_critic = critic_vector
        elif not actor_vector.equal(initial_actor) or not critic_vector.equal(
            initial_critic
        ):
            raise RuntimeError("LLM候选没有从相同网络初始化开始")
        runner = make_runner(
            BaselineMethod.LLM_PPO,
            trainer,
            evaluator,
            config,
            encoder,
            baseline_config,
            candidate_dir,
        )
        runner.training["episode_count"] = int(
            search["candidate_training_episodes"]
        )
        runner.training["task_count"] = int(search["candidate_task_count"])
        runner.training["training_seed"] = int(
            search["common_training_seeds"][0]
        )
        runner.training["validation"]["protocol"] = search[
            "evaluation_protocol"
        ]
        # 每个候选固定使用自己的last.pt，阶段追加前均从该文件恢复。
        runner.training["checkpoint"]["last_path"] = str(candidate_dir / "last.pt")
        # 搜索候选只需最新恢复点；逐Episode阶段Checkpoint由正式训练流程保留。
        runner.training["save_episode_checkpoints"] = False
        runtimes.append({"candidate_id": candidate_id, "spec": spec, "runner": runner, "trainer": trainer, "config": config, "encoder": encoder, "episodes": 0, "updates": 0, "summary": None, "stage_reached": 0, "eliminated_after_stage": None, "final_rank_in_stage": None})

    def train_one(runtime):
        """从同一候选的 last 状态续训一个完整Episode，不共享其他候选参数。"""
        start = runtime["episodes"]
        previous = runtime["summary"] or {}
        if start > 0:
            state = restore_baseline_checkpoint(
                runtime["runner"].training["checkpoint"]["last_path"],
                runtime["trainer"].actor,
                runtime["trainer"].critic,
                runtime["trainer"],
                runtime["encoder"],
                BaselineMethod.LLM_PPO,
                protocol_name=search["evaluation_protocol"],
            )
            if int(state["next_episode_index"]) != start:
                raise RuntimeError("候选{0}的last.pt续训Episode索引不一致".format(runtime["candidate_id"]))
        summary = runtime["runner"].run(
            target_episode_count=start + 1,
            start_episode_index=start,
            update_index=runtime["updates"],
            best_validation_result=previous.get("best_validation_result"),
            best_episode_index=previous.get("best_episode_index"),
            best_validation_details=previous.get("best_validation_details"),
            # smoke仅验证控制流；缩短轨迹绝不标记为正式实验或正式候选资格。
            max_steps_per_episode=16 if smoke else None,
            validation_max_steps=16 if smoke else None,
            resume=start > 0,
        )
        runtime["episodes"] += 1
        runtime["updates"] = summary["total_update_index"]
        summary["episodes_run"] = runtime["episodes"]
        summary["target_episode_count"] = max_episodes
        summary["candidate_id"] = runtime["candidate_id"]
        runtime["summary"] = summary

    def train_to(runtime, target_episodes):
        """逐Episode续训到目标；每次均经由候选自己的last.pt恢复。"""
        while runtime["episodes"] < target_episodes:
            train_one(runtime)

    def rank_stage(active, stage, keep_count):
        """用统一HERON比较器排序，并记录淘汰与精简阶段审计。"""
        candidates_for_rank = []
        stage_records = []
        for item in active:
            window = aggregate_tail_validation(
                item["summary"]["learning_curve_paths"]["json"],
                baseline_config["best_model_rule"]["hierarchy"],
                min(3, item["episodes"]), item["candidate_id"],
                include_shortened=smoke,
            )
            eligibility = assess_candidate_eligibility(
                item["summary"], window["aggregated_validation"],
                baseline_config["candidate_eligibility"], selection_window=window,
                smoke=smoke,
            )
            item["selection_window"] = window
            item["eligibility"] = eligibility
            item["stage_reached"] = stage
            if eligibility["eligible"]:
                candidates_for_rank.append({"candidate_id": item["candidate_id"], "reward_spec_id": item["spec"].spec_id, "validation": window["aggregated_validation"], "runtime": item})
        ranked, traces = rank_validation_results(
            candidates_for_rank, rule=baseline_config["best_model_rule"], return_traces=True,
        )
        kept = ranked[: min(keep_count, len(ranked))]
        kept_ids = {record["candidate_id"] for record in kept}
        ranks = {record["candidate_id"]: index for index, record in enumerate(ranked, start=1)}
        for item in active:
            item["final_rank_in_stage"] = ranks.get(item["candidate_id"])
            if item["candidate_id"] not in kept_ids:
                item["eliminated_after_stage"] = stage
            stage_records.append({"stage": stage, "candidate": item["candidate_id"], "episodes_trained": item["episodes"], "rank": item["final_rank_in_stage"], "kept": item["candidate_id"] in kept_ids})
        return [record["runtime"] for record in kept], traces, stage_records

    # 固定预算：阶段1的所有候选训练2轮，随后按4、2、1逐阶段追加。
    for runtime in runtimes:
        train_to(runtime, int(staged["initial_episodes"]))
    keep_1 = stage_plan[1]["candidates"]
    survivors_1, traces_1, stage_log = rank_stage(runtimes, 1, keep_1)
    if not survivors_1:
        raise RuntimeError("第一阶段没有满足硬资格条件的候选")

    for runtime in survivors_1:
        train_to(runtime, stage_plan[1]["episodes"])
    keep_2 = min(stage_plan[2]["candidates"], len(survivors_1))
    survivors_2, traces_2, records_2 = rank_stage(survivors_1, 2, keep_2)
    stage_log.extend(records_2)
    if not survivors_2:
        raise RuntimeError("第二阶段没有满足硬资格条件的候选")

    target_3 = stage_plan[2]["episodes"]
    for runtime in survivors_2:
        train_to(runtime, target_3)
    keep_3 = min(stage_plan[3]["candidates"], len(survivors_2))
    survivors_3, traces_3, records_3 = rank_stage(survivors_2, 3, keep_3)
    stage_log.extend(records_3)
    if not survivors_3:
        raise RuntimeError("第三阶段没有满足硬资格条件的候选")

    for runtime in survivors_3:
        train_to(runtime, max_episodes)
    finalists, traces_4, records_4 = rank_stage(survivors_3, 4, 1)
    stage_log.extend(records_4)
    if not finalists:
        raise RuntimeError("确认阶段没有满足硬资格条件的候选")

    for runtime in runtimes:
        summary = runtime["summary"]
        selection_window = aggregate_tail_validation(summary["learning_curve_paths"]["json"], baseline_config["best_model_rule"]["hierarchy"], min(3, runtime["episodes"]), runtime["candidate_id"], include_shortened=smoke)
        results.append({"candidate_id": runtime["candidate_id"], "reward_spec_id": runtime["spec"].spec_id, "diagnosis": diagnose_reward_spec(runtime["spec"], mappo_config["manual_reward"]["numerical"], mappo_config["manual_reward"]["weights"]), "status": "trained", "official_experiment": not smoke, "training_summary": summary, "validation": selection_window["aggregated_validation"], "selection_window": selection_window, "spec": runtime["spec"], "weight_metadata": runtime["trainer"].reward_model.weight_metadata, "stage_reached": runtime["stage_reached"], "episodes_trained": runtime["episodes"], "eliminated_after_stage": runtime["eliminated_after_stage"], "final_rank_in_stage": runtime["final_rank_in_stage"], "stage_records": stage_log})
    trained = [item for item in results if item["status"] == "trained"]
    if not trained:
        raise RuntimeError("没有通过诊断并完成训练的LLM奖励候选")
    eligibility = baseline_config["candidate_eligibility"]
    for item in trained:
        validation = item["validation"]
        item["eligibility"] = assess_candidate_eligibility(
            item["training_summary"],
            validation,
            eligibility,
            selection_window=item["selection_window"],
            smoke=smoke,
        )
        item["rank"] = None
    eligible = [item for item in trained if item["eligibility"]["eligible"]]
    ranked, selection_traces = rank_validation_results(eligible, metrics_getter=lambda item: item["validation"], rule=baseline_config["best_model_rule"], return_traces=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    for item in results:
        item["selection_traces"] = selection_traces if item in eligible else []
    return results, ranked[0] if ranked else None
