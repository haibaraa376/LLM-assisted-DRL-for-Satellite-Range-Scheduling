"""安全生成并公平训练固定预算的直接LLM奖励候选。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cmp_to_key
import hashlib
import json
import math
import os
from pathlib import Path
import time

from mappo.manual_reward import RewardFeatures, combine_manual_reward
from mappo.trainer import parameter_vector

from .baseline_runner import build_baseline_components, make_runner
from .llm_prompt import system_prompt
from .llm_provider import FatalProviderError, RetryableProviderError
from .llm_reward import reward_spec_weights
from .llm_schema import LlmRewardSpec
from .methods import BaselineMethod


RANKING_METRICS = (
    ("completion_rate", "max"),
    ("delivered_data_mbit", "max"),
    ("load_balance_mean_per_task", "max"),
)
_COMPONENTS = (
    "weighted_sgl_progress", "weighted_relay_progress", "weighted_completion",
    "weighted_balance", "weighted_expiration", "weighted_invalid_action",
    "weighted_coordination_conflict", "weighted_relay_cost",
)


@dataclass
class GenerationBudget:
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
        self.api_calls += 1

    def record_tokens(self, input_tokens, output_tokens):
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        if self.input_tokens > self.maximum_input_tokens or self.output_tokens > self.maximum_output_tokens:
            raise RuntimeError("LLM实际Token超过预算")


def candidate_cache_key(provider_name, model, thinking_config, prompt, cache_identity=None):
    """缓存身份包含任务规模、直接奖励模式、Schema与冲突固定规则。"""
    payload = {
        "provider": provider_name,
        "model": model,
        "thinking": thinking_config,
        "prompt": prompt,
        "cache_identity": cache_identity or {},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class CachedRewardGenerator:
    """保留缓存、有限重试、预算和严格JSON Schema校验。"""

    def __init__(self, provider, provider_config, llm_config, budget, sleep=time.sleep):
        self.provider, self.provider_config, self.llm_config = provider, provider_config, llm_config
        self.budget, self.sleep = budget, sleep

    def generate(self, prompt, metadata=None):
        metadata = metadata or {}
        key = candidate_cache_key(
            self.provider.name,
            getattr(self.provider, "model", self.provider_config.get("model", "unknown")),
            {"enabled": self.provider_config["thinking_enabled"], "reasoning_effort": self.provider_config["reasoning_effort"]},
            prompt,
            {
                "task_count": int(metadata.get("task_count", self.llm_config.get("task_count", 150))),
                "reward_mode": self.llm_config["reward_mode"],
                "reward_schema_version": self.llm_config["reward_schema_version"],
                "conflict_fixed_zero": self.llm_config["conflict_fixed_zero"],
                "l1_target_scale": self.llm_config["l1_target_scale"],
                "round_index": metadata.get("round_index"),
                "candidate_index": metadata.get("candidate_index"),
                "candidate_id": metadata.get("candidate_id"),
            },
        )
        root = Path(self.llm_config["cache"]["directory"]) / key
        cached = root / "reward_spec.json"
        limits = self.llm_config["weight_limits"]
        if self.llm_config["cache"]["enabled"] and cached.is_file():
            return LlmRewardSpec.load(cached, limits["minimum"], limits["maximum"]), {"call_id": key, "cache_hit": True, "retry_count": 0, "input_tokens": 0, "output_tokens": 0}
        if os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_API_KEY") in prompt:
            raise ValueError("Prompt不得包含API Key")
        estimated = max(1, len(prompt) // 4)
        delays = self.provider_config["retry_delays_seconds"]
        last_error = None
        for attempt in range(int(self.provider_config["max_retries"]) + 1):
            self.budget.ensure_request_allowed(estimated)
            try:
                self.budget.record_call()
                result = self.provider.generate_reward_spec(prompt, {**(metadata or {}), "system_prompt": system_prompt()})
                self.budget.record_tokens(result.input_tokens, result.output_tokens)
                spec = LlmRewardSpec.from_json(result.content, limits["minimum"], limits["maximum"])
                root.mkdir(parents=True, exist_ok=True)
                spec.save(cached)
                (root / "metadata.json").write_text(json.dumps({"call_id": key, "model": result.model, "timestamp": datetime.now(timezone.utc).isoformat(), "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "input_tokens": result.input_tokens, "output_tokens": result.output_tokens, "response_id": result.response_id}, ensure_ascii=False, indent=2), encoding="utf-8")
                return spec, {"call_id": key, "cache_hit": False, "retry_count": attempt, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
            except FatalProviderError:
                raise
            except (RetryableProviderError, ValueError) as error:
                last_error = error
                if attempt < int(self.provider_config["max_retries"]):
                    self.sleep(delays[min(attempt, len(delays) - 1)] if delays else 0)
        raise RuntimeError("LLM候选生成在有限重试后失败") from last_error


def diagnose_reward_spec(spec, numerical, target_l1, manual_weights=None):
    """仅检查实际直接LLM公式的合法性、有限性和数值边界。"""
    del manual_weights
    weights = reward_spec_weights(spec, target_l1=target_l1)
    checks = {
        "conflict_fixed_zero": weights["coordination_conflict"] == 0.0,
        "weights_finite": all(math.isfinite(value) and value >= 0.0 for value in weights.values()),
        "seven_weights_nonzero": sum(abs(value) for name, value in weights.items() if name != "coordination_conflict") > 0.0,
        "l1_matches_target": math.isclose(sum(abs(value) for value in weights.values()), float(target_l1), rel_tol=0.0, abs_tol=1e-12),
    }
    samples = {}
    for name, features in {
        "idle": RewardFeatures(*(0.0 for _ in range(8))),
        "typical": RewardFeatures(0.2, 0.1, 0.1, -0.1, 0.1, 0.1, 0.5, 0.1),
    }.items():
        samples[name] = combine_manual_reward(features, weights, numerical).total_reward
    checks["reward_finite"] = all(math.isfinite(value) for value in samples.values())
    return {"status": "accepted_for_training" if all(checks.values()) else "rejected_before_training", "checks": checks, "samples": samples}


def aggregate_last_two_validation(learning_curve_path, candidate_id, tail_episodes=2, include_shortened=False):
    """聚合Episode 4--5的validation；缺字段立即指出候选ID。"""
    points = json.loads(Path(learning_curve_path).read_text(encoding="utf-8"))
    complete = [point for point in points if (point.get("full_episode") or include_shortened) and point.get("data_conservation_passed") and point.get("validation")]
    if len(complete) < int(tail_episodes):
        raise ValueError("候选{0}缺少足够的完整尾部Episode".format(candidate_id))
    window = complete[-int(tail_episodes):]
    metrics = {}
    for metric in (
        "completion_rate", "delivered_data_mbit", "load_balance_mean_per_task",
        "expiration_rate", "delivered_timeliness_raw", "rejected_subaction_rate",
        "accepted_sgl_count",
    ):
        field = metric + "_mean"
        values = []
        for point in window:
            value = point.get(field)
            if value is None or not math.isfinite(float(value)):
                raise ValueError("候选{0}的尾部Episode缺少有限validation字段：{1}".format(candidate_id, field))
            values.append(float(value))
        metrics[field] = sum(values) / len(values)
    contributions = {name: 0.0 for name in _COMPONENTS}
    for point in window:
        values = point.get("reward_component_abs_sums")
        if not isinstance(values, dict):
            raise ValueError("候选{0}缺少实际奖励贡献".format(candidate_id))
        for name in _COMPONENTS:
            if name not in values:
                raise ValueError("候选{0}缺少奖励贡献字段：{1}".format(candidate_id, name))
            contributions[name] += abs(float(values[name]))
    total = sum(contributions.values())
    shares = {name: value / total if total else 0.0 for name, value in contributions.items()}
    return {"selection_window_start_episode": int(window[0]["episode_index"]) + 1, "selection_window_end_episode": int(window[-1]["episode_index"]) + 1, "selection_window_size": len(window), "aggregated_validation": metrics, "tail_episode_records": window, "reward_contribution": {"component_abs_sum": contributions, "component_share": shares, "dominant_component": max(shares, key=shares.get), "max_component_share": max(shares.values()) if shares else 0.0}}


def rank_direct_candidates(candidates, tolerance=1.0e-12):
    """严格按 Completion > Delivered Data > Load Balance 排序。"""
    def compare(left, right):
        left_validation = left.get("validation", left.get("last2_validation"))
        right_validation = right.get("validation", right.get("last2_validation"))
        for metric, direction in RANKING_METRICS:
            a = float(left_validation[metric + "_mean"])
            b = float(right_validation[metric + "_mean"])
            if math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
                continue
            return -1 if (a > b) == (direction == "max") else 1
        return -1 if left["candidate_id"] < right["candidate_id"] else (1 if left["candidate_id"] > right["candidate_id"] else 0)
    ranked = sorted(candidates, key=cmp_to_key(compare))
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index
    return ranked


def train_and_rank_candidates(candidates, baseline_config, mappo_config, output_root, smoke=False):
    """所有候选从同一初始化和同一固定150任务场景完整训练五个Episode。"""
    search = baseline_config["methods"]["llm_ppo"]["search"]
    target = int(search["candidate_training_episodes"])
    results, initial_actor, initial_critic = [], None, None
    for candidate_id, spec in candidates:
        diagnosis = diagnose_reward_spec(
            spec,
            mappo_config["manual_reward"]["numerical"],
            baseline_config["methods"]["llm_ppo"]["l1_target_scale"],
        )
        if diagnosis["status"] != "accepted_for_training":
            raise ValueError("候选{0}未通过直接奖励诊断".format(candidate_id))
        directory = Path(output_root) / "cands" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        config, encoder, actor, critic, trainer, evaluator = build_baseline_components(BaselineMethod.LLM_PPO, baseline_config, mappo_config, spec)
        actor_vector, critic_vector = parameter_vector(actor), parameter_vector(critic)
        if initial_actor is None:
            initial_actor, initial_critic = actor_vector, critic_vector
        elif not actor_vector.equal(initial_actor) or not critic_vector.equal(initial_critic):
            raise RuntimeError("LLM候选没有从相同网络初始化开始")
        runner = make_runner(BaselineMethod.LLM_PPO, trainer, evaluator, config, encoder, baseline_config, directory)
        runner.training.update({"episode_count": target, "task_count": int(baseline_config["training"]["task_count"]), "training_seed": int(baseline_config["training"]["training_seed"]), "save_episode_checkpoints": False})
        runner.training["validation"]["protocol"] = search["evaluation_protocol"]
        summary = runner.run(target_episode_count=target, max_steps_per_episode=16 if smoke else None, validation_max_steps=16 if smoke else None)
        window = aggregate_last_two_validation(summary["learning_curve_paths"]["json"], candidate_id, 2, include_shortened=smoke)
        results.append({"candidate_id": candidate_id, "reward_spec_id": spec.spec_id, "status": "trained", "diagnosis": diagnosis, "training_summary": summary, "validation": window["aggregated_validation"], "selection_window": window, "spec": spec, "weight_metadata": trainer.reward_model.weight_metadata, "episodes_trained": target})
    return results, rank_direct_candidates(results)
