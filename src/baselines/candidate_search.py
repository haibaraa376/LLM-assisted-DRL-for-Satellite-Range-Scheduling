"""提供LLM奖励候选诊断、缓存、预算和公平基线搜索。"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from mappo.evaluation_protocol import build_evaluation_protocol
from mappo.model_selection import rank_validation_results
from mappo.manual_reward import RewardFeatures, combine_manual_reward
from mappo.trainer import parameter_vector

from .baseline_runner import (
    build_baseline_components,
    make_runner,
)
from .llm_prompt import system_prompt
from .llm_provider import FatalProviderError, RetryableProviderError
from .llm_reward import reward_spec_weights
from .llm_schema import LlmRewardSpec
from .methods import BaselineMethod


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


def assess_candidate_eligibility(training_summary, validation, config):
    """根据完整Episode、守恒、SGL和奖励支配比例判定候选资格。"""
    scenarios = training_summary.get("best_validation_scenarios", [])
    checks = {
        "full_episode": bool(scenarios)
        and all(scenario.get("full_episode", False) for scenario in scenarios),
        "data_conservation": bool(
            training_summary.get("best_validation_data_conservation", False)
        ),
        "accepted_sgl": float(validation["accepted_sgl_count_mean"])
        >= float(config["minimum_accepted_sgl_mean"]),
        "reward_dominance": float(
            training_summary["reward_diagnostics"][
                "maximum_single_component_dominance"
            ]
        )
        <= float(config["maximum_single_component_dominance"]),
    }
    return {"eligible": all(checks.values()), "checks": checks}


def train_and_rank_candidates(
    candidates,
    baseline_config,
    mappo_config,
    output_root,
    smoke=False,
):
    """从相同初始化公平训练候选，并按固定validation规则排序。"""
    results = []
    initial_actor = None
    initial_critic = None
    search = baseline_config["methods"]["llm_ppo"]["search"]
    for candidate_id, spec in candidates:
        diagnosis = diagnose_reward_spec(
            spec,
            mappo_config["manual_reward"]["numerical"],
            mappo_config["manual_reward"]["weights"],
        )
        candidate_dir = Path(output_root) / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        spec.save(candidate_dir / "reward_spec.json")
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
        runner.training["base_episode_seed"] = int(
            search["common_training_seeds"][0]
        )
        runner.training["validation"]["protocol"] = search[
            "evaluation_protocol"
        ]
        summary = runner.run(
            target_episode_count=runner.training["episode_count"],
            max_steps_per_episode=64 if smoke else None,
            validation_max_steps=32 if smoke else None,
        )
        results.append(
            {
                "candidate_id": candidate_id,
                "reward_spec_id": spec.spec_id,
                "diagnosis": diagnosis,
                "status": "trained",
                "official_experiment": not smoke,
                "training_summary": summary,
                "validation": summary["best_validation_result"],
                "spec": spec,
                "weight_metadata": trainer.reward_model.weight_metadata,
            }
        )
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
        )
        item["rank"] = None
    eligible = [item for item in trained if item["eligibility"]["eligible"]]
    ranked = rank_validation_results(
        eligible,
        metrics_getter=lambda item: item["validation"],
        rule=baseline_config["best_model_rule"]["metrics"],
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return results, ranked[0] if ranked else None
