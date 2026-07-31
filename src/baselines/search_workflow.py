"""复用安全Provider和公平候选训练实现完整LLM奖励搜索。"""

from copy import deepcopy
import json
from pathlib import Path

from .candidate_search import (
    CachedRewardGenerator,
    GenerationBudget,
    train_and_rank_candidates,
)
from .llm_prompt import build_feedback_reward_prompt, build_initial_reward_prompt
from .llm_provider import (
    DeepSeekRewardGenerationProvider,
    MockRewardGenerationProvider,
)
from .llm_reward import reward_spec_weights
from .llm_schema import default_mock_specs
from .run_management import atomic_write_json


def run_reward_search(
    provider_name,
    baseline_config,
    mappo_config,
    output_directory,
    rounds,
    candidates_per_round,
    candidate_episodes,
    smoke=False,
    live_api_approval=None,
):
    """生成、诊断、公平训练候选并冻结validation最佳规范。"""
    config = deepcopy(baseline_config)
    llm = config["methods"]["llm_ppo"]
    search = llm["search"]
    search["rounds"] = int(rounds)
    search["candidates_per_round"] = int(candidates_per_round)
    search["candidate_training_episodes"] = int(candidate_episodes)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("LLM搜索目录已存在，拒绝覆盖")
    output.mkdir(parents=True)
    llm["cache"]["directory"] = str(output / "api_calls")
    total_candidates = int(rounds) * int(candidates_per_round)
    mock_specs = default_mock_specs()
    responses = [
        mock_specs[index % len(mock_specs)].to_dict()
        for index in range(total_candidates)
    ]
    provider = (
        MockRewardGenerationProvider(responses)
        if provider_name == "mock"
        else DeepSeekRewardGenerationProvider(
            llm["provider"],
            approval=live_api_approval,
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
    rounds_path = output / "search_rounds.jsonl"
    if rounds_path.exists():
        raise FileExistsError("搜索结果目录已存在，拒绝覆盖")
    parent = None
    all_results = []
    with rounds_path.open("x", encoding="utf-8") as rounds_stream:
        for round_index in range(1, int(rounds) + 1):
            candidates = []
            for candidate_index in range(1, int(candidates_per_round) + 1):
                candidate_id = "round_{0:02d}_candidate_{1:02d}".format(
                    round_index,
                    candidate_index,
                )
                if parent is None:
                    prompt = build_initial_reward_prompt(
                        mappo_config["manual_reward"]["weights"],
                        {"candidate_id": candidate_id},
                    )
                else:
                    prompt = build_feedback_reward_prompt(
                        reward_spec_weights(parent["spec"]),
                        parent["candidate_id"],
                        parent["training_summary"],
                        parent["validation"],
                    ) + "\n本次候选ID：{0}".format(candidate_id)
                spec, audit = generator.generate(
                    prompt,
                    {"candidate_id": candidate_id},
                )
                candidate_dir = output / "candidates" / candidate_id
                spec.save(candidate_dir / "reward_spec.json")
                atomic_write_json(candidate_dir / "generation.json", audit)
                candidates.append((candidate_id, spec))
            round_results, best = train_and_rank_candidates(
                candidates,
                config,
                mappo_config,
                output,
                smoke=smoke,
            )
            parent = best
            serializable = [_safe_record(item) for item in round_results]
            rounds_stream.write(
                json.dumps(
                    {
                        "round": round_index,
                        "official_experiment": not smoke,
                        "candidates": serializable,
                        "selected_candidate_id": best["candidate_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rounds_stream.flush()
            all_results.extend(serializable)
    selected_path = output / "selected_reward_spec.json"
    parent["spec"].save(selected_path)
    summary = {
        "provider": provider_name,
        "model": (
            llm["provider"]["model"]
            if provider_name == "deepseek"
            else provider.model
        ),
        "official_experiment": not smoke,
        "rounds": int(rounds),
        "candidates_per_round": int(candidates_per_round),
        "selected_candidate_id": parent["candidate_id"],
        "selected_reward_spec_id": parent["spec"].spec_id,
        "selected_reward_spec_path": str(selected_path),
        "budget": {
            "api_calls": budget.api_calls,
            "input_tokens": budget.input_tokens,
            "output_tokens": budget.output_tokens,
        },
        "candidates": all_results,
    }
    atomic_write_json(output / "search_summary.json", summary)
    return selected_path, summary


def _safe_record(result):
    return {
        key: (value.to_dict() if key == "spec" else value)
        for key, value in result.items()
    }
