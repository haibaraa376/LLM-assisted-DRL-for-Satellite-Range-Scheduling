"""执行固定5轮×8候选×5 Episode的直接LLM奖励搜索。"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil

from .candidate_search import CachedRewardGenerator, GenerationBudget, rank_direct_candidates, train_and_rank_candidates
from .llm_prompt import build_feedback_reward_prompt, build_initial_reward_prompt
from .llm_provider import DeepSeekRewardGenerationProvider, MockRewardGenerationProvider
from .llm_schema import default_mock_specs
from .run_management import atomic_write_json, utc_now


def _write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _candidate_record(result, round_index, candidate_index):
    window = result["selection_window"]
    episode_metrics = []
    for point in window["tail_episode_records"]:
        episode_metrics.append({
            "episode": int(point["episode_index"]) + 1,
            "validation": point["validation"],
            "episode_reward": point.get("mean_step_reward"),
        })
    return {
        "candidate_id": result["candidate_id"], "round": round_index, "index": candidate_index,
        "reward_spec_id": result["reward_spec_id"], "status": result["status"],
        "episodes_trained": result["episodes_trained"], "raw_weights": result["weight_metadata"]["raw_weights"],
        "effective_weights": result["weight_metadata"]["effective_weights"],
        "target_weight_l1": result["weight_metadata"]["target_weight_l1"],
        "last2_validation": result["validation"], "last2_window": {key: window[key] for key in ("selection_window_start_episode", "selection_window_end_episode", "selection_window_size")},
        "reward_contribution": window["reward_contribution"], "tail_episode_metrics": episode_metrics,
        "round_rank": result.get("rank"), "global_rank": None,
        "duplicate_reward_weights": False, "duplicate_of": None, "warnings": [],
    }


def _compact_candidate_artifacts(candidate_dir):
    """保留候选曲线、Episode日志和生成规格，删除候选checkpoint及冗余日志。"""
    directory = Path(candidate_dir)
    points = json.loads((directory / "learning_curve.json").read_text(encoding="utf-8"))
    fields = ("episode", "completion_rate_mean", "delivered_data_mbit_mean", "load_balance_mean_per_task_mean", "expiration_rate_mean", "delivered_timeliness_raw_mean", "rejected_subaction_rate_mean", "mean_step_reward")
    with (directory / "curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({"episode": int(point["episode_index"]) + 1, **{name: point.get(name) for name in fields if name != "episode"}})
    for name in ("learning_curve.json", "learning_curve.csv", "summary.json", "train_updates.jsonl", "validation.jsonl", "best_checkpoint.pt", "last_checkpoint.pt"):
        path = directory / name
        if path.exists():
            path.unlink()
    stages = directory / "checkpoints"
    if stages.exists():
        shutil.rmtree(stages)


def run_reward_search(provider_name, baseline_config, mappo_config, output_directory, rounds, candidates_per_round, candidate_episodes, smoke=False, live_api_approval=None):
    """所有候选使用相同任务、初始化、MAPPO参数及其配置的训练预算。"""
    config, output = deepcopy(baseline_config), Path(output_directory)
    if output.exists():
        raise FileExistsError("LLM搜索目录已存在，拒绝覆盖")
    output.mkdir(parents=True)
    llm, search = config["methods"]["llm_ppo"], config["methods"]["llm_ppo"]["search"]
    search.update({"rounds": int(rounds), "candidates_per_round": int(candidates_per_round), "candidate_training_episodes": int(candidate_episodes)})
    atomic_write_json(output / "run.json", {"schema_version": "direct_llm_ppo_2.1", "timestamp": utc_now(), "provider": provider_name, "rounds": int(rounds), "candidates_per_round": int(candidates_per_round), "episodes_per_candidate": int(candidate_episodes), "task_count": int(config["training"]["task_count"]), "training_seed": int(config["training"]["training_seed"]), "reward_mode": llm["reward_mode"], "reward_schema_version": llm["reward_schema_version"], "l1_target_scale": llm["l1_target_scale"], "conflict_fixed_zero": True, "ranking": "completion_rate > delivered_data_mbit > load_balance_mean_per_task"})
    responses = [default_mock_specs()[index % len(default_mock_specs())].to_dict() for index in range(int(rounds) * int(candidates_per_round))]
    provider = MockRewardGenerationProvider(responses) if provider_name == "mock" else DeepSeekRewardGenerationProvider(llm["provider"], approval=live_api_approval)
    budget = GenerationBudget(search["maximum_api_calls"], search["maximum_total_input_tokens"], search["maximum_total_output_tokens"])
    generator = CachedRewardGenerator(provider, llm["provider"], llm, budget)
    archive, rounds_log, parent = [], [], None
    for round_index in range(1, int(rounds) + 1):
        generated = []
        for candidate_index in range(1, int(candidates_per_round) + 1):
            candidate_id = "round_{0:02d}_candidate_{1:02d}".format(round_index, candidate_index)
            prompt_args = (
                config["training"]["task_count"], llm["l1_target_scale"], round_index,
                candidate_index, candidates_per_round, candidate_id,
            )
            prompt = (
                build_initial_reward_prompt(*prompt_args)
                if parent is None
                else build_feedback_reward_prompt(
                    *prompt_args, parent["candidate_id"], parent
                )
            )
            spec, audit = generator.generate(prompt, {
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "round_index": round_index,
                "task_count": int(config["training"]["task_count"]),
            })
            directory = output / "cands" / candidate_id
            directory.mkdir(parents=True, exist_ok=True)
            spec.save(directory / "spec.json")
            atomic_write_json(directory / "gen.json", audit)
            generated.append((candidate_id, spec))
        results, ranked = train_and_rank_candidates(generated, config, mappo_config, output, smoke=smoke)
        records = [_candidate_record(result, round_index, index) for index, result in enumerate(results, start=1)]
        seen_raw, seen_effective = {}, {}
        for record in records:
            raw_key = json.dumps(record["raw_weights"], sort_keys=True, separators=(",", ":"))
            effective_key = json.dumps(record["effective_weights"], sort_keys=True, separators=(",", ":"))
            duplicate_of = seen_raw.get(raw_key) or seen_effective.get(effective_key)
            if duplicate_of:
                record["duplicate_reward_weights"] = True
                record["duplicate_of"] = duplicate_of
                record["warnings"].append("duplicate_reward_weights")
            seen_raw.setdefault(raw_key, record["candidate_id"])
            seen_effective.setdefault(effective_key, record["candidate_id"])
        archive.extend(records)
        parent = next(record for record in records if record["candidate_id"] == ranked[0]["candidate_id"])
        rounds_log.append({"round": round_index, "round_best_candidate_id": parent["candidate_id"], "candidate_ids": [item[0] for item in generated]})
    ranked_archive = rank_direct_candidates(archive)
    for item in ranked_archive:
        item["global_rank"] = item["rank"]
    final = ranked_archive[0]
    _write_jsonl(output / "candidates.jsonl", archive)
    _write_jsonl(output / "rounds.jsonl", rounds_log)
    atomic_write_json(output / "ranking.json", {"ranking": ranked_archive})
    reward_path = output / "reward.json"
    shutil.copy2(output / "cands" / final["candidate_id"] / "spec.json", reward_path)
    for directory in (output / "cands").iterdir():
        _compact_candidate_artifacts(directory)
    return reward_path, {"provider": provider_name, "rounds": int(rounds), "candidates_per_round": int(candidates_per_round), "candidate_training_episodes": int(candidate_episodes), "global_best_candidate_id": final["candidate_id"], "selected_reward_spec_id": final["reward_spec_id"], "budget": {"api_calls": budget.api_calls, "input_tokens": budget.input_tokens, "output_tokens": budget.output_tokens}}
