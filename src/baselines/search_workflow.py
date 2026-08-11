"""执行 core-preserving LLM reward shaping 的固定分阶段候选搜索。"""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mappo.model_selection import rank_validation_results

from .candidate_search import (
    CachedRewardGenerator,
    GenerationBudget,
    train_and_rank_candidates,
)
from .llm_prompt import build_feedback_reward_prompt, build_initial_reward_prompt
from .llm_provider import DeepSeekRewardGenerationProvider, MockRewardGenerationProvider
from .llm_schema import default_mock_specs
from .run_management import atomic_write_json, sha256_file, utc_now


def _write_jsonl(path, records):
    """原子重写短JSONL审计文件，避免中途中断产生半行。"""
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)


def _task_manifest_hash():
    """哈希任务数据库、切分和摘要，运行结果可定位到固定任务清单。"""
    paths = (
        Path("data/tasks/task_database.jsonl"),
        Path("data/tasks/task_splits.json"),
        Path("data/tasks/summary.json"),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("任务清单文件不存在：{0}".format(path))
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _git_commit():
    """记录当前提交；未提交工作树不被伪装成一个新提交。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_metadata(config, provider_name, rounds, candidates_per_round):
    """写一次运行级可复现信息，候选和Episode不重复存同一份元数据。"""
    llm = config["methods"]["llm_ppo"]
    validation = config["evaluation_protocols"][llm["search"]["evaluation_protocol"]]
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "staged_llm_ppo_1.0",
        "method": "core-preserving LLM reward shaping; fixed staged candidate selection; HERON-inspired hierarchical selection",
        "timestamp": utc_now(),
        "git_commit": _git_commit(),
        "config_hash": config_hash,
        "rounds": int(rounds),
        "candidates_per_round": int(candidates_per_round),
        "provider": provider_name,
        "model": llm["provider"]["model"] if provider_name == "deepseek" else "mock",
        "alpha": config["reward_composition"]["alpha"],
        "base_reward": config["reward_composition"]["base"],
        "staged_search_config": config["staged_search"],
        "selection_rule": config["best_model_rule"],
        "train_split": config["training"]["split"],
        "train_task_count": config["methods"]["llm_ppo"]["search"]["candidate_task_count"],
        "train_seed": config["methods"]["llm_ppo"]["search"]["common_training_seeds"][0],
        "validation_protocol": config["methods"]["llm_ppo"]["search"]["evaluation_protocol"],
        "validation_task_count": validation["task_count"],
        "validation_seeds": validation["seeds"],
        "task_manifest_sha256": _task_manifest_hash(),
    }


def _candidate_record(result, round_index, candidate_index):
    """投影为短档案；完整验证样本与绝对路径不进入根目录日志。"""
    summary = result.get("training_summary") or {}
    window = result.get("selection_window") or {}
    eligibility = result.get("eligibility") or {"eligible": False, "checks": {}}
    weights = result.get("weight_metadata") or {}
    spec = result.get("spec")
    return {
        "candidate": result["candidate_id"],
        "round": round_index,
        "index": candidate_index,
        "reward_spec_id": result.get("reward_spec_id"),
        "parent": result.get("parent_candidate_id") or (
            spec.parent_candidate_id if spec else None
        ),
        "status": result.get("status"),
        "passed": bool(eligibility.get("passed", False)),
        "eligible": bool(eligibility.get("eligible", False)),
        "reasons": eligibility.get("reasons", []),
        "warnings": eligibility.get("warnings", []),
        "eligibility_checks": eligibility.get("checks", {}),
        "episodes": summary.get("episodes_run", 0),
        "validation": result.get("validation"),
        "tail": {
            "start": window.get("selection_window_start_episode"),
            "end": window.get("selection_window_end_episode"),
            "size": window.get("selection_window_size"),
        },
        "diagnostics": window.get("aggregated_diagnostics"),
        "raw_weights": weights.get("raw_weights"),
        "effective_weights": weights.get("effective_weights"),
        "rank": result.get("rank"),
        "stage_reached": result.get("stage_reached"),
        "episodes_trained": result.get("episodes_trained"),
        "eliminated_after_stage": result.get("eliminated_after_stage"),
        "final_rank_in_stage": result.get("final_rank_in_stage"),
    }


def _episode_diagnostic(point):
    """由训练器的绝对贡献汇总得到一行候选诊断。"""
    values = point["reward_component_abs_sums"]
    base_names = tuple(name for name in values if name.startswith("weighted_"))
    llm_names = (
        "llm_sgl_progress", "llm_relay_progress", "llm_completion", "llm_balance",
        "llm_expiration", "llm_invalid_action", "llm_coordination_conflict",
        "llm_relay_cost",
    )
    for name in llm_names:
        if name not in values:
            raise ValueError("候选曲线缺少LLM贡献字段：{0}".format(name))
    base = sum(float(values[name]) for name in base_names)
    llm = sum(float(values[name]) for name in llm_names)
    total = base + llm
    shares = {
        name: float(value) / total if total > 0.0 else 0.0
        for name, value in values.items()
        if name in base_names or name in llm_names
    }
    dominant = max(shares, key=shares.get) if shares else None
    return {
        "episode": int(point["episode_index"]) + 1,
        "base_abs_sum": base,
        "llm_abs_sum": llm,
        "total_abs_sum": total,
        "llm_contribution_ratio": llm / total if total > 0.0 else 0.0,
        "component_shares": shares,
        "dominant_component": dominant,
        "max_component_share": shares[dominant] if dominant else 0.0,
    }


def _compact_candidate_artifacts(candidate_dir):
    """把搜索中必要的临时日志压缩为规定的五个候选文件。"""
    candidate_dir = Path(candidate_dir)
    source = candidate_dir / "learning_curve.json"
    points = json.loads(source.read_text(encoding="utf-8"))
    curve_columns = (
        "episode", "seed", "steps", "completion_mean", "completion_std",
        "expiration_mean", "expiration_std", "timeliness_mean", "timeliness_std",
        "data_mbit_mean", "data_mbit_std", "reject_rate_mean", "reject_rate_std",
        "balance_mean", "balance_std", "sgl_frac_mean", "sgl_frac_std",
        "isl_mean", "idl_mean", "sgl_mean", "base_reward_sum", "llm_reward_sum",
        "llm_contribution_ratio", "max_component_share", "dominant_component",
        "full_episode", "conservation_ok", "eligible",
    )
    diagnostics = []
    with (candidate_dir / "curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_columns)
        writer.writeheader()
        for point in points:
            diagnostic = _episode_diagnostic(point)
            diagnostics.append(diagnostic)
            validation = point["validation"]
            sums = point["reward_component_sums"]
            writer.writerow({
                "episode": int(point["episode_index"]) + 1,
                "seed": point["episode_seed"],
                "steps": point["environment_steps"],
                "completion_mean": validation["completion_rate_mean"],
                "completion_std": validation["completion_rate_std"],
                "expiration_mean": validation["expiration_rate_mean"],
                "expiration_std": validation["expiration_rate_std"],
                "timeliness_mean": validation["delivered_timeliness_raw_mean"],
                "timeliness_std": validation["delivered_timeliness_raw_std"],
                "data_mbit_mean": validation["delivered_data_mbit_mean"],
                "data_mbit_std": validation["delivered_data_mbit_std"],
                "reject_rate_mean": validation["rejected_subaction_rate_mean"],
                "reject_rate_std": validation["rejected_subaction_rate_std"],
                "balance_mean": validation["load_balance_mean_per_task_mean"],
                "balance_std": validation["load_balance_mean_per_task_std"],
                "sgl_frac_mean": validation["sgl_action_fraction_mean"],
                "sgl_frac_std": validation["sgl_action_fraction_std"],
                "isl_mean": point["accepted_isl_count"],
                "idl_mean": point["accepted_idl_count"],
                "sgl_mean": point["accepted_sgl_count"],
                "base_reward_sum": sums["base_reward"],
                "llm_reward_sum": sums["llm_shaping_reward"],
                "llm_contribution_ratio": diagnostic["llm_contribution_ratio"],
                "max_component_share": diagnostic["max_component_share"],
                "dominant_component": diagnostic["dominant_component"],
                "full_episode": point["full_episode"],
                "conservation_ok": point["data_conservation_passed"],
                "eligible": bool(point["full_episode"] and point["data_conservation_passed"]),
            })
    _write_jsonl(candidate_dir / "diag.jsonl", diagnostics)
    last = candidate_dir / "last.pt"
    if not last.is_file():
        raise FileNotFoundError("候选缺少最新恢复Checkpoint：{0}".format(last))
    for name in (
        "learning_curve.json", "learning_curve.csv", "learning_curve.png", "summary.json",
        "train_updates.jsonl", "validation.jsonl", "episodes.jsonl", "best_checkpoint.pt",
    ):
        path = candidate_dir / name
        if path.exists():
            path.unlink()
    stages = candidate_dir / "checkpoints"
    if stages.exists():
        shutil.rmtree(stages)


def _short_selection_records(scope, traces):
    """仅记录HERON比较所需数值字段。"""
    records = []
    for item in traces:
        for trace in item["comparison"]:
            records.append({
                "scope": scope,
                "candidate_a": trace.get("candidate_a", item["higher_ranked"]),
                "candidate_b": trace.get("candidate_b", item["lower_ranked"]),
                "metric": trace["metric"],
                "mean_a": trace.get("mean_a"), "mean_b": trace.get("mean_b"),
                "se_a": trace.get("standard_error_a"), "se_b": trace.get("standard_error_b"),
                "delta": trace.get("effective_delta"),
                "winner": trace.get("decision"), "reason": trace.get("reason"),
            })
    return records


def run_reward_search(
    provider_name, baseline_config, mappo_config, output_directory, rounds,
    candidates_per_round, candidate_episodes, smoke=False, live_api_approval=None,
):
    """生成候选、按固定阶段追加训练，并用HERON冻结全部轮次的最佳奖励。"""
    if any(int(value) <= 0 for value in (rounds, candidates_per_round, candidate_episodes)):
        raise ValueError("轮数、每轮候选数和候选Episode数必须均为正整数")
    config, output = deepcopy(baseline_config), Path(output_directory)
    if output.exists():
        raise FileExistsError("LLM搜索目录已存在，拒绝覆盖")
    output.mkdir(parents=True)
    llm, search = config["methods"]["llm_ppo"], config["methods"]["llm_ppo"]["search"]
    search.update({"rounds": int(rounds), "candidates_per_round": int(candidates_per_round), "candidate_training_episodes": int(candidate_episodes)})
    atomic_write_json(output / "run.json", _run_metadata(config, provider_name, rounds, candidates_per_round))

    responses = [
        default_mock_specs()[index % len(default_mock_specs())].to_dict()
        for index in range(int(rounds) * int(candidates_per_round))
    ]
    provider = MockRewardGenerationProvider(responses) if provider_name == "mock" else DeepSeekRewardGenerationProvider(llm["provider"], approval=live_api_approval)
    budget = GenerationBudget(search["maximum_api_calls"], search["maximum_total_input_tokens"], search["maximum_total_output_tokens"])
    generator = CachedRewardGenerator(provider, llm["provider"], llm, budget)
    archive, round_history, stage_records, selection_records = [], [], [], []
    parent = last_round_best = None

    for round_index in range(1, int(rounds) + 1):
        candidates = []
        for candidate_index in range(1, int(candidates_per_round) + 1):
            candidate_id = "round_{0:02d}_candidate_{1:02d}".format(round_index, candidate_index)
            prompt = build_initial_reward_prompt(mappo_config["manual_reward"]["weights"], {"candidate_id": candidate_id}) if parent is None else build_feedback_reward_prompt(parent["weight_metadata"]["effective_weights"], parent["candidate_id"], parent["training_summary"], parent["validation"]) + "\n本次候选ID：{0}".format(candidate_id)
            spec, audit = generator.generate(prompt, {"candidate_id": candidate_id})
            candidate_dir = output / "cands" / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            spec.save(candidate_dir / "spec.json")
            atomic_write_json(candidate_dir / "gen.json", audit)
            candidates.append((candidate_id, spec))
        round_results, best = train_and_rank_candidates(candidates, config, mappo_config, output, smoke=smoke)
        for index, result in enumerate(round_results, start=1):
            result["parent_candidate_id"] = parent["candidate_id"] if parent else None
            archive.append(_candidate_record(result, round_index, index))
        eligible_records = [
            {"candidate_id": item["candidate"], "reward_spec_id": item["reward_spec_id"], "validation": item["validation"], "entry": item}
            for item in archive if item["eligible"] and item["validation"]
        ]
        ranked, traces = rank_validation_results(eligible_records, rule=config["best_model_rule"], return_traces=True)
        selection_records.extend(_short_selection_records("round_{0:02d}".format(round_index), traces))
        global_best = ranked[0]["entry"] if ranked else None
        round_history.append({"round": round_index, "candidates": [item["candidate_id"] for item in round_results], "round_best": best["candidate_id"] if best else None, "global_best": global_best["candidate"] if global_best else None})
        if best is None:
            raise RuntimeError("本轮没有满足资格门槛的候选")
        for record in round_results[0].get("stage_records", []):
            stage_records.append({"round": round_index, **record})
        parent, last_round_best = best, best

    eligible_records = [
        {"candidate_id": item["candidate"], "reward_spec_id": item["reward_spec_id"], "validation": item["validation"], "entry": item}
        for item in archive if item["eligible"] and item["validation"]
    ]
    ranked, traces = rank_validation_results(eligible_records, rule=config["best_model_rule"], return_traces=True)
    if not ranked:
        raise RuntimeError("全部轮次均无合格候选，无法冻结奖励规范")
    final_best = ranked[0]["entry"]
    selection_records.extend(_short_selection_records("global", traces))
    for rank, record in enumerate(ranked, start=1):
        record["entry"]["global_rank"] = rank
    _write_jsonl(output / "cands.jsonl", archive)
    _write_jsonl(output / "rounds.jsonl", round_history)
    _write_jsonl(output / "stages.jsonl", stage_records)
    _write_jsonl(output / "select.jsonl", selection_records)
    atomic_write_json(output / "rank.json", {"ranking": [item["entry"] for item in ranked]})
    reward_path = output / "reward.json"
    shutil.copy2(output / "cands" / final_best["candidate"] / "spec.json", reward_path)
    for candidate_dir in (output / "cands").iterdir():
        _compact_candidate_artifacts(candidate_dir)
    summary = {
        "provider": provider_name, "model": llm["provider"]["model"] if provider_name == "deepseek" else provider.model,
        "rounds": int(rounds), "candidates_per_round": int(candidates_per_round),
        "candidate_training_episodes": int(candidate_episodes),
        "last_round_best_candidate_id": last_round_best["candidate_id"],
        "global_best_candidate_id": final_best["candidate"], "selected_candidate_id": final_best["candidate"],
        "selected_reward_spec_id": final_best["reward_spec_id"], "selected_reward_spec_path": "reward.json",
        "budget": {"api_calls": budget.api_calls, "input_tokens": budget.input_tokens, "output_tokens": budget.output_tokens},
    }
    return reward_path, summary
