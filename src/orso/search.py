"""复用既有LLM-PPO与MAPPO组件，实现ORSO的固定候选集和D3RB调度。"""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

from baselines.baseline_runner import build_baseline_components, make_runner
from baselines.candidate_search import (
    CachedRewardGenerator,
    GenerationBudget,
    diagnose_reward_spec,
    rank_direct_candidates,
)
from baselines.llm_prompt import build_initial_reward_prompt
from baselines.llm_provider import (
    DeepSeekRewardGenerationProvider,
    MockRewardGenerationProvider,
)
from baselines.llm_schema import default_mock_specs
from baselines.llm_reward import normalized_reward_spec_weights
from baselines.methods import BaselineMethod
from baselines.run_management import atomic_write_json, utc_now
from mappo.trainer import parameter_vector

from .config import validate_orso_config
from .d3rb import D3RBSelector


_SMOKE_OVERRIDES = {
    "generation": {"candidates": 3},
    "training": {
        "warmup_episodes_per_candidate": 1,
        "total_candidate_episode_budget": 5,
        "max_episodes_per_candidate": 3,
    },
}


@dataclass
class _OrsoLearner:
    """把一个独立MAPPO训练器及其跨Episode Runner状态绑定到固定奖励候选。"""

    candidate_id: str
    spec: object
    trainer: object
    runner: object
    weight_metadata: dict
    initial_actor: object
    initial_critic: object
    update_index: int = 0
    best_validation_result: object = None
    best_episode_index: object = None
    best_validation_details: object = None


def apply_smoke_overrides(config):
    """只在--smoke时使用小预算；正式配置文件不会被改写。"""
    smoke = deepcopy(config)
    for section, values in _SMOKE_OVERRIDES.items():
        smoke[section].update(values)
    validate_orso_config(smoke)
    return smoke


def build_warmup_schedule(candidate_ids, warmup_episodes):
    """按candidate_01...顺序循环，确保warmup的每次观测都更新D3RB。"""
    return [
        candidate_id
        for _ in range(int(warmup_episodes))
        for candidate_id in candidate_ids
    ]


def _append_jsonl(path, record):
    """逐次落盘预算分配记录，便于搜索中断后检查已完成决策。"""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _task_utility(candidate_id, validation, utility_config):
    """D3RB唯一任务效用：reward_search validation的completion_rate_mean。"""
    metric = utility_config["primary_metric"]
    if not isinstance(validation, dict) or metric not in validation:
        raise ValueError("候选{0}缺少任务效用字段：{1}".format(candidate_id, metric))
    value = float(validation[metric])
    if not math.isfinite(value):
        raise ValueError("候选{0}的任务效用不是有限数：{1}".format(candidate_id, metric))
    lower, upper = float(utility_config["valid_min"]), float(utility_config["valid_max"])
    if not lower <= value <= upper:
        raise ValueError("候选{0}的任务效用超出范围[{1}, {2}]：{3}".format(candidate_id, lower, upper, value))
    return value


def _latest_validation(candidate_id, summary):
    """从刚写入的统一学习曲线取得本次完整reward_search验证汇总。"""
    curve_path = Path(summary["learning_curve_paths"]["json"])
    points = json.loads(curve_path.read_text(encoding="utf-8"))
    if not points:
        raise ValueError("候选{0}缺少学习曲线记录".format(candidate_id))
    validation = points[-1].get("validation")
    if not isinstance(validation, dict):
        raise ValueError("候选{0}缺少本次reward_search validation".format(candidate_id))
    return validation


def _is_better_validation(current, previous):
    """ORSO最终比较保持既有 Completion > Data > Balance 层级。"""
    if previous is None:
        return True
    metrics = (
        "completion_rate_mean",
        "delivered_data_mbit_mean",
        "load_balance_mean_per_task_mean",
    )
    for metric in metrics:
        current_value, previous_value = float(current[metric]), float(previous[metric])
        if not math.isfinite(current_value) or not math.isfinite(previous_value):
            raise ValueError("最终选择验证字段不是有限数：{0}".format(metric))
        if not math.isclose(current_value, previous_value, rel_tol=0.0, abs_tol=1.0e-12):
            return current_value > previous_value
    return False


def _build_learners(candidates, baseline_config, mappo_config, output):
    """为每个奖励创建完全独立、但初始参数一致的MAPPO learner。"""
    learners = {}
    initial_actor, initial_critic = None, None
    for candidate_id, spec in candidates:
        diagnosis = diagnose_reward_spec(
            spec,
            mappo_config["manual_reward"]["numerical"],
            baseline_config["methods"]["llm_ppo"]["l1_target_scale"],
        )
        if diagnosis["status"] != "accepted_for_training":
            raise ValueError("ORSO候选{0}未通过直接LLM奖励诊断".format(candidate_id))
        directory = Path(output) / "cands" / candidate_id
        config, encoder, actor, critic, trainer, evaluator = build_baseline_components(
            BaselineMethod.LLM_PPO,
            baseline_config,
            mappo_config,
            spec,
        )
        config["candidate_id"] = candidate_id
        actor_vector = parameter_vector(actor).detach().clone()
        critic_vector = parameter_vector(critic).detach().clone()
        if initial_actor is None:
            initial_actor, initial_critic = actor_vector, critic_vector
        elif not actor_vector.equal(initial_actor) or not critic_vector.equal(initial_critic):
            raise RuntimeError("ORSO候选没有从相同网络初始化开始")
        if id(actor) in {id(item.trainer.actor) for item in learners.values()}:
            raise RuntimeError("ORSO候选错误共享Actor对象")
        if id(critic) in {id(item.trainer.critic) for item in learners.values()}:
            raise RuntimeError("ORSO候选错误共享Critic对象")
        runner = make_runner(
            BaselineMethod.LLM_PPO,
            trainer,
            evaluator,
            config,
            encoder,
            baseline_config,
            directory,
        )
        runner.training.update(
            {
                "task_count": int(baseline_config["training"]["task_count"]),
                "training_seed": int(baseline_config["training"]["training_seed"]),
                "save_episode_checkpoints": False,
                "write_learning_curves": True,
            }
        )
        # ORSO的搜索阶段只能访问reward_search，不能接触checkpoint_selection或test。
        runner.training["validation"]["protocol"] = "reward_search"
        learners[candidate_id] = _OrsoLearner(
            candidate_id=candidate_id,
            spec=spec,
            trainer=trainer,
            runner=runner,
            weight_metadata=deepcopy(trainer.reward_model.weight_metadata),
            initial_actor=actor_vector,
            initial_critic=critic_vector,
        )
    return learners


def _train_one_episode(learner, selector, config, smoke):
    """把一个Episode预算交给指定独立learner，并立即做reward_search验证。"""
    state = selector.states[learner.candidate_id]
    before = state.episodes_trained
    untouched = next(
        (
            item
            for item in selector.states.values()
            if item.candidate_id != learner.candidate_id and item.episodes_trained == 0
        ),
        None,
    )
    peer_actor_before = None
    peer_critic_before = None
    if untouched is not None:
        peer = learner._all_learners[untouched.candidate_id]
        peer_actor_before = parameter_vector(peer.trainer.actor).detach().clone()
        peer_critic_before = parameter_vector(peer.trainer.critic).detach().clone()
    summary = learner.runner.run(
        target_episode_count=before + 1,
        start_episode_index=before,
        update_index=learner.update_index,
        best_validation_result=learner.best_validation_result,
        best_episode_index=learner.best_episode_index,
        best_validation_details=learner.best_validation_details,
        max_steps_per_episode=16 if smoke else None,
        validation_max_steps=16 if smoke else None,
        resume=before > 0,
    )
    learner.update_index = int(summary["total_update_index"])
    learner.best_validation_result = summary["best_validation_result"]
    learner.best_episode_index = summary["best_episode_index"]
    learner.best_validation_details = summary["best_validation_details"]
    validation = _latest_validation(learner.candidate_id, summary)
    reward = _task_utility(
        learner.candidate_id,
        validation,
        config["task_utility"],
    )
    if _is_better_validation(validation, selector.states[learner.candidate_id].best_validation):
        selector.states[learner.candidate_id].best_validation = dict(validation)

    # 候选要继续被D3RB选中，保留last即可；统一Runner自动生成的best不保留。
    best_path = Path(learner.runner.training["checkpoint"]["best_path"])
    if best_path.exists():
        best_path.unlink()
    if peer_actor_before is not None:
        peer = learner._all_learners[untouched.candidate_id]
        if not parameter_vector(peer.trainer.actor).equal(peer_actor_before):
            raise RuntimeError("ORSO训练一个候选时修改了未训练候选的Actor")
        if not parameter_vector(peer.trainer.critic).equal(peer_critic_before):
            raise RuntimeError("ORSO训练一个候选时修改了未训练候选的Critic")
    return validation, reward


def _candidate_record(candidate_id, learner, state, duplicate_of, selected):
    """输出与既有LLM-PPO候选记录兼容的精简字段。"""
    return {
        "candidate_id": candidate_id,
        "reward_spec_id": learner.spec.spec_id,
        "raw_weights": learner.weight_metadata["raw_weights"],
        "effective_weights": learner.weight_metadata["effective_weights"],
        "target_weight_l1": learner.weight_metadata["target_weight_l1"],
        "episodes_trained": state.episodes_trained,
        "latest_validation": state.latest_validation,
        "best_validation": state.best_validation,
        "mean_task_reward": state.mean_task_reward,
        "d_hat": state.d_hat,
        "phi": state.phi,
        "misspecification_count": state.misspecification_count,
        "duplicate_reward_weights": duplicate_of is not None,
        "duplicate_of": duplicate_of,
        "final_rank": None,
        "selected": selected,
    }


def run_orso_search(
    provider_name,
    orso_config,
    baseline_config,
    mappo_config,
    output_directory,
    smoke=False,
    live_api_approval=None,
):
    """运行固定奖励集合的ORSO/D3RB搜索；不调用现有多轮LLM-PPO工作流。"""
    config = apply_smoke_overrides(orso_config) if smoke else deepcopy(orso_config)
    validate_orso_config(config)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("ORSO搜索目录已存在，拒绝覆盖")
    output.mkdir(parents=True)

    generation, training = config["generation"], config["training"]
    llm = baseline_config["methods"]["llm_ppo"]
    candidate_count = int(generation["candidates"])
    candidate_ids = tuple(
        "candidate_{0:02d}".format(index)
        for index in range(1, candidate_count + 1)
    )
    atomic_write_json(
        output / "run.json",
        {
            "schema_version": "orso_d3rb_1.0",
            "timestamp": utc_now(),
            "provider": provider_name,
            "generation_rounds": int(generation["rounds"]),
            "candidate_count": candidate_count,
            "task_count": int(baseline_config["training"]["task_count"]),
            "training_seed": int(baseline_config["training"]["training_seed"]),
            "evaluation_protocol": training["evaluation_protocol"],
            "selection_algorithm": "d3rb",
            "allocation_unit": "episode",
            "warmup_mode": "project_round_robin",
            "candidate_cap": int(training["max_episodes_per_candidate"]),
            "project_specific_orso_adaptation": True,
        },
    )
    mock_responses = [
        default_mock_specs()[index % len(default_mock_specs())].to_dict()
        for index in range(candidate_count)
    ]
    provider = (
        MockRewardGenerationProvider(mock_responses)
        if provider_name == "mock"
        else DeepSeekRewardGenerationProvider(llm["provider"], approval=live_api_approval)
    )
    search_budget = llm["search"]
    generator = CachedRewardGenerator(
        provider,
        llm["provider"],
        llm,
        GenerationBudget(
            search_budget["maximum_api_calls"],
            search_budget["maximum_total_input_tokens"],
            search_budget["maximum_total_output_tokens"],
        ),
    )
    candidates, duplicate_of = [], {}
    seen_raw, seen_effective = {}, {}
    for index, candidate_id in enumerate(candidate_ids, start=1):
        prompt = build_initial_reward_prompt(
            baseline_config["training"]["task_count"],
            llm["l1_target_scale"],
            1,
            index,
            candidate_count,
            candidate_id,
        )
        spec, audit = generator.generate(
            prompt,
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "round_index": 1,
                "task_count": int(baseline_config["training"]["task_count"]),
            },
        )
        directory = output / "cands" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        spec.save(directory / "spec.json")
        atomic_write_json(directory / "gen.json", audit)
        effective_weights, weight_metadata = normalized_reward_spec_weights(
            spec,
            target_l1=llm["l1_target_scale"],
        )
        raw_key = json.dumps(
            weight_metadata["raw_weights"],
            ensure_ascii=False,
            sort_keys=True,
        )
        effective_key = json.dumps(
            effective_weights,
            ensure_ascii=False,
            sort_keys=True,
        )
        duplicate_of[candidate_id] = seen_raw.get(raw_key) or seen_effective.get(effective_key)
        seen_raw.setdefault(raw_key, candidate_id)
        seen_effective.setdefault(effective_key, candidate_id)
        candidates.append((candidate_id, spec))

    learners = _build_learners(candidates, baseline_config, mappo_config, output)
    for learner in learners.values():
        learner._all_learners = learners
    selector = D3RBSelector(
        candidate_ids,
        config["d3rb"]["d_min"],
        config["d3rb"]["delta"],
        config["d3rb"]["confidence_constant"],
    )
    allocation_path = output / "allocation.jsonl"
    warmup_schedule = build_warmup_schedule(
        candidate_ids,
        training["warmup_episodes_per_candidate"],
    )
    total_budget = int(training["total_candidate_episode_budget"])
    maximum = int(training["max_episodes_per_candidate"])
    for global_step in range(1, total_budget + 1):
        phase = "warmup" if global_step <= len(warmup_schedule) else "d3rb"
        candidate_id = (
            warmup_schedule[global_step - 1]
            if phase == "warmup"
            else selector.select(maximum)
        )
        learner = learners[candidate_id]
        state = selector.states[candidate_id]
        episodes_before = state.episodes_trained
        validation, task_reward = _train_one_episode(
            learner,
            selector,
            config,
            smoke,
        )
        update = selector.update(candidate_id, task_reward, validation)
        _append_jsonl(
            allocation_path,
            {
                "global_step": global_step,
                "phase": phase,
                "candidate_id": candidate_id,
                "episodes_before": episodes_before,
                "episodes_after": state.episodes_trained,
                "task_reward": task_reward,
                "completion_rate_mean": validation["completion_rate_mean"],
                "delivered_data_mbit_mean": validation["delivered_data_mbit_mean"],
                "load_balance_mean_per_task_mean": validation[
                    "load_balance_mean_per_task_mean"
                ],
                **update,
                "remaining_budget": total_budget - global_step,
            },
        )

    used_budget = sum(state.episodes_trained for state in selector.states.values())
    if used_budget != total_budget:
        raise RuntimeError("ORSO实际预算{0}不等于配置总预算{1}".format(used_budget, total_budget))
    if any(state.episodes_trained > maximum for state in selector.states.values()):
        raise RuntimeError("ORSO候选训练次数超过配置上限")

    ranking_input = [
        {"candidate_id": candidate_id, "validation": state.latest_validation}
        for candidate_id, state in selector.states.items()
    ]
    ranked = rank_direct_candidates(ranking_input)
    selected_candidate_id = ranked[0]["candidate_id"]
    final_rank = {item["candidate_id"]: item["rank"] for item in ranked}
    records = []
    for candidate_id in candidate_ids:
        record = _candidate_record(
            candidate_id,
            learners[candidate_id],
            selector.states[candidate_id],
            duplicate_of[candidate_id],
            candidate_id == selected_candidate_id,
        )
        record["final_rank"] = final_rank[candidate_id]
        records.append(record)
    for record in records:
        _append_jsonl(output / "candidates.jsonl", record)
    atomic_write_json(output / "ranking.json", {"ranking": sorted(records, key=lambda item: item["final_rank"])})
    reward_path = output / "reward.json"
    shutil.copy2(output / "cands" / selected_candidate_id / "spec.json", reward_path)
    d3rb_final = {
        "selection_algorithm": "d3rb",
        "allocation_unit": "episode",
        "warmup_mode": "project_round_robin",
        "project_specific_orso_adaptation": True,
        "total_budget": total_budget,
        "used_budget": used_budget,
        "warmup_budget": len(warmup_schedule),
        "dynamic_budget": total_budget - len(warmup_schedule),
        "candidate_cap": maximum,
        "selected_candidate_id": selected_candidate_id,
        "candidates": [selector.states[candidate_id].to_dict() for candidate_id in candidate_ids],
    }
    atomic_write_json(output / "d3rb_final.json", d3rb_final)
    summary = {
        "method": "orso",
        "selection_algorithm": "d3rb",
        "allocation_unit": "episode",
        "warmup_mode": "project_round_robin",
        "candidate_cap": maximum,
        "project_specific_orso_adaptation": True,
        "selected_candidate_id": selected_candidate_id,
        "selected_reward_spec_id": learners[selected_candidate_id].spec.spec_id,
        "total_candidate_episode_budget": total_budget,
        "used_candidate_episode_budget": used_budget,
        "provider": provider_name,
        "api_calls": generator.budget.api_calls,
        "initial_networks_identical": True,
        "independent_learner_updates_verified": True,
    }
    atomic_write_json(output / "orso_summary.json", summary)
    return reward_path, summary
