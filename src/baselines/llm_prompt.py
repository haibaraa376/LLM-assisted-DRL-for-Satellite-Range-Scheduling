"""构造无RAG、无代码且候选身份明确的直接LLM奖励提示。"""

import json


_FEATURES = """固定RewardFeatures：sgl_progress、relay_progress、completion_score、
balance_score、expiration_loss、invalid_action_rate、coordination_conflict_rate、relay_cost。"""


def system_prompt():
    return "你是卫星调度奖励设计审查员。只能输出单个合法JSON对象，不得输出代码、路径或Markdown。"


def _prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, parent_candidate_id, parent_summary):
    """每个候选携带独立身份，避免同轮Prompt和缓存意外共享。"""
    example = {
        "schema_version": "2.1",
        "reward_name": "deadline_delivery_v2",
        "positive_weights": {"sgl_progress": 1.2, "relay_progress": 0.1, "completion_score": 0.8, "balance_score": 0.03},
        "penalty_weights": {"expiration_loss": 0.9, "invalid_action_rate": 0.08, "coordination_conflict_rate": 0.0, "relay_cost": 0.04},
        "rationale": "优先最终下传与完成，并抑制过期和无效动作。",
        "expected_behavior": ["提高完成率"], "risk_notes": [],
        "parent_candidate_id": parent_candidate_id,
    }
    return """请为无RAG的DeepSeek LLM-PPO生成最终奖励权重JSON。

训练任务数：{task_count}。当前搜索轮次：{round_index}。当前候选：{candidate_id}。
这是本轮第{candidate_index}/{candidate_total}个独立候选；请生成新的独立方案，不要简单复制父候选权重。
LLM生成的奖励直接作为MAPPO训练总奖励，不存在R_base、alpha或附加塑形。
{features}
七项raw权重只表达相对偏好，程序会保持比例并统一缩放，使七项有效权重的L1总和为{target_l1}，与Manual MAPPO人工奖励尺度一致。
coordination_conflict_rate必须固定为0.0，不计入可调维度。正向项为sgl、relay、completion、balance；expiration、invalid、relay_cost由系统固定为惩罚。
所有raw权重必须在[0,3]且有限，七项可调权重不能全为0；不要自行把权重凑成{target_l1}。

上一轮best候选ID：{parent}
上一轮反馈JSON：{feedback}

只输出JSON，不得新增字段、特征、代码、函数、import、路径或Markdown。示例：
{example}
""".format(
        task_count=int(task_count), target_l1=float(l1_target_scale),
        round_index=int(round_index), candidate_index=int(candidate_index),
        candidate_total=int(candidates_per_round), candidate_id=candidate_id,
        features=_FEATURES, parent=parent_candidate_id or "null",
        feedback=json.dumps(parent_summary or {"status": "initial"}, ensure_ascii=False, sort_keys=True),
        example=json.dumps(example, ensure_ascii=False),
    )


def build_initial_reward_prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id):
    return _prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, None, None)


def build_feedback_reward_prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, parent_candidate_id, parent_summary):
    return _prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, parent_candidate_id, parent_summary)
