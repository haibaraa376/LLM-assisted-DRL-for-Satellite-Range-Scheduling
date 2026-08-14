"""构造无RAG、无代码的直接LLM奖励权重提示。"""

import json


_FEATURES = """固定RewardFeatures：sgl_progress、relay_progress、completion_score、
balance_score、expiration_loss、invalid_action_rate、coordination_conflict_rate、relay_cost。"""


def system_prompt():
    return "你是卫星调度奖励设计审查员。只能输出单个合法JSON对象，不得输出代码、路径或Markdown。"


def _prompt(task_count, feedback, parent_candidate_id):
    example = {
        "schema_version": "2.0",
        "reward_name": "deadline_delivery_v2",
        "positive_weights": {"sgl_progress": 1.2, "relay_progress": 0.1, "completion_score": 0.8, "balance_score": 0.03},
        "penalty_weights": {"expiration_loss": 0.9, "invalid_action_rate": 0.08, "coordination_conflict_rate": 0.0, "relay_cost": 0.04},
        "rationale": "优先最终下传与完成，并抑制过期和无效动作。",
        "expected_behavior": ["提高完成率"],
        "risk_notes": [],
        "parent_candidate_id": parent_candidate_id,
    }
    return """请为无RAG的DeepSeek LLM-PPO生成最终奖励权重JSON。

训练任务数为{task_count}。LLM生成的奖励直接作为MAPPO训练总奖励，不存在R_base、alpha或附加塑形。
{features}
七项可调权重将做L1归一化；coordination_conflict_rate必须固定为0.0，不能参与搜索。
正向项为sgl、relay、completion、balance；expiration、invalid、relay_cost由系统固定为惩罚，LLM不得改变符号。
所有权重必须在[0,3]且有限，七项可调权重不能全为0。环境冲突动作已由环境拒绝。

上一轮反馈JSON：{feedback}
父候选ID：{parent}

只输出JSON，不得新增字段、特征、代码、函数、import、路径或Markdown。示例：
{example}
""".format(
        task_count=int(task_count),
        features=_FEATURES,
        feedback=json.dumps(feedback or {"status": "initial"}, ensure_ascii=False, sort_keys=True),
        parent=parent_candidate_id or "null",
        example=json.dumps(example, ensure_ascii=False),
    )


def build_initial_reward_prompt(task_count, training_summary=None):
    return _prompt(task_count, training_summary, None)


def build_feedback_reward_prompt(task_count, parent_candidate_id, parent_summary):
    """把上一轮最优候选的权重、末二轮指标和贡献反馈给下一轮。"""
    return _prompt(task_count, parent_summary, parent_candidate_id)
