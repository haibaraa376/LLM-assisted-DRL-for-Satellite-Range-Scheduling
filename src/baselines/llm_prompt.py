"""构造不含密钥、源码、逐步轨迹和本地路径的奖励设计提示。"""

import json


_FEATURE_DESCRIPTION = """八个固定RewardFeatures：
- sgl_progress：[0,+∞)，最终下传进展，正向；
- relay_progress：[0,+∞)，ISL/IDL中继进展，弱正向；
- completion_score：[0,1]，首次完成事件，正向；
- balance_score：[-1,1]，欠载发送为正、过载发送为负；
- expiration_loss：[0,1]，首次过期未送达比例，负向；
- invalid_action_rate：[0,1]，明显无效动作率，负向；
- coordination_conflict_rate：[0,1]，并发协调冲突率，负向；
- relay_cost：[0,+∞)，中继成本，负向。"""

_EXAMPLE = {
    "schema_version": "1.0",
    "reward_name": "deadline_delivery_v1",
    "positive_weights": {
        "sgl_progress": 1.2,
        "relay_progress": 0.1,
        "completion_score": 0.8,
        "balance_score": 0.03,
    },
    "penalty_weights": {
        "expiration_loss": 0.9,
        "invalid_action_rate": 0.08,
        "coordination_conflict_rate": 0.05,
        "relay_cost": 0.04,
    },
    "rationale": "奖励最终下传与任务完成，抑制过期、冲突和循环中继。",
    "expected_behavior": ["提高SGL送达", "降低任务过期", "减少重复竞争"],
    "risk_notes": ["过高过期惩罚可能导致策略保守"],
    "parent_candidate_id": None,
}


def system_prompt():
    """返回固定系统约束，要求只输出JSON奖励权重。"""
    return (
        "你是卫星调度奖励设计审查员。只能设计八项固定特征的非负权重，"
        "不得新增特征、输出代码、路径或Markdown。最终回答必须是单个JSON对象。"
    )


def _base_prompt(current_weights, training_summary, parent_candidate_id):
    """组合固定环境说明、特征、反馈和完整JSON示例。"""
    safe_summary = training_summary or {
        "status": "initial",
        "timeliness_raw_mean": None,
        "load_balance_mean_per_task_mean": None,
    }
    return """请为无RAG的DeepSeek-LLM-PPO生成奖励权重JSON。

环境：15颗卫星、4个地面站、24小时、30秒决策步长。动态任务具有优先级、
生存时间和过期状态；链路包括ISL、IDL、SGL。每颗卫星每步采用最多3个星间
链路加1个SGL的复合动作。算法为共享Actor、集中式Critic的多智能体MAPPO，
所有智能体共享团队奖励。

{features}

当前人工奖励权重JSON：
{weights}

当前训练与validation摘要JSON：
{summary}

父候选ID：{parent}

严格要求：只输出JSON；八项权重必须齐全且在[0,3]；不得改变四项正奖励和
四项惩罚的符号；不得新增字段、代码、函数、import、路径或Markdown代码块。
完整JSON格式示例：
{example}
""".format(
        features=_FEATURE_DESCRIPTION,
        weights=json.dumps(current_weights, ensure_ascii=False, sort_keys=True),
        summary=json.dumps(safe_summary, ensure_ascii=False, sort_keys=True),
        parent=parent_candidate_id or "null",
        example=json.dumps(_EXAMPLE, ensure_ascii=False, indent=2),
    )


def build_initial_reward_prompt(current_weights, training_summary=None):
    """构造第一轮候选Prompt。"""
    return _base_prompt(current_weights, training_summary, None)


def build_feedback_reward_prompt(
    current_weights,
    parent_candidate_id,
    parent_training_summary,
    validation_summary,
):
    """构造包含父候选表现和调权原因要求的后续轮Prompt。"""
    feedback = {
        "parent_training": parent_training_summary,
        "validation": validation_summary,
        "phenomena_to_review": [
            "协调冲突",
            "任务过期",
            "中继成本",
            "最终送达",
        ],
        "instruction": "在rationale中说明相对父候选的权重修改原因",
    }
    return _base_prompt(current_weights, feedback, parent_candidate_id)
