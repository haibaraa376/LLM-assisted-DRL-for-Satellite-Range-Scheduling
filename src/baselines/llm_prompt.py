"""构造无RAG、无代码且候选身份明确的直接LLM奖励提示。"""

import json


_TASK_CONTEXT = """当前任务为卫星数据调度：
- 15颗卫星、4个地面站；调度时域为24小时；每个Episode包含{task_count}个任务。
- 任务需要通过卫星间中继以及SGL最终下传至指定地面站。
- 核心目标是提高任务完成率和真实下传数据量，同时兼顾负载均衡，减少任务过期和无效动作。
- SGL：卫星到地面站的最终下传链路。ISL/IDL：卫星之间的数据中继链路。"""


_FEATURES = """RewardFeatures的实际语义如下，必须严格按此理解并仅为其分配权重：
- sgl_progress：成功SGL最终下传产生的进度奖励；综合考虑本次传输数据占任务总数据量的比例、任务优先级和剩余生存时间。
- relay_progress：成功ISL/IDL中继产生的有效数据推进奖励；同样考虑数据比例、优先级和剩余生存时间。
- completion_score：任务在当前slot首次完成时产生的优先级加权完成奖励。
- balance_score：鼓励通过当前相对欠载的卫星进行传输，用于改善卫星间负载均衡；取值可正可负。
- expiration_loss：任务在当前slot首次过期时产生的损失；综合考虑任务优先级和仍未下传的数据比例。
- invalid_action_rate：当前slot中明显无效动作所占比例，对无效调度进行惩罚。
- relay_cost：ISL/IDL中继传输的数据成本，用于抑制无必要的重复中继。
- coordination_conflict_rate：资源竞争/协调冲突比例；当前实验中该项权重永久固定为0，不允许LLM调整，因为环境已经负责拒绝冲突动作。"""


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

{task_context}

训练任务数：{task_count}。当前搜索轮次：{round_index}。当前候选：{candidate_id}。
这是本轮第{candidate_index}/{candidate_total}个独立候选；请生成新的独立方案，不要简单复制父候选权重。
R_total = R_llm。LLM生成的奖励直接作为MAPPO训练总奖励，不存在R_base、alpha或附加塑形。
{features}
LLM只决定其余7项raw weights，程序会保持比例并统一缩放，使最终有效权重L1总和 = {target_l1}，与Manual MAPPO人工奖励尺度一致。
coordination_conflict_rate = 0，必须固定为0.0，不计入可调维度。正向项：sgl_progress、relay_progress、completion_score、balance_score。惩罚项：expiration_loss、invalid_action_rate、relay_cost。
所有raw weights范围[0,3]且有限，七项可调权重不能全为0；不要自行把权重凑成{target_l1}。

上一轮best候选ID：{parent}
上一轮反馈JSON：{feedback}

只输出JSON，不得新增字段、特征、代码、函数、import、路径或Markdown。示例：
{example}
""".format(
        task_count=int(task_count), target_l1=float(l1_target_scale),
        round_index=int(round_index), candidate_index=int(candidate_index),
        candidate_total=int(candidates_per_round), candidate_id=candidate_id,
        task_context=_TASK_CONTEXT.format(task_count=int(task_count)),
        features=_FEATURES, parent=parent_candidate_id or "null",
        feedback=json.dumps(parent_summary or {"status": "initial"}, ensure_ascii=False, sort_keys=True),
        example=json.dumps(example, ensure_ascii=False),
    )


def build_initial_reward_prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id):
    return _prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, None, None)


def build_feedback_reward_prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, parent_candidate_id, parent_summary):
    return _prompt(task_count, l1_target_scale, round_index, candidate_index, candidates_per_round, candidate_id, parent_candidate_id, parent_summary)
