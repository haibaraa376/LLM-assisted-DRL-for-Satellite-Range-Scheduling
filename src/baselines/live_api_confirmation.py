"""集中实现不可绕过的真实DeepSeek API交互确认。"""

from dataclasses import dataclass
from datetime import datetime, timezone
import sys
from typing import Optional


@dataclass(frozen=True)
class LiveApiPlan:
    """展示给用户确认的真实API与候选训练预算。"""

    model: str
    rounds: int
    candidates_per_round: int
    candidate_training_episodes: int
    maximum_api_calls: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    output_directory: str


@dataclass(frozen=True)
class LiveApiApproval:
    """只能由精确YES交互确认后产生的运行时批准对象。"""

    approved: bool
    approved_at: str


def confirm_live_api_call(
    plan,
    input_stream=None,
    output_stream=None,
):
    """仅交互终端输入精确YES时返回批准对象。"""
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    if not getattr(input_stream, "isatty", lambda: False)():
        raise RuntimeError("非交互stdin拒绝真实API调用")
    lines = (
        "即将调用真实DeepSeek API",
        "模型：{0}".format(plan.model),
        "搜索轮数：{0}".format(plan.rounds),
        "每轮候选数：{0}".format(plan.candidates_per_round),
        "候选短训练Episode：{0}".format(plan.candidate_training_episodes),
        "最大API调用数：{0}".format(plan.maximum_api_calls),
        "最大输入Token：{0}".format(plan.maximum_input_tokens),
        "最大输出Token：{0}".format(plan.maximum_output_tokens),
        "结果目录：{0}".format(plan.output_directory),
    )
    output_stream.write("\n".join(lines) + "\n")
    output_stream.write("请输入 YES 继续，输入其他内容取消：")
    output_stream.flush()
    answer = input_stream.readline().rstrip("\r\n")
    if answer != "YES":
        output_stream.write("已取消真实API调用\n")
        output_stream.flush()
        raise RuntimeError("已取消真实API调用")
    return LiveApiApproval(
        approved=True,
        approved_at=datetime.now(timezone.utc).isoformat(),
    )
