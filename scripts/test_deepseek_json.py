"""验证DeepSeek JSON Output是否可用于奖励权重生成。"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI


def main() -> None:
    """调用DeepSeek并验证返回内容能够解析为JSON对象。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有读取到DEEPSEEK_API_KEY。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1,
    )

    system_prompt = """
你是强化学习奖励设计助手。
你必须只输出一个JSON对象，不要输出Markdown代码块或额外解释。

JSON格式示例：
{
  "reward_name": "delivery_reward_v1",
  "sgl_progress_weight": 1.0,
  "completion_weight": 0.5
}
""".strip()

    user_prompt = """
请生成一个简单的卫星调度奖励权重JSON。

要求：
1. reward_name必须是字符串；
2. sgl_progress_weight必须是0到3之间的数字；
3. completion_weight必须是0到3之间的数字；
4. 只输出JSON。
""".strip()

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={
                "type": "json_object",
            },
            stream=False,
            max_tokens=300,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
    except Exception as exc:
        print(f"API调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("JSON模式返回了空内容，请重新运行一次。")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print("模型返回内容不是合法JSON：")
        print(content)
        raise RuntimeError("JSON解析失败。") from exc

    required_fields = {
        "reward_name",
        "sgl_progress_weight",
        "completion_weight",
    }
    missing_fields = required_fields - parsed.keys()
    if missing_fields:
        raise RuntimeError(f"缺少字段：{sorted(missing_fields)}")

    print("JSON Output调用成功")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if response.usage is not None:
        print(f"总Token：{response.usage.total_tokens}")


if __name__ == "__main__":
    main()