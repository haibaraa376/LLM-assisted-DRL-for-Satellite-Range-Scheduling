"""验证DeepSeek官方API、API Key和账户余额是否正常。"""

from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> None:
    """调用一次DeepSeek-V4-Flash并打印简短结果。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有读取到DEEPSEEK_API_KEY。"
            "请先在CMD中设置环境变量。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1,
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严格遵循指令的中文助手。",
                },
                {
                    "role": "user",
                    "content": "请只回复：DeepSeek API连接成功。",
                },
            ],
            stream=False,
            max_tokens=100,
            # 第一次连接测试关闭思考模式，减少延迟和输出开销。
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
    except Exception as exc:
        print("DeepSeek API调用失败。", file=sys.stderr)
        print(f"异常类型：{type(exc).__name__}", file=sys.stderr)
        print(f"异常信息：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("API返回成功，但回复内容为空。")

    print("API调用成功")
    print(f"模型回复：{content}")

    if response.usage is not None:
        print(f"输入Token：{response.usage.prompt_tokens}")
        print(f"输出Token：{response.usage.completion_tokens}")
        print(f"总Token：{response.usage.total_tokens}")


if __name__ == "__main__":
    main()