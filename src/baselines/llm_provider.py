"""实现默认Mock和显式批准的DeepSeek Chat Completions Provider。"""

from dataclasses import dataclass
import json
import os
from typing import Optional

from .live_api_confirmation import LiveApiApproval


class RetryableProviderError(RuntimeError):
    """表示可按有限重试策略再次请求的错误。"""


class FatalProviderError(RuntimeError):
    """表示认证、模型或安全配置错误，不得重试。"""


@dataclass(frozen=True)
class ProviderResult:
    """只保留最终内容和非敏感审计元数据。"""

    content: str
    model: str
    response_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


class MockRewardGenerationProvider:
    """可注入合法、空、非法或异常响应，绝不访问网络。"""

    name = "mock"

    def __init__(self, responses, model="mock-reward-generator"):
        self.responses = list(responses)
        self.model = model
        self.call_count = 0

    def generate_reward_spec(self, prompt, metadata):
        del prompt, metadata
        self.call_count += 1
        if not self.responses:
            raise FatalProviderError("Mock响应已耗尽")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        return ProviderResult(
            content=str(response),
            model=self.model,
            response_id="mock-{0:04d}".format(self.call_count),
            input_tokens=100,
            output_tokens=100,
        )


class DeepSeekRewardGenerationProvider:
    """通过OpenAI兼容Chat Completions调用DeepSeek，默认不可启用。"""

    name = "deepseek"

    def __init__(self, config, approval=None, client=None):
        if not isinstance(approval, LiveApiApproval) or not approval.approved:
            raise FatalProviderError("真实DeepSeek调用缺少交互批准对象")
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise FatalProviderError("未设置DEEPSEEK_API_KEY。")
        self.config = config
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=config["base_url"],
                timeout=config["timeout_seconds"],
                max_retries=0,
            )
        self.client = client

    def generate_reward_spec(self, prompt, metadata):
        """发送一次请求；重试、缓存和预算由上层统一控制。"""
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": metadata["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=self.config["reasoning_effort"],
                extra_body={
                    "thinking": {
                        "type": (
                            "enabled"
                            if self.config["thinking_enabled"]
                            else "disabled"
                        )
                    }
                },
                max_tokens=self.config["max_tokens"],
                stream=False,
            )
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status in {401, 402, 404}:
                raise FatalProviderError(
                    "DeepSeek认证、余额或模型配置错误"
                ) from error
            if status in {429, 500, 502, 503} or status is None:
                raise RetryableProviderError("DeepSeek请求暂时失败") from error
            raise FatalProviderError("DeepSeek请求不可重试地失败") from error
        content = response.choices[0].message.content
        if not content:
            raise RetryableProviderError("DeepSeek返回空content")
        usage = getattr(response, "usage", None)
        return ProviderResult(
            content=content,
            model=self.config["model"],
            response_id=getattr(response, "id", None),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
