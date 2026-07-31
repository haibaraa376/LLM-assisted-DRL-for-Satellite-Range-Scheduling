"""验证并原子注册已有LLM奖励规范，不调用任何Provider。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .llm_schema import LlmRewardSpec


def sha256_file(path):
    """流式计算文件SHA-256，避免读取无关内容。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path, content):
    """在目标目录写临时文件后原子替换单个目标。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def register_reward_spec(source, destination, minimum=0.0, maximum=3.0, force=False):
    """校验源规范，写入标准目标和来源哈希旁路元数据。"""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    metadata_path = destination_path.with_name(
        destination_path.stem + ".metadata.json"
    )
    if not source_path.is_file():
        raise FileNotFoundError("奖励规范源文件不存在")
    if source_path == destination_path:
        raise ValueError("源规范与注册目标不能是同一文件")
    if destination_path.exists() and not force:
        raise FileExistsError("目标奖励规范已存在；如需替换请显式使用--force")
    source_hash = sha256_file(source_path)
    spec = LlmRewardSpec.load(source_path, minimum, maximum)
    canonical = json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(destination_path, canonical)
    reloaded = LlmRewardSpec.load(destination_path, minimum, maximum)
    if reloaded.spec_id != spec.spec_id:
        raise RuntimeError("注册后的奖励规范复核失败")
    destination_hash = sha256_file(destination_path)
    metadata = {
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "reward_spec_id": spec.spec_id,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return metadata
