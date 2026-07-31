"""提供基线Run目录、原子JSON、训练锁和配置哈希工具。"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import sys
import tempfile


_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def generate_run_id():
    """生成时间戳和随机后缀组成的安全Run ID。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "{0}_{1}".format(timestamp, secrets.token_hex(3))


def validate_run_name(value):
    if not isinstance(value, str) or not _SAFE_RUN_NAME.fullmatch(value):
        raise ValueError("run-name必须是安全的字母、数字、下划线或连字符")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    """把JSON写入同目录临时文件并原子替换目标。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


class TrainingLock:
    """使用独占创建的锁文件避免相同Run被并发启动。"""

    def __init__(self, run_directory):
        self.run_directory = Path(run_directory)
        self.path = self.run_directory / ".training.lock"
        self.acquired = False

    def acquire(self):
        self.run_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": utc_now(),
            "command": " ".join(sys.argv),
        }
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError as error:
            existing = self.path.read_text(encoding="utf-8")
            raise RuntimeError(
                "Run存在训练锁，请先确认原进程已经结束并人工处理锁文件：\n{0}".format(
                    existing
                )
            ) from error
        self.acquired = True

    def release(self):
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
