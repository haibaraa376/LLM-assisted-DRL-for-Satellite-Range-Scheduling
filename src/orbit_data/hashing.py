"""Stable SHA-256 helpers used by data and manifest exporters."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_object(value: Any) -> str:
    return sha256_bytes(stable_json_bytes(value))


def hash_files(paths: Iterable[Path]) -> Dict[str, str]:
    return {str(Path(path)): sha256_file(Path(path)) for path in paths}
