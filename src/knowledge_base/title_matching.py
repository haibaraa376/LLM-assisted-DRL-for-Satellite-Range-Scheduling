"""确定性标题标准化与保守的PDF-文献匹配。"""

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def normalize_title(value):
    """规范化标题或文件名，不删除数字、缩写等语义字符。"""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s*\((?:\d+)\)\s*$", "", text)
    text = re.sub(r"\s+\b(?:copy|final)\b\s*$", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def title_similarity(left, right):
    """使用标准库序列相似度，避免隐式的非确定性模型依赖。"""
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def match_document(path, records, candidate_titles, threshold=0.92, margin=0.08):
    """仅在精确或唯一高置信条件满足时返回自动匹配。"""
    candidates = []
    seen = set()
    for title in candidate_titles:
        normalized = normalize_title(title)
        if normalized and normalized not in seen:
            candidates.append(normalized)
            seen.add(normalized)
    scored = []
    for record in records:
        score = max(
            (title_similarity(title, record["题目"]) for title in candidates),
            default=0.0,
        )
        method = "exact_normalized" if any(
            title == normalize_title(record["题目"]) for title in candidates
        ) else "fuzzy"
        scored.append((score, method, record))
    scored.sort(key=lambda item: (-item[0], item[2]["文献ID"]))
    if not scored:
        return None, "needs_manual_resolution", 0.0, 0.0, "清单为空"
    best_score, method, record = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if method == "exact_normalized":
        return record, method, best_score, second, ""
    if best_score >= threshold and best_score - second >= margin:
        return record, method, best_score, second, ""
    return None, "needs_manual_resolution", best_score, second, "匹配不唯一或置信度不足"
