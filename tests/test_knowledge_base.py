"""知识库安全逻辑的离线单元测试。"""

import pytest

from knowledge_base.title_matching import match_document, normalize_title


def test_title_normalization_preserves_numbers_and_removes_download_suffixes():
    assert normalize_title("CodeBERT-2020_final (1).pdf") == "codebert 2020"


def test_ambiguous_fuzzy_match_requires_manual_resolution():
    records = [
        {"文献ID": "A", "题目": "satellite scheduling with reinforcement learning"},
        {"文献ID": "B", "题目": "satellite scheduling using reinforcement learning"},
    ]
    matched, method, _, _, _ = match_document("x.pdf", records, ["satellite scheduling reinforcement learning"])
    assert matched is None
    assert method == "needs_manual_resolution"
