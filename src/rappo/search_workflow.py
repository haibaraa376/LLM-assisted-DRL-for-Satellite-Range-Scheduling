"""RAPPO仅负责Prompt前的检索；候选训练复用基线公共实现。"""

from pathlib import Path
import json

from baselines.llm_prompt import build_initial_reward_prompt
from knowledge_base.embedder import CodeBertEmbedder
from knowledge_base.retriever import KnowledgeRetriever
from .prompt_builder import build_rappo_prompt


def build_retrieved_reward_prompt(rappo_config, manual_weights, query, audit_path):
    """构建一次RAPPO候选Prompt，并留下不含路径/密钥的检索审计。"""
    chunks = [
        json.loads(line)
        for line in Path(rappo_config["knowledge_base"]["chunks"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    retriever = KnowledgeRetriever(
        rappo_config["knowledge_base"]["index_directory"],
        chunks,
        CodeBertEmbedder(rappo_config["embedding"]),
    )
    retrieval = retriever.retrieve(query, rappo_config["retrieval"]["top_k"])
    retriever.audit(audit_path, query, retrieval, rappo_config["retrieval"]["top_k"])
    base = build_initial_reward_prompt(manual_weights, {"status": "rappo_initial"})
    return build_rappo_prompt(base, retrieval), retrieval
