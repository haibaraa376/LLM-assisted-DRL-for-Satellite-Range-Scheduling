"""本地NumPy向量索引及完整性manifest。"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .source_catalog import sha256_file


def atomic_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def build_index(chunks, embedder, config, source_register, documents_path, chunks_path):
    """构建向量矩阵、稳定ID表及由输入决定的版本manifest。"""
    if not chunks:
        raise ValueError("没有可入库知识块")
    raw_vectors = embedder.encode([chunk["text"] for chunk in chunks], normalize=False)
    corpus_mean = raw_vectors.mean(axis=0).astype(np.float32)
    vectors = raw_vectors - corpus_mean
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    if vectors.shape[0] != len(chunks):
        raise RuntimeError("嵌入行数与chunk数量不一致")
    directory = Path(config["knowledge_base"]["index_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    embeddings_path = directory / "embeddings.npy"
    mean_path = directory / "corpus_mean.npy"
    ids_path = directory / "chunk_ids.json"
    np.save(embeddings_path, vectors.astype(np.float32))
    np.save(mean_path, corpus_mean)
    atomic_json(ids_path, [chunk["chunk_id"] for chunk in chunks])
    payload = {
        "schema_version": "2.0", "document_count": len({chunk["document_id"] for chunk in chunks}),
        "chunk_count": len(chunks), "embedding_dimension": int(vectors.shape[1]),
        "embedding_model_name": embedder.model_name, "embedding_model_revision": embedder.revision,
        "tokenizer_name": embedder.tokenizer.name_or_path, "pooling": config["embedding"]["pooling"],
        "normalize_l2": True, "corpus_mean_centering": True, "similarity": "cosine", "chunking_config": config["chunking"],
        "keyword_score_enabled": True, "combined_score_weights": {"dense":config["retrieval"]["dense_weight"],"keyword":config["retrieval"]["keyword_weight"],"quality":config["retrieval"]["quality_weight"]},
        "mmr_enabled": True, "mmr_lambda": config["retrieval"]["mmr_lambda"], "candidate_pool_size":config["retrieval"]["candidate_pool_size"], "max_chunks_per_document":config["retrieval"]["max_chunks_per_document"],
        "source_register_sha256": sha256_file(source_register),
        "documents_jsonl_sha256": sha256_file(documents_path), "chunks_jsonl_sha256": sha256_file(chunks_path),
        "embeddings_sha256": sha256_file(embeddings_path), "chunk_ids_sha256": sha256_file(ids_path),
        "corpus_mean_sha256": sha256_file(mean_path),
        "included_document_ids": sorted({chunk["document_id"] for chunk in chunks}),
        "excluded_document_ids": ["RS-01", "RS-06"], "missing_optional_document_ids": [],
    }
    version_source = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["knowledge_base_version"] = "kb_" + hashlib.sha256(version_source.encode("utf-8")).hexdigest()[:12]
    atomic_json(directory / "index_manifest.json", payload)
    return payload


def load_index(directory):
    """读取索引前先验证二进制和ID表哈希，拒绝陈旧或篡改索引。"""
    directory = Path(directory)
    manifest = json.loads((directory / "index_manifest.json").read_text(encoding="utf-8"))
    for name, file_name in (("embeddings_sha256", "embeddings.npy"), ("chunk_ids_sha256", "chunk_ids.json")):
        if manifest[name] != sha256_file(directory / file_name):
            raise ValueError("索引哈希不匹配：{0}".format(file_name))
    vectors = np.load(directory / "embeddings.npy")
    ids = json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("索引向量与ID表不一致或包含非有限值")
    return vectors, ids, manifest
