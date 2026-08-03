"""全库余弦Top-k检索与本地审计。"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import re
from .index import load_index

class KnowledgeRetriever:
    def __init__(self, directory, chunks, embedder):
        self.vectors, ids, self.manifest = load_index(directory)
        self.chunks = {item["chunk_id"]: item for item in chunks}
        self.ids = ids
        self.embedder = embedder
        self.mean = np.load(Path(directory) / "corpus_mean.npy")
    def retrieve(self, query, top_k=5):
        if not str(query).strip(): raise ValueError("查询不得为空")
        raw = self.embedder.encode([query], normalize=False)[0]
        vector = raw - self.mean
        vector = vector / np.linalg.norm(vector)
        dense = self.vectors @ vector
        query_tokens = set(re.findall(r"[a-z]{3,}", query.lower()))
        keyword = np.asarray([
            len(query_tokens & set(re.findall(r"[a-z]{3,}", self.chunks[item]["text"].lower()))) / max(len(query_tokens), 1)
            for item in self.ids
        ])
        scores = 0.85 * dense + 0.15 * keyword
        # 先确保候选池含有查询词项，再在池内按组合分数和MMR选择。
        candidates = sorted(
            range(len(self.ids)),
            key=lambda i: (-float(keyword[i]), -float(scores[i]), self.ids[i]),
        )[:30]
        order = []
        seen_documents = {}
        while candidates and len(order) < top_k:
            def mmr(index):
                duplicate = max((float(self.vectors[index] @ self.vectors[chosen]) for chosen in order), default=0.0)
                return 0.85 * float(scores[index]) - 0.15 * duplicate
            index = max(candidates, key=lambda i: (mmr(i), self.ids[i]))
            candidates.remove(index)
            document = self.chunks[self.ids[index]]["document_id"]
            if seen_documents.get(document, 0) >= 2:
                continue
            order.append(index); seen_documents[document] = seen_documents.get(document, 0) + 1
        return {"knowledge_base_version": self.manifest["knowledge_base_version"], "results": [
            {"rank": rank, "score": float(scores[i]), **self.chunks[self.ids[i]]}
            for rank, i in enumerate(order, 1)]}
    def audit(self, path, query, result, top_k):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "knowledge_base_version": result["knowledge_base_version"], "top_k": top_k, "retrieved_chunk_ids": [x["chunk_id"] for x in result["results"]], "scores": [x["score"] for x in result["results"]], "timestamp": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
