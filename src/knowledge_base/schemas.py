"""知识库持久化记录的严格数据结构。"""

from dataclasses import asdict, dataclass
import hashlib
import json


def sha256_text(text):
    """返回UTF-8文本的稳定SHA256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentRecord:
    schema_version: str
    document_id: str
    title: str
    authors: str
    year: str
    venue: str
    doi: str
    category: str
    priority: str
    source_sha256: str
    canonical_filename: str
    researcher_approved: bool
    technical_extraction_status: str
    include_in_knowledge_base: bool
    cleaned_text_sha256: str
    page_count: int
    character_count: int

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ChunkRecord:
    schema_version: str
    chunk_id: str
    document_id: str
    category: str
    title: str
    year: str
    text: str
    text_sha256: str
    chunk_index: int
    start_offset: int
    end_offset: int
    page_start: int
    page_end: int
    character_count: int
    token_estimate: int

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)
