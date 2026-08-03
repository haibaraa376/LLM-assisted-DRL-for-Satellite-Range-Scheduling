"""PDF确定性提取、质量检查、清洗与字符切块。"""

import json
import re
import unicodedata

import fitz

from .schemas import ChunkRecord, DocumentRecord, sha256_text


def default_quality_config():
    return {
        "minimum_total_characters": 500,
        "minimum_nonempty_page_ratio": 0.5,
        "maximum_replacement_character_ratio": 0.02,
        "maximum_very_short_page_ratio": 0.5,
    }


def extract_pdf(path, document_id, quality_config):
    """提取每页文本和质量数据；不使用OCR或改写PDF内容。"""
    document = fitz.open(path)
    pages = []
    for number, page in enumerate(document, start=1):
        raw = page.get_text("text") or ""
        pages.append({
            "document_id": document_id,
            "page_number": number,
            "raw_text": raw,
            "character_count": len(raw),
            "replacement_character_count": raw.count("�"),
        })
    page_count = len(pages)
    document.close()
    total = sum(item["character_count"] for item in pages)
    nonempty = sum(bool(item["raw_text"].strip()) for item in pages)
    replacement = sum(item["replacement_character_count"] for item in pages)
    controls = sum(
        sum(ord(char) < 32 and char not in "\n\t\r" for char in item["raw_text"])
        for item in pages
    )
    short = sum(item["character_count"] < 50 for item in pages)
    report = {
        "page_count": page_count,
        "nonempty_page_count": nonempty,
        "total_character_count": total,
        "mean_characters_per_page": total / max(page_count, 1),
        "replacement_character_ratio": replacement / max(total, 1),
        "control_character_ratio": controls / max(total, 1),
        "repeated_line_ratio": 0.0,
        "very_short_page_ratio": short / max(page_count, 1),
    }
    reasons = []
    if total < quality_config["minimum_total_characters"]:
        reasons.append("正文总字符过少")
    if nonempty / max(page_count, 1) < quality_config["minimum_nonempty_page_ratio"]:
        reasons.append("大部分页面无可提取正文")
    if report["replacement_character_ratio"] > quality_config["maximum_replacement_character_ratio"]:
        reasons.append("替换字符比例异常")
    if report["very_short_page_ratio"] > quality_config["maximum_very_short_page_ratio"]:
        reasons.append("过多页面正文过短")
    report["technical_extraction_status"] = "approved_for_index" if not reasons else "needs_manual_review"
    report["review_reason"] = "；".join(reasons)
    return pages, report


_INVALID_SECTION = re.compile(
    r"(?im)^\s*(acknowledg(?:ment)?s?|funding(?: statement| information)?|"
    r"declarations?|declaration of competing interest|conflict of interest|"
    r"competing interests?|credit authorship contribution statement|author contributions?|"
    r"data availability|availability of data and materials|ethics statement|publisher'?s note|"
    r"copyright notice|corresponding author|author biograph(?:y|ies)|致谢|基金项目|资金支持|"
    r"作者贡献|利益冲突|竞争性利益声明|数据可用性|伦理声明|作者简介)\s*$"
)
_REFERENCES = re.compile(r"(?im)^\s*(references|bibliography|参考文献)\s*$")
_APPENDIX = re.compile(r"(?im)^\s*(appendix|appendices|supplementary (?:material|information)|additional experiments|prompt examples|reward examples|appendix [a-z]|appendix\s+[a-z]|附录)\b.*$")


def clean_pages(pages, cleaning=None):
    """仅进行可解释的格式清洗，并默认移除参考文献正文。"""
    cleaning = cleaning or {}
    raw = "\n".join(item["raw_text"] for item in pages)
    text = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.!?])\n(?=[a-z])", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    removed = []
    invalid = list(_INVALID_SECTION.finditer(text))
    if invalid:
        first = invalid[0]
        text = text[:first.start()].rstrip()
        removed.append(first.group(0).strip())
    match = _REFERENCES.search(text)
    metadata = {"references_removed": False, "references_start_offset": None,
                "appendix_detected": False, "appendix_retained": False,
                "appendix_character_count": 0, "removed_sections": removed}
    if match:
        appendix = _APPENDIX.search(text, match.end())
        metadata["references_removed"] = True
        metadata["references_start_offset"] = match.start()
        if appendix and cleaning.get("retain_appendices_after_references", True):
            retained = text[appendix.start():].strip()
            text = text[:match.start()].rstrip() + "\n\n" + retained
            metadata.update({"appendix_detected": True, "appendix_retained": True,
                             "appendix_character_count": len(retained)})
        else:
            text = text[:match.start()].rstrip()
    return text, metadata


def chunk_text(document, text, pages, chunking):
    """按字符窗口切分，偏好句界且保留真实全文偏移和页码。"""
    size = int(chunking["chunk_size"])
    overlap = int(chunking["chunk_overlap"])
    minimum = int(chunking["minimum_chunk_size"])
    if not 0 <= overlap < size:
        raise ValueError("chunk_overlap必须小于chunk_size")
    page_offsets = []
    cursor = 0
    for page in pages:
        end = cursor + len(page["raw_text"])
        page_offsets.append((cursor, end, page["page_number"]))
        cursor = end + 1
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text) and chunking["prefer_sentence_boundary"]:
            boundary = max(text.rfind(mark, start + minimum, end) for mark in (".", "?", "!", "\n"))
            if boundary >= start + minimum:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece and (len(piece) >= minimum or end == len(text)):
            if len(piece) < minimum and chunks:
                previous = chunks.pop()
                piece = previous["text"] + "\n" + piece
                start = previous["start_offset"]
            page_start = next((page for left, right, page in page_offsets if right > start), 1)
            page_end = next((page for left, right, page in reversed(page_offsets) if left < end), page_start)
            index = len(chunks)
            text_hash = sha256_text(piece)
            chunk_id = "{0}:{1}:{2}".format(document.document_id, index, text_hash[:12])
            chunks.append({"chunk_id": chunk_id, "text": piece, "start_offset": start, "end_offset": end, "page_start": page_start, "page_end": page_end})
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [ChunkRecord(
        schema_version="1.0", chunk_id=item["chunk_id"], document_id=document.document_id,
        category=document.category, title=document.title, year=document.year, text=item["text"],
        text_sha256=sha256_text(item["text"]), chunk_index=index,
        start_offset=item["start_offset"], end_offset=item["end_offset"],
        page_start=item["page_start"], page_end=item["page_end"],
        character_count=len(item["text"]), token_estimate=max(1, len(item["text"]) // 4),
    ) for index, item in enumerate(chunks)]
