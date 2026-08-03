"""提取已安全整理并登记的PDF；技术失败文献不会进入索引。"""
import json
import csv
from pathlib import Path
import yaml
from knowledge_base.pdf_extractor import clean_pages, default_quality_config, extract_pdf, chunk_text
from knowledge_base.schemas import DocumentRecord, sha256_text
from knowledge_base.source_catalog import sha256_file

def main():
    config = yaml.safe_load(Path('configs/rappo.yaml').read_text(encoding='utf-8'))['rappo']
    register = Path(config['knowledge_base']['source_register'])
    if not register.exists(): raise FileNotFoundError('缺少source_register；请先完成唯一匹配和Apply')
    documents, chunks = [], []
    updated_sources, review_rows = [], []
    for line in register.read_text(encoding='utf-8').splitlines():
        source = json.loads(line)
        if source['technical_extraction_status'] == 'needs_manual_review':
            updated_sources.append(source); continue
        pages, report = extract_pdf(Path('knowledge_sources') / source['canonical_relative_path'], source['document_id'], default_quality_config())
        directory = Path('knowledge_sources/processed/documents') / source['document_id']; directory.mkdir(parents=True, exist_ok=True)
        (directory/'pages.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in pages)+'\n',encoding='utf-8')
        cleaned, metadata = clean_pages(pages, config.get('cleaning')); report.update(metadata)
        (directory/'cleaned.txt').write_text(cleaned,encoding='utf-8'); (directory/'extraction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        status = report['technical_extraction_status']
        excluded_ids = set(config['knowledge_base']['excluded_document_ids'])
        if source['document_id'] == 'RS-01':
            source.update({'document_kind':'presentation_slides','include_in_knowledge_base':False,'exclusion_reason':'secondary_slide_deck_not_original_paper'})
        elif source['document_id'] == 'RS-06':
            source.update({'document_kind':'implementation_reference','include_in_knowledge_base':False,'exclusion_reason':'implementation_reference_not_reward_domain_knowledge'})
        else:
            source.setdefault('document_kind','research_paper')
            source.setdefault('exclusion_reason','')
        source['technical_extraction_status'] = status
        source['page_count'] = report['page_count']
        updated_sources.append(source)
        review_rows.append({"document_id": source['document_id'], "canonical_filename": source['canonical_filename'], "extraction_status": status, "needs_manual_review": status == 'needs_manual_review', "review_reason": report['review_reason'], "cleaned_text_preview": cleaned[:300].replace('\n', ' '), "researcher_final_approval": "", "review_notes": ""})
        record = DocumentRecord('1.0', source['document_id'], source['title'], source['authors'], source['year'], source['venue'], source['doi'], source['category'], source['priority'], source['sha256'], source['canonical_filename'], True, status, source['include_in_knowledge_base'], sha256_text(cleaned), report['page_count'], len(cleaned))
        documents.append(record)
        if status == 'approved_for_index' and source['include_in_knowledge_base'] and source['document_id'] not in excluded_ids:
            chunks.extend(chunk_text(record, cleaned, pages, config['chunking']))
    processed = Path('knowledge_sources/processed'); processed.mkdir(parents=True,exist_ok=True)
    (processed/'documents.jsonl').write_text('\n'.join(x.to_json() for x in documents)+'\n',encoding='utf-8')
    (processed/'chunks.jsonl').write_text('\n'.join(x.to_json() for x in chunks)+'\n',encoding='utf-8')
    register.write_text('\n'.join(json.dumps(item, ensure_ascii=False) for item in updated_sources)+'\n', encoding='utf-8')
    review = Path('knowledge_sources/review'); review.mkdir(parents=True, exist_ok=True)
    with (review/'document_review.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=["document_id","canonical_filename","extraction_status","needs_manual_review","review_reason","cleaned_text_preview","researcher_final_approval","review_notes"]); writer.writeheader(); writer.writerows(review_rows)

if __name__ == '__main__': main()
