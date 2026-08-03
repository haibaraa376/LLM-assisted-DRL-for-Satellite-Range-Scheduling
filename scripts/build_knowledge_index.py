"""用CodeBERT构建本地NumPy索引；只读取已技术通过的知识块。"""
import json
from pathlib import Path
import yaml
from knowledge_base.embedder import CodeBertEmbedder
from knowledge_base.index import build_index

def main():
    config=yaml.safe_load(Path('configs/rappo.yaml').read_text(encoding='utf-8'))['rappo']
    chunks=[json.loads(line) for line in Path(config['knowledge_base']['chunks']).read_text(encoding='utf-8').splitlines() if line]
    allowed=set(config['knowledge_base']['include_categories'])-set(config['knowledge_base']['exclude_categories'])
    chunks=[item for item in chunks if item['category'] in allowed]
    manifest=build_index(chunks,CodeBertEmbedder(config['embedding']),config,config['knowledge_base']['source_register'],config['knowledge_base']['documents'],config['knowledge_base']['chunks'])
    print(json.dumps({"knowledge_base_version":manifest['knowledge_base_version'],"chunk_count":manifest['chunk_count'],"embedding_dimension":manifest['embedding_dimension']},ensure_ascii=False))
if __name__=='__main__': main()
