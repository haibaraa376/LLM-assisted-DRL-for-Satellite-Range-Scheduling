"""查询本地知识库Top-k，不向控制台打印全文。"""
import argparse,json
from pathlib import Path
import yaml
from knowledge_base.embedder import CodeBertEmbedder
from knowledge_base.retriever import KnowledgeRetriever
def main():
 p=argparse.ArgumentParser();p.add_argument('--query',required=True);p.add_argument('--top-k',type=int,default=5);a=p.parse_args()
 c=yaml.safe_load(Path('configs/rappo.yaml').read_text(encoding='utf-8'))['rappo']; chunks=[json.loads(x) for x in Path(c['knowledge_base']['chunks']).read_text(encoding='utf-8').splitlines() if x]
 r=KnowledgeRetriever(c['knowledge_base']['index_directory'],chunks,CodeBertEmbedder(c['embedding']));out=r.retrieve(a.query,a.top_k);r.audit('knowledge_sources/reports/retrieval_audit.jsonl',a.query,out,a.top_k)
 print(json.dumps({"knowledge_base_version":out['knowledge_base_version'],"results":[{k:x[k] for k in ('rank','chunk_id','document_id','title','category','score','page_start','page_end')} for x in out['results']]},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
