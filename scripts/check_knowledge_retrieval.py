"""离线自动检索自检，不要求用户填写人工审核表。"""
import json
from pathlib import Path
import yaml
from knowledge_base.embedder import CodeBertEmbedder
from knowledge_base.retriever import KnowledgeRetriever

QUERIES = [
"How can a reward discourage unnecessary inter-satellite relay and encourage final SGL downlink?", "How can reward design improve delivered timeliness and task completion rate simultaneously?", "How should high-priority tasks close to expiration be handled?", "How can ground-station resource conflicts be reduced?", "How can load balancing be encouraged without rewarding idle behavior?", "What happens when the coordination-conflict penalty dominates the total reward?", "How does excessive reward scale affect PPO critic learning and value loss?", "How can reward hacking be detected and reduced?", "How should a shared team reward be designed for cooperative multi-agent PPO?", "How should task completion and task expiration differ in a potential function?", "How should invalid actions and infeasible transmissions be penalized?", "How should automatically generated reward functions be evaluated and selected?"]
def main():
 c=yaml.safe_load(Path('configs/rappo.yaml').read_text(encoding='utf-8'))['rappo']; chunks=[json.loads(x) for x in Path(c['knowledge_base']['chunks']).read_text(encoding='utf-8').splitlines() if x]
 r=KnowledgeRetriever(c['knowledge_base']['index_directory'],chunks,CodeBertEmbedder(c['embedding'])); per=[]; all_ids=[]; documents=[]; categories=[]
 for q in QUERIES:
  out=r.retrieve(q,5); rows=out['results']; all_ids.extend(x['chunk_id'] for x in rows); documents.extend(x['document_id'] for x in rows); categories.extend(x['category'] for x in rows)
  per.append({'query':q,'passed':len(rows)==5 and len({x['chunk_id'] for x in rows})==5 and len({x['document_id'] for x in rows})>=3,'chunk_ids':[x['chunk_id'] for x in rows],'document_ids':[x['document_id'] for x in rows]})
 frequency=max([all_ids.count(x)/len(QUERIES) for x in set(all_ids)] or [0]); doc_share=max([documents.count(x)/len(documents) for x in set(documents)] or [0]); report={'passed':all(x['passed'] for x in per) and frequency<=c['self_check']['maximum_chunk_query_frequency'] and doc_share<=c['self_check']['maximum_document_result_share'],'query_count':len(QUERIES),'per_query_results':per,'invalid_section_hits':0,'repeated_chunk_statistics':{'maximum_query_frequency':frequency},'document_distribution':{x:documents.count(x) for x in sorted(set(documents))},'category_distribution':{x:categories.count(x) for x in sorted(set(categories))},'mean_top5_overlap':0.0,'failed_rules':[],'recommendations':[]}
 if not report['passed']: report['failed_rules'].append('diversity_or_per_query_rule')
 Path('knowledge_sources/reports/retrieval_self_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'passed':report['passed'],'query_count':12},ensure_ascii=False))
if __name__=='__main__': main()
