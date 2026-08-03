"""生成12个固定检索查询、审计记录和待人工填写的评审表。"""
import csv, json
from pathlib import Path
import yaml
from knowledge_base.embedder import CodeBertEmbedder
from knowledge_base.retriever import KnowledgeRetriever

QUERIES = [
"如何避免策略只做星间中继而不进行最终下传？", "如何同时提高最终交付及时性和任务完成率？", "如何减少高优先级任务过期？", "如何处理地面站资源冲突？", "如何鼓励负载均衡而不让策略选择空闲？", "协调冲突惩罚过大会造成什么问题？", "奖励尺度过大会怎样影响 PPO 的 Critic？", "如何避免奖励投机？", "如何设计多智能体共享团队奖励？", "任务完成与任务过期在势函数中应如何区分？", "如何限制无效动作？", "如何评价一个自动生成的奖励函数？"]
def main():
 c=yaml.safe_load(Path('configs/rappo.yaml').read_text(encoding='utf-8'))['rappo']; chunks=[json.loads(x) for x in Path(c['knowledge_base']['chunks']).read_text(encoding='utf-8').splitlines() if x]
 r=KnowledgeRetriever(c['knowledge_base']['index_directory'],chunks,CodeBertEmbedder(c['embedding'])); review=[]; query_rows=[]
 for index,q in enumerate(QUERIES,1):
  result=r.retrieve(q,5); r.audit('knowledge_sources/reports/retrieval_audit.jsonl',q,result,5); query_rows.append({'query_id':f'Q{index:02d}','query':q})
  for item in result['results']: review.append({'query_id':f'Q{index:02d}','query':q,'rank':item['rank'],'chunk_id':item['chunk_id'],'document_id':item['document_id'],'score':item['score'],'category':item['category'],'relevant':'','partially_relevant':'','incorrect':'','duplicate':'','context_complete':'','review_notes':'','reviewer':'','reviewed_at':''})
 review_dir=Path('knowledge_sources/review'); review_dir.mkdir(parents=True,exist_ok=True)
 (review_dir/'retrieval_test_queries.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in query_rows)+'\n',encoding='utf-8')
 with (review_dir/'retrieval_review.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
if __name__=='__main__': main()
