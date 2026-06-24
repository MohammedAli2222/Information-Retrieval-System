import os
from typing import Set, Dict, List
from services.dataset_service import load_dataset
from services.preprocessing_service import preprocess
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService
from services.query_refinement_service import QueryRefinementService

def run_evaluation() -> None:
    print("جاري تحميل النظام وتقييم النماذج، يرجى الانتظار بضع دقائق...")
    
    dataset = load_dataset()
    indexing_service = IndexingService()
    
    inverted_index, doc_lengths = indexing_service.build_inverted_index(
        dataset=dataset, 
        preprocess_fn=preprocess, 
        cache_name="cord19_index.pkl"
    )
    doc_embeddings = indexing_service.compute_documents_embeddings(
        dataset=dataset, 
        cache_name="cord19_embeddings.pkl"
    )
    
    retrieval_service = RetrievalService(inverted_index, doc_lengths, doc_embeddings)
    evaluation_service = EvaluationService()
    
    dataset_vocabulary: Set[str] = set(inverted_index.keys())
    refinement_service = QueryRefinementService(dataset_vocabulary=dataset_vocabulary)
    
    all_qrels: Dict[str, Dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in all_qrels:
            all_qrels[qrel.query_id] = {}
        all_qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    results_tfidf: Dict[str, List[str]] = {}
    results_bm25: Dict[str, List[str]] = {}
    results_bert: Dict[str, List[str]] = {}
    results_hp_before: Dict[str, List[str]] = {}
    results_hs_before: Dict[str, List[str]] = {}
    results_hp_after_spell: Dict[str, List[str]] = {}

    for query in dataset.queries_iter():
        q_id = query.query_id
        raw_query = getattr(query, "text", "") or getattr(query, "query", "") or getattr(query, "title", "")
        raw_tokens = preprocess(raw_query)
        
        results_tfidf[q_id] = [doc_id for doc_id, _ in retrieval_service.search_tfidf(raw_tokens, top_k=10)]
        results_bm25[q_id] = [doc_id for doc_id, _ in retrieval_service.search_bm25(raw_tokens, top_k=10)]
        results_bert[q_id] = [doc_id for doc_id, _ in retrieval_service.search_embeddings(raw_query, top_k=10)]
        results_hp_before[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_parallel(raw_query, raw_tokens, top_k=10)]
        results_hs_before[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_serial(raw_query, raw_tokens, top_k=10)]

        refined_spell_query = refinement_service.refine_query(raw_query, use_spell_check=True, use_synonyms=False)
        refined_spell_tokens = preprocess(refined_spell_query)
        
        results_hp_after_spell[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_parallel(refined_spell_query, refined_spell_tokens, top_k=10)]

    def print_metrics(name: str, results: dict) -> None:
        metrics = evaluation_service.evaluate_system(results, all_qrels, k=10)
        print(f"\n[{name}]")
        print(f"  - MAP     : {metrics.get('MAP', 0.0):.4f}")
        print(f"  - Recall  : {metrics.get('Recall', 0.0):.4f}")
        print(f"  - P@10    : {metrics.get('P@10', 0.0):.4f}")
        print(f"  - nDCG@10 : {metrics.get('nDCG', 0.0):.4f}")

    print("\n" + "="*70)
    print("=== القسم الأول: مقارنة أداء النماذج المختلفة (الاستعلام الخام) ===")
    print("="*70)
    print_metrics("نموذج VSM (TF-IDF)", results_tfidf)
    print_metrics("النموذج الاحتمالي (BM25)", results_bm25)
    print_metrics("نموذج التضمين (BERT Embeddings)", results_bert)
    print_metrics("النموذج الهجين التسلسلي (Hybrid Serial)", results_hs_before)
    print_metrics("النموذج الهجين المتوازي (Hybrid Parallel)", results_hp_before)

    print("\n" + "="*70)
    print("=== القسم الثاني: تأثير الميزات الإضافية على جودة الاسترجاع ===")
    print("="*70)
    print_metrics("قبل تطبيق الميزة (المحرك الهجين المتوازي - استعلام خام)", results_hp_before)
    print_metrics("بعد تطبيق الميزة (المحرك الهجين المتوازي + التصحيح الإملائي الذكي)", results_hp_after_spell)
    print("\n" + "="*70)

if __name__ == "__main__":
    run_evaluation()