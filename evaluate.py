import os
import pickle
from typing import Set, Dict, List
from services.dataset_service import load_dataset
from services.preprocessing_service import preprocess
from services.database_service import DatabaseService
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService
from services.query_refinement_service import QueryRefinementService

def run_evaluation() -> None:
    cache_file = "cache/evaluation_results_v3.pkl"
    evaluation_service = EvaluationService()

    if os.path.exists(cache_file):

        with open(cache_file, "rb") as f:
            cached_results = pickle.load(f)
        
        all_qrels = cached_results["all_qrels"]
        results_tfidf = cached_results["tfidf"]
        results_bm25 = cached_results["bm25"]
        results_bert = cached_results["bert"]
        results_hs_before = cached_results["hs_before"]
        results_hp_before = cached_results["hp_before"]
        results_hp_after_spell = cached_results["hp_after_spell"]
        
    else:
        dataset = load_dataset()
        db_service = DatabaseService()
        indexing_service = IndexingService(db_service=db_service)
        
        inverted_index, doc_lengths = indexing_service.build_inverted_index(
            dataset=dataset, 
            preprocess_fn=preprocess
        )
        
        doc_embeddings = indexing_service.compute_documents_embeddings(
            dataset=dataset, 
            cache_name="cord19_embeddings.pkl"
        )
        
        retrieval_service = RetrievalService(inverted_index, doc_lengths, doc_embeddings)
        
        dataset_vocabulary: Set[str] = set(inverted_index.keys())
        refinement_service = QueryRefinementService(
            dataset_vocabulary=dataset_vocabulary,
            enable_spellcheck=True,
            enable_synonyms=True
        )
        
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
            
            # شرح الكود باللغة العربية: تم رفع عمق الاسترجاع إلى 5000 بناءً على طلبك لزيادة الاستدعاء الأقصى
            results_tfidf[q_id] = [doc_id for doc_id, _ in retrieval_service.search_tfidf(raw_tokens, top_k=5000)]
            results_bm25[q_id] = [doc_id for doc_id, _ in retrieval_service.search_bm25(raw_tokens, top_k=5000)]
            results_bert[q_id] = [doc_id for doc_id, _ in retrieval_service.search_embeddings(raw_query, top_k=5000)]
            results_hp_before[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_parallel(raw_query, raw_tokens, top_k=5000)]
            results_hs_before[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_serial(raw_query, raw_tokens, top_k=5000)]

            refined_spell_query = refinement_service.refine_query(raw_query, use_spell_check=True, use_synonyms=False)
            refined_spell_tokens = preprocess(refined_spell_query)
            
            results_hp_after_spell[q_id] = [doc_id for doc_id, _ in retrieval_service.search_hybrid_parallel(refined_spell_query, refined_spell_tokens, top_k=5000)]

        os.makedirs("cache", exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({
                "all_qrels": all_qrels,
                "tfidf": results_tfidf,
                "bm25": results_bm25,
                "bert": results_bert,
                "hs_before": results_hs_before,
                "hp_before": results_hp_before,
                "hp_after_spell": results_hp_after_spell
            }, f)

    def print_metrics(name: str, results: dict, show_hits: bool = False) -> None:
        # شرح الكود باللغة العربية: إبقاء التقييم على أول 10 وثائق للحفاظ على دقة المعايير
        metrics = evaluation_service.evaluate_system(results, all_qrels, k=10)
        print(f"\n[{name}]")
        print(f"  - MAP     : {metrics.get('MAP', 0.0):.4f}")
        print(f"  - Recall  : {metrics.get('Recall', 0.0):.4f}")
        print(f"  - P@10    : {metrics.get('P@10', 0.0):.4f}")
        print(f"  - nDCG@10 : {metrics.get('nDCG', 0.0):.4f}")
        
        # شرح الكود باللغة العربية: تفعيل خيار طباعة تقرير الإصابات للمحركات المطلوبة فقط
        if show_hits:
            report = evaluation_service.per_query_relevant_retrieval_report(results, all_qrels)
            print(f"\n--- {name.upper()} RELEVANT HITS ---")
            # شرح الكود باللغة العربية: طباعة أول 50 استعلاماً بالصيغة التي طلبها صديقك
            for qid, stats in list(report.items())[:50]:
                print(qid, f"{int(stats['hit'])}/{int(stats['total_relevant'])} ({stats['recall']:.3f})")

    print("\n" + "="*70)
    print("=== القسم الأول: مقارنة أداء النماذج المختلفة (الاستعلام الخام) ===")
    print("="*70)
    print_metrics("نموذج VSM (TF-IDF)", results_tfidf)
    print_metrics("النموذج الاحتمالي (BM25)", results_bm25)
    print_metrics("نموذج التضمين (BERT Embeddings)", results_bert)
    print_metrics("النموذج الهجين التسلسلي (Hybrid Serial)", results_hs_before)
    
    # شرح الكود باللغة العربية: طباعة التقرير الكامل للمحرك الهجين المتوازي (أفضل محرك لديك)
    print_metrics("النموذج الهجين المتوازي (Hybrid Parallel)", results_hp_before, show_hits=True)

    print("\n" + "="*70)
    print("=== القسم الثاني: تأثير الميزات الإضافية على جودة الاسترجاع ===")
    print("="*70)
    print_metrics("قبل تطبيق الميزة (المحرك الهجين المتوازي - استعلام خام)", results_hp_before)
    print_metrics("بعد تطبيق الميزة (المحرك الهجين المتوازي + التصحيح الإملائي الذكي)", results_hp_after_spell)
    print("\n" + "="*70)

if __name__ == "__main__":
    run_evaluation()