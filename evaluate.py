import os
from services.dataset_service import load_dataset
from services.preprocessing_service import preprocess
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.evaluation_service import EvaluationService

def run_evaluation() -> None:
    print("=== جاري تجهيز النظام للتقييم ===")
    dataset = load_dataset()
    
    indexing_service = IndexingService()
    

    inverted_index, doc_lengths = indexing_service.build_inverted_index(
        dataset, preprocess, "cord19_index.pkl"
    )
    doc_embeddings = indexing_service.compute_documents_embeddings(
        dataset, "cord19_embeddings.pkl"
    )
    
    retrieval_service = RetrievalService(inverted_index, doc_lengths, doc_embeddings)
    evaluation_service = EvaluationService()
    
    print("\n=== جاري جلب الإجابات النموذجية (Qrels) ===")
    all_qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in all_qrels:
            all_qrels[qrel.query_id] = {}

        all_qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
        
    print("تم جلب الـ Qrels بنجاح.")
    print("جاري تنفيذ الاستعلامات على جميع النماذج (قد يستغرق بضع ثوانٍ)...")
    
    results_tfidf = {}
    results_bm25 = {}
    results_bert = {}
    results_hybrid_parallel = {}
    

    for query in dataset.queries_iter():
        q_id = query.query_id
        # استخراج نص الاستعلام
        q_text = getattr(query, "text", "") or getattr(query, "query", "") or getattr(query, "title", "")
        q_tokens = preprocess(q_text)
        
        # 1. تنفيذ محرك VSM (TF-IDF)
        res_tfidf = retrieval_service.search_tfidf(q_tokens, top_k=10)
        results_tfidf[q_id] = [doc_id for doc_id, _ in res_tfidf]
        
        # 2. تنفيذ محرك BM25
        res_bm25 = retrieval_service.search_bm25(q_tokens, top_k=10)
        results_bm25[q_id] = [doc_id for doc_id, _ in res_bm25]
        
        # 3. تنفيذ محرك BERT
        res_bert = retrieval_service.search_embeddings(q_text, top_k=10)
        results_bert[q_id] = [doc_id for doc_id, _ in res_bert]
        
        # 4. تنفيذ المحرك الهجين (Hybrid Parallel)
        res_hp = retrieval_service.search_hybrid_parallel(q_text, q_tokens, top_k=10)
        results_hybrid_parallel[q_id] = [doc_id for doc_id, _ in res_hp]

    print("\n" + "="*50)
    print("=== نتائج التقييم النهائي (انسخها إلى التقرير الجامعي) ===")
    print("="*50)
    
    def print_metrics(name: str, results: dict) -> None:
        metrics = evaluation_service.evaluate_system(results, all_qrels, k=10)
        print(f"\n[نموذج {name}]")
        print(f"  - MAP     : {metrics['MAP']:.4f}")
        print(f"  - Recall  : {metrics['Recall']:.4f}")
        print(f"  - P@10    : {metrics['P@10']:.4f}")
        print(f"  - nDCG@10 : {metrics['nDCG']:.4f}")

    print_metrics("VSM (TF-IDF)", results_tfidf)
    print_metrics("BM25", results_bm25)
    print_metrics("Embeddings (BERT)", results_bert)
    print_metrics("Hybrid Parallel", results_hybrid_parallel)
    print("\n" + "="*50)

if __name__ == "__main__":
    run_evaluation()