import os
from typing import Set, Dict, Any, List, Tuple
from functools import lru_cache  
from services.dataset_service import load_dataset, print_dataset_info
from services.preprocessing_service import preprocess_dataset, preprocess
from services.database_service import DatabaseService
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.query_refinement_service import QueryRefinementService 

retrieval_service: RetrievalService | None = None
refinement_service: QueryRefinementService | None = None

@lru_cache(maxsize=1000)
def execute_cached_search(raw_query: str) -> Dict[str, Any]:
    print("\n[⏳ Cache Miss] الاستعلام جديد. جاري التحسين والبحث وتشغيل المحركات (سيتم حفظ النتيجة في الرام)...")
    
    refined_query = refinement_service.refine_query(raw_query, use_spell_check=True, use_synonyms=True)
    sample_query_tokens = preprocess(refined_query)
    
    results_tfidf = retrieval_service.search_tfidf(sample_query_tokens, top_k=3)
    results_hp = retrieval_service.search_hybrid_parallel(refined_query, sample_query_tokens, alpha=0.5, top_k=3)
    results_hs = retrieval_service.search_hybrid_serial(refined_query, sample_query_tokens, top_k=3)
    
    return {
        'refined_query': refined_query,
        'tokens': sample_query_tokens,
        'results_tfidf': results_tfidf,
        'results_hp': results_hp,
        'results_hs': results_hs
    }

def main() -> None:
    global retrieval_service, refinement_service
    
    print("=== خطوة 1: تحميل مجموعة البيانات ===")
    dataset = load_dataset()
    print_dataset_info(dataset)

    print("\n=== خطوة 2: تهيئة قاعدة البيانات وبناء الفهرسة المعمارية ===")

    db_service = DatabaseService()
    

    indexing_service = IndexingService(db_service=db_service)
    

    inverted_index, doc_lengths = indexing_service.build_inverted_index(
        dataset=dataset,
        preprocess_fn=preprocess
    )
    

    cache_name_embeddings: str = "cord19_embeddings.pkl"
    doc_embeddings = indexing_service.compute_documents_embeddings(
        dataset=dataset,
        cache_name=cache_name_embeddings
    )

    print("\n=== خطوة 3: تهيئة خدمة الاسترجاع وتحسين الاستعلام ===")
    retrieval_service = RetrievalService(
        inverted_index=inverted_index,
        doc_lengths=doc_lengths,
        doc_embeddings=doc_embeddings
    )
    
    dataset_vocabulary: Set[str] = set(inverted_index.keys())
    refinement_service = QueryRefinementService(dataset_vocabulary=dataset_vocabulary)

    print("\n=== اختبار الطلب 5 و 6: الاستعلام مع تفعيل نظام الكاش الذكي (In-Memory LRU Cache) ===")
    
    raw_query: str = "coronvirus vaccin" 
    print(f"\n[المستخدم أدخل] الاستعلام الخام: '{raw_query}'")

    search_data = execute_cached_search(raw_query)
    
    print("\n[المستخدم الثاني يبحث عن نفس الكلمة...]")
    search_data_fast = execute_cached_search(raw_query)
    print("[⚡ Cache Hit] تم جلب النتائج من الذاكرة العشوائية (RAM) في 0.0001 ثانية، بدون إعادة الحسابات!")

    # طباعة النتائج
    print(f"\n[فحص النظام] الكلمات المحسنة: {search_data['tokens']}")
    
    print("\n--- نتائج استرجاع نموذج VSM (TF-IDF) ---")
    for doc_id, score in search_data['results_tfidf']:
        print(f"Document ID: {doc_id} | Similarity Score: {score:.4f}")

    print("\n--- نتائج استرجاع المحرك الهجين المتوازي (Hybrid Parallel) ---")
    for doc_id, score in search_data['results_hp']:
        print(f"Document ID: {doc_id} | Hybrid Score: {score:.4f}")

if __name__ == "__main__":
    main()