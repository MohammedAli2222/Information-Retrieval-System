import os
from services.dataset_service import load_dataset, print_dataset_info
from services.preprocessing_service import preprocess_dataset, preprocess
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.query_refinement_service import QueryRefinementService 

def main() -> None:

    print("=== خطوة 1: تحميل مجموعة البيانات ===")
    dataset = load_dataset()
    print_dataset_info(dataset)


    print("\n=== خطوة 2: المعالجة المسبقة للبيانات ===")
    cache_name_preprocess: str = "cord19_preprocessed.pkl"
    docs, queries = preprocess_dataset(dataset, cache_name_preprocess)
    print(f"تمت المعالجة المسبقة بنجاح. عدد الوثائق المعالجة في القاموس: {len(docs)}")


    print("\n=== خطوة 3: بناء الفهرسة والتمثيل المعماري ===")
    indexing_service = IndexingService()
    
    cache_name_index: str = "cord19_index.pkl"
    inverted_index, doc_lengths = indexing_service.build_inverted_index(
        dataset=dataset,
        preprocess_fn=preprocess,
        cache_name=cache_name_index
    )
    
    cache_name_embeddings: str = "cord19_embeddings.pkl"
    doc_embeddings = indexing_service.compute_documents_embeddings(
        dataset=dataset,
        cache_name=cache_name_embeddings
    )

    print("\n=== خطوة 4: تهيئة خدمة الاسترجاع واختبار النماذج المطلوبة في التقرير ===")
    retrieval_service = RetrievalService(
        inverted_index=inverted_index,
        doc_lengths=doc_lengths,
        doc_embeddings=doc_embeddings
    )


    print("\n=== اختبار الطلب 5: تحسين الاستعلام (Query Refinement) ===")
    

    raw_query: str = "coronvirus vaccin" 
    print(f"\n[المستخدم أدخل] الاستعلام الخام: '{raw_query}'")


    refinement_service = QueryRefinementService()
    refined_query = refinement_service.refine_query(raw_query, use_spell_check=True, use_synonyms=True)
    

    sample_query_tokens = preprocess(refined_query)
    
    print(f"[فحص النظام] الكلمات المفتاحية بعد التحسين والـ Stemming: {sample_query_tokens}")


    print("\n--- نتائج استرجاع نموذج VSM (TF-IDF) ---")
    results_tfidf = retrieval_service.search_tfidf(sample_query_tokens, top_k=3)
    for doc_id, score in results_tfidf:
        print(f"Document ID: {doc_id} | Similarity Score: {score:.4f}")


    print("\n--- نتائج استرجاع نموذج Embeddings (BERT) ---")
    results_bert = retrieval_service.search_embeddings(refined_query, top_k=3)
    for doc_id, score in results_bert:
        print(f"Document ID: {doc_id} | Dense Score: {score:.4f}")

if __name__ == "__main__":
    main()