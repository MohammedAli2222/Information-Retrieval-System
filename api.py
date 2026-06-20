import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from services.dataset_service import load_dataset
from services.preprocessing_service import preprocess, extract_text
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.query_refinement_service import QueryRefinementService
from services.rag_service import RagService


retrieval_service: RetrievalService | None = None
refinement_service: QueryRefinementService | None = None
rag_service: RagService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_service, refinement_service, rag_service
    print("[API] جاري تهيئة الخوادم وتحميل النماذج في الذاكرة...")
    
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
    
    retrieval_service = RetrievalService(
        inverted_index=inverted_index,
        doc_lengths=doc_lengths,
        doc_embeddings=doc_embeddings
    )
    
    refinement_service = QueryRefinementService()
    rag_service = RagService(retrieval_service=retrieval_service, dataset=dataset)
    
    print("[API] تمت التهيئة بنجاح! البوابة جاهزة لاستقبال الطلبات.")
    
    yield  
    
    print("[API] جاري إغلاق الخوادم وتحرير الذاكرة...")

app = FastAPI(
    title="IR System API", 
    description="REST API for Information Retrieval System with RAG Capabilities", 
    version="1.0.0",
    lifespan=lifespan
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_spell_check: bool = True
    use_synonyms: bool = True
    search_model: str = "hybrid_parallel" 

class RagRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
async def search(request: SearchRequest):
    if not retrieval_service or not refinement_service:
        raise HTTPException(status_code=503, detail="النظام قيد التهيئة، يرجى المحاولة بعد قليل.")
        
    try:
        refined_query = refinement_service.refine_query(
            request.query, 
            use_spell_check=request.use_spell_check, 
            use_synonyms=request.use_synonyms
        )
        
        query_tokens = preprocess(refined_query)
        
        results = []
        if request.search_model == "tfidf":
            results = retrieval_service.search_tfidf(query_tokens, top_k=request.top_k)
        elif request.search_model == "bm25":
            results = retrieval_service.search_bm25(query_tokens, top_k=request.top_k)
        elif request.search_model == "bert":
            results = retrieval_service.search_embeddings(refined_query, top_k=request.top_k)
        elif request.search_model == "hybrid_serial":
            results = retrieval_service.search_hybrid_serial(refined_query, query_tokens, top_k=request.top_k)
        elif request.search_model == "hybrid_parallel":
            results = retrieval_service.search_hybrid_parallel(refined_query, query_tokens, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail="نموذج البحث غير صالح.")
            
        formatted_results = [{"doc_id": doc_id, "score": float(score)} for doc_id, score in results]
        
        return {
            "original_query": request.query,
            "refined_query": refined_query,
            "search_model": request.search_model,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: RagRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="خدمة RAG قيد التهيئة، يرجى المحاولة بعد قليل.")
        
    try:
        response = rag_service.ask_rag(user_question=request.question, top_k=request.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    if not rag_service:
        raise HTTPException(status_code=503, detail="النظام قيد التهيئة، يرجى المحاولة بعد قليل.")
        
    try:

        doc = rag_service.docs_store.get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="الوثيقة غير موجودة في قاعدة البيانات.")
            
        doc_text: str = extract_text(doc)
        
        return {
            "doc_id": doc_id,
            "content": doc_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))