import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Set

from services.dataset_service import load_dataset
from services.preprocessing_service import preprocess, extract_text
from services.database_service import DatabaseService
from services.indexing_service import IndexingService
from services.retrieval_service import RetrievalService
from services.query_refinement_service import QueryRefinementService
from services.rag_service import RagService
from services.clustering_service import ClusteringService
from services.cache_service import load_cache, save_cache

retrieval_service: RetrievalService | None = None
refinement_service: QueryRefinementService | None = None
rag_service: RagService | None = None
docs_store: Any = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retrieval_service, refinement_service, rag_service, docs_store
    print("[API] Starting initialization...")
    
    dataset = load_dataset()
    docs_store = dataset.docs_store()
    
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


    CLUSTER_CACHE = "cord19_clusters.pkl"
    doc_clusters = load_cache(CLUSTER_CACHE)
    
    if not doc_clusters and len(doc_embeddings) > 0:
        try:
            print("[API] Building document clusters for the first time...")

            n_clusters = min(50, max(5, len(doc_embeddings) // 4000))
            clustering_service = ClusteringService(doc_embeddings)
            doc_clusters = clustering_service.build_clusters(n_clusters=n_clusters)
            save_cache(doc_clusters, CLUSTER_CACHE)
        except ImportError:
            print("[API WARNING] scikit-learn not installed. Clustering feature disabled.")
            doc_clusters = None
        except Exception as e:
            print(f"[API WARNING] Clustering failed: {e}. Continuing without clusters.")
            doc_clusters = None
    
    retrieval_service = RetrievalService(
        inverted_index=inverted_index,
        doc_lengths=doc_lengths,
        doc_embeddings=doc_embeddings,
        doc_clusters=doc_clusters
    )
    
    dataset_vocabulary: Set[str] = set(inverted_index.keys())
    refinement_service = QueryRefinementService(dataset_vocabulary=dataset_vocabulary)
    
    rag_service = RagService(retrieval_service=retrieval_service, dataset=dataset)
    
    print("[API] Initialization complete. Ready to accept requests.")
    yield  
    print("[API] Shutting down...")

app = FastAPI(
    title="IR System API", 
    description="REST API for Information Retrieval System with RAG and Clustering Capabilities", 
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_spell_check: bool = True
    use_synonyms: bool = True
    use_clustering: bool = False
    search_model: str = "hybrid_parallel" 
    k1: float = 1.5      
    b: float = 0.75      
    alpha: float = 0.5   

class RagRequest(BaseModel):
    question: str
    top_k: int = 3
    alpha: float = 0.7 

@app.post("/search")
async def search(request: SearchRequest):
    if not retrieval_service or not refinement_service:
        raise HTTPException(status_code=503, detail="Service Unavailable")
        
    try:
        refined_query = refinement_service.refine_query(
            request.query, 
            use_spell_check=request.use_spell_check, 
            use_synonyms=request.use_synonyms
        )
        
        query_tokens = preprocess(refined_query)
        

        query_cluster = None
        if request.use_clustering:
            query_cluster = retrieval_service.get_query_cluster(refined_query)

        if request.search_model == "tfidf":
            results = retrieval_service.search_tfidf(query_tokens, top_k=request.top_k, use_clustering=request.use_clustering, query_cluster=query_cluster)
        elif request.search_model == "bm25":
            results = retrieval_service.search_bm25(query_tokens, k1=request.k1, b=request.b, top_k=request.top_k, use_clustering=request.use_clustering, query_cluster=query_cluster)
        elif request.search_model == "embeddings":
            results = retrieval_service.search_embeddings(refined_query, top_k=request.top_k, use_clustering=request.use_clustering, query_cluster=query_cluster)
        elif request.search_model == "hybrid_parallel":
            results = retrieval_service.search_hybrid_parallel(
                refined_query, query_tokens, alpha=request.alpha, k1=request.k1, b=request.b, top_k=request.top_k, use_clustering=request.use_clustering, query_cluster=query_cluster
            )
        elif request.search_model == "hybrid_serial":
            results = retrieval_service.search_hybrid_serial(
                refined_query, query_tokens, k1=request.k1, b=request.b, top_k=request.top_k, use_clustering=request.use_clustering, query_cluster=query_cluster
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search model.")

        return {
            "original_query": request.query,
            "refined_query": refined_query,
            "search_model": request.search_model,
            "used_clustering": request.use_clustering,
            "query_cluster_id": query_cluster,
            "results_count": len(results),
            "results": [{"doc_id": doc_id, "score": float(score)} for doc_id, score in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: RagRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service Unavailable")
        
    try:

        response = rag_service.ask_rag(
            request.question, 
            top_k=request.top_k,
            alpha=request.alpha
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    if not docs_store:
        raise HTTPException(status_code=503, detail="Service Unavailable")
        
    try:
        doc = docs_store.get(doc_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
            
        title = getattr(doc, "title", "No Title Available")
        content = extract_text(doc)
        
        if not content or content.strip() == "":
            content = "لا يوجد نص متوفر لهذه الوثيقة في قاعدة البيانات (No Content Available)."
        
        return {
            "doc_id": doc_id,
            "title": title,
            "content": content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))