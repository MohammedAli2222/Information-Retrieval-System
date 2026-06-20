import os
from typing import Dict, Tuple, Callable, Any, List
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from services.cache_service import save_cache, load_cache
from services.preprocessing_service import extract_text

class IndexingService:

    def __init__(self) -> None:

        self.embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    def build_inverted_index(self, dataset: Any, preprocess_fn: Callable[[str], List[str]], cache_name: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        cached: Tuple[Dict[str, Dict[str, int]], Dict[str, int]] | None = load_cache(cache_name)
        
        if cached is not None:
            print(f"Loaded index cache: {cache_name}")
            return cached

        print(f"Building index: {cache_name}")
        
        inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        doc_lengths: Dict[str, int] = {}

        for doc in dataset.docs_iter():
            text: str = extract_text(doc)
            tokens: List[str] = preprocess_fn(text)
            doc_lengths[doc.doc_id] = len(tokens)
            
            term_counts: Counter[str] = Counter(tokens)
            for term, freq in term_counts.items():
                inverted_index[term][doc.doc_id] = freq

        result: Tuple[Dict[str, Dict[str, int]], Dict[str, int]] = (dict(inverted_index), doc_lengths)
        
        save_cache(result, cache_name)
        print(f"Saved index cache: {cache_name}")
        
        return result

    def compute_documents_embeddings(self, dataset: Any, cache_name: str) -> Dict[str, Any]:
        cached: Dict[str, Any] | None = load_cache(cache_name)
        
        if cached is not None:
            print(f"Loaded embeddings cache: {cache_name}")
            return cached
            
        print("جاري توليد الـ Embeddings للوثائق...")
        doc_ids: List[str] = []
        texts: List[str] = []
        
        for doc in dataset.docs_iter():
            text: str = getattr(doc, "title", "") + " " + getattr(doc, "abstract", "")
            if text.strip():
                doc_ids.append(doc.doc_id)
                texts.append(text)
                
        embeddings: Any = self.embedding_model.encode(texts, show_progress_bar=True)
        
        doc_embeddings: Dict[str, Any] = {}
        for doc_id, emb in zip(doc_ids, embeddings):
            doc_embeddings[doc_id] = emb
            
        save_cache(doc_embeddings, cache_name)
        print("تم توليد وحفظ الـ Embeddings بنجاح!")
        
        return doc_embeddings