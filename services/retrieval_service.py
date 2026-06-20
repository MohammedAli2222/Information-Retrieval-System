import math
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

class RetrievalService:

    def __init__(self, inverted_index: Dict[str, Dict[str, int]], doc_lengths: Dict[str, int], doc_embeddings: Dict[str, Any]) -> None:
        self.index = inverted_index
        self.doc_lengths = doc_lengths
        self.N = len(doc_lengths)
        self.avg_doc_len = sum(doc_lengths.values()) / self.N if self.N > 0 else 0
        self.doc_embeddings = doc_embeddings
        
        self.idf: Dict[str, float] = {}
        for term, postings in self.index.items():
            df = len(postings)
            self.idf[term] = math.log10(self.N / df) if df > 0 else 0
            
        print("جاري تحميل نموذج BERT لمعالجة الاستعلامات...")
        self.embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    def search_tfidf(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        query_counts = Counter(query_tokens)
        query_weights: Dict[str, float] = {}
        query_length_sq: float = 0.0
        
        for term, freq in query_counts.items():
            if term in self.idf:
                tf = 1 + math.log10(freq) if freq > 0 else 0
                weight = tf * self.idf[term]
                query_weights[term] = weight
                query_length_sq += weight ** 2
                
        if query_length_sq == 0:
            return []
            
        query_norm = math.sqrt(query_length_sq)
        doc_scores: Counter = Counter()
        
        for term, q_weight in query_weights.items():
            if term in self.index:
                for doc_id, tf_raw in self.index[term].items():
                    doc_tf = 1 + math.log10(tf_raw)
                    doc_weight = doc_tf * self.idf[term]
                    doc_scores[doc_id] += q_weight * doc_weight

        final_ranked: List[Tuple[str, float]] = []
        for doc_id, dot_product in doc_scores.items():
            doc_norm = math.sqrt(self.doc_lengths[doc_id]) 
            cosine_sim = dot_product / (query_norm * doc_norm) if doc_norm > 0 else 0
            final_ranked.append((doc_id, float(cosine_sim)))

        final_ranked.sort(key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]

    def search_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75, top_k: int = 10) -> List[Tuple[str, float]]:
        doc_scores: Counter = Counter()
        query_counts = Counter(query_tokens)
        
        for term, q_freq in query_counts.items():
            if term in self.index:
                postings = self.index[term]
                df = len(postings)
                idf_bm25 = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                
                for doc_id, tf in postings.items():
                    doc_len = self.doc_lengths[doc_id]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    doc_scores[doc_id] += idf_bm25 * (numerator / denominator)

        final_ranked = list(doc_scores.items())
        final_ranked.sort(key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]

    def search_embeddings(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.doc_embeddings:
            return []
            
        query_vector = self.embedding_model.encode(query_text)
        final_ranked: List[Tuple[str, float]] = []
        q_norm = np.linalg.norm(query_vector)
        
        for doc_id, doc_vector in self.doc_embeddings.items():
            d_norm = np.linalg.norm(doc_vector)
            if q_norm > 0 and d_norm > 0:
                cosine_sim = np.dot(query_vector, doc_vector) / (q_norm * d_norm)
                final_ranked.append((doc_id, float(cosine_sim)))
                
        final_ranked.sort(key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]

    def search_hybrid_serial(self, query_text: str, query_tokens: List[str], candidate_k: int = 50, top_k: int = 10) -> List[Tuple[str, float]]:
        raw_bm25 = self.search_bm25(query_tokens, top_k=candidate_k)
        candidate_ids = [doc_id for doc_id, _ in raw_bm25]
        
        if not candidate_ids or not self.doc_embeddings:
            return []
        
        query_vector = self.embedding_model.encode(query_text)
        q_norm = np.linalg.norm(query_vector)
        
        serial_ranked: List[Tuple[str, float]] = []
        for doc_id in candidate_ids:
            doc_vector = self.doc_embeddings[doc_id]
            d_norm = np.linalg.norm(doc_vector)
            cosine_sim = np.dot(query_vector, doc_vector) / (q_norm * d_norm) if q_norm > 0 and d_norm > 0 else 0
            serial_ranked.append((doc_id, float(cosine_sim)))
            
        serial_ranked.sort(key=lambda x: x[1], reverse=True)
        return serial_ranked[:top_k]

    def search_hybrid_parallel(self, query_text: str, query_tokens: List[str], alpha: float = 0.5, top_k: int = 10) -> List[Tuple[str, float]]:
        bm25_res = self.search_bm25(query_tokens, top_k=self.N)
        bert_res = self.search_embeddings(query_text, top_k=self.N)
        
        bm25_dict = dict(bm25_res)
        bert_dict = dict(bert_res)
        
        max_bm25 = max(bm25_dict.values()) if bm25_dict else 1.0
        
        parallel_scores: Dict[str, float] = {}
        all_docs = set(bm25_dict.keys()).union(set(bert_dict.keys()))
        
        for doc_id in all_docs:

            norm_bm25 = (bm25_dict.get(doc_id, 0.0) / max_bm25) if max_bm25 > 0 else 0.0
            score_bert = bert_dict.get(doc_id, 0.0)
            
            parallel_scores[doc_id] = alpha * norm_bm25 + (1 - alpha) * score_bert
            
        final_ranked = list(parallel_scores.items())
        final_ranked.sort(key=lambda x: x[1], reverse=True)
        return final_ranked[:top_k]