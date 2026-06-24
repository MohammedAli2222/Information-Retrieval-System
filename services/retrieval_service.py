import math
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from collections import defaultdict

class RetrievalService:
    def __init__(self, inverted_index: Dict[str, Dict[str, int]], doc_lengths: Dict[str, int], doc_embeddings: Dict[str, Any]) -> None:
       
       
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.doc_embeddings = doc_embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. حساب إجمالي عدد الوثائق (N)
        self.N: int = len(self.doc_lengths)
        
        # 2. حساب متوسط طول الوثيقة (avgdl) المطلوب لمعادلة BM25
        total_length: int = sum(self.doc_lengths.values())
        self.avgdl: float = total_length / self.N if self.N > 0 else 1.0

    def search_tfidf(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
       
       
        scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                df = len(postings)
                

                idf = math.log(self.N / (df + 1)) + 1.0
                
                for doc_id, tf in postings.items():

                    tf_weight = 1.0 + math.log(tf) if tf > 0 else 0.0
                    scores[doc_id] += tf_weight * idf

        # ترتيب النتائج تنازلياً
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

    def search_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75, top_k: int = 10) -> List[Tuple[str, float]]:
       
       
        scores: Dict[str, float] = defaultdict(float)
        
        for term in query_tokens:
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                df = len(postings)
                
                # حساب الـ IDF الخاص بمعادلة BM25 (Robertson-Spärck Jones)
                idf_numerator = self.N - df + 0.5
                idf_denominator = df + 0.5
                idf = math.log((idf_numerator / idf_denominator) + 1.0)
                
                for doc_id, tf in postings.items():
                    dl = self.doc_lengths.get(doc_id, self.avgdl)
                    
                    # معادلة BM25 لحساب الـ Term Frequency Score
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
                    
                    scores[doc_id] += idf * (numerator / denominator)

        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

    def search_embeddings(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
       
       
        # توليد المتجه للاستعلام
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        scores: Dict[str, float] = {}
        query_norm = norm(query_embedding)
        
        # حساب تشابه جيب التمام (Cosine Similarity) مع جميع الوثائق
        for doc_id, doc_emb in self.doc_embeddings.items():
            doc_norm = norm(doc_emb)
            if query_norm > 0 and doc_norm > 0:
                similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                scores[doc_id] = float(similarity)

        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

    def search_hybrid_parallel(self, query_text: str, query_tokens: List[str], alpha: float = 0.5, k1: float = 1.5, b: float = 0.75, top_k: int = 10) -> List[Tuple[str, float]]:
       
       
        # 1. جلب النتائج من المحركين (نجلب عدد أكبر لضمان التقاطع الجيد)
        bm25_results = dict(self.search_bm25(query_tokens, k1=k1, b=b, top_k=top_k * 5))
        bert_results = dict(self.search_embeddings(query_text, top_k=top_k * 5))
        
        # 2. توحيد مجالات النقاط (Min-Max Normalization) لتصبح بين 0 و 1
        norm_bm25 = self._normalize_scores(bm25_results)
        norm_bert = self._normalize_scores(bert_results)
        
        hybrid_scores: Dict[str, float] = defaultdict(float)
        
        # دمج كل الوثائق المسترجعة
        all_docs = set(norm_bm25.keys()).union(set(norm_bert.keys()))
        
        # 3. احتساب الدرجة النهائية حسب طريقة الدمج (Score Fusion)
        for doc_id in all_docs:
            score_bm25 = norm_bm25.get(doc_id, 0.0)
            score_bert = norm_bert.get(doc_id, 0.0)
            
            # المعادلة الخطية للدمج (Linear Combination Fusion)
            hybrid_scores[doc_id] = (alpha * score_bert) + ((1.0 - alpha) * score_bm25)
            
        sorted_results = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

    def search_hybrid_serial(self, query_text: str, query_tokens: List[str], k1: float = 1.5, b: float = 0.75, top_k: int = 10) -> List[Tuple[str, float]]:
       
       
        # 1. الاسترجاع الأولي السريع
        initial_results = self.search_bm25(query_tokens, k1=k1, b=b, top_k=top_k * 5)
        if not initial_results:
            return []
            
        candidate_doc_ids = [doc_id for doc_id, _ in initial_results]
        
        # 2. توليد متجه الاستعلام
        query_embedding = self.embedding_model.encode([query_text])[0]
        query_norm = norm(query_embedding)
        
        scores: Dict[str, float] = {}
        
        # 3. إعادة الترتيب الدلالي للمرشحين فقط (تسريع الأداء وتقليل العمليات الحسابية)
        for doc_id in candidate_doc_ids:
            if doc_id in self.doc_embeddings:
                doc_emb = self.doc_embeddings[doc_id]
                doc_norm = norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                    scores[doc_id] = float(similarity)
                    
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:top_k]

    def _normalize_scores(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        
        
        if not scores_dict:
            return {}
            
        scores = list(scores_dict.values())
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {k: 1.0 for k in scores_dict.keys()}
            
        normalized = {}
        for doc_id, score in scores_dict.items():
            normalized[doc_id] = (score - min_score) / (max_score - min_score)
            
        return normalized