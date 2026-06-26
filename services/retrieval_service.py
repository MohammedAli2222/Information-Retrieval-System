import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

class RetrievalService:
    def __init__(
        self,
        inverted_index: Dict[str, Dict[str, int]],
        doc_lengths: Dict[str, int],
        doc_embeddings: Dict[str, Any],
        doc_clusters: Dict[str, int] = None
    ) -> None:
        self.index = inverted_index
        self.doc_lengths = doc_lengths
        self.N = len(doc_lengths)
        self.avgdl = sum(doc_lengths.values()) / self.N if self.N > 0 else 1.0

        self.doc_embeddings = doc_embeddings
        self.doc_clusters = doc_clusters or {}


        self.cluster_to_docs = defaultdict(list)
        for doc_id, cluster_id in self.doc_clusters.items():
            self.cluster_to_docs[cluster_id].append(doc_id)

        self.all_docs = list(doc_lengths.keys())

        self.idf: Dict[str, float] = {}
        for term, postings in self.index.items():
            df = len(postings)
            self.idf[term] = math.log10(self.N / df) if df > 0 else 0

        print("جاري تحميل نموذج BERT لمعالجة الاستعلامات...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_query_cluster(self, query_text: str):

        if not self.doc_clusters:
            return None

        query_vec = self.embedding_model.encode(query_text)
        cluster_scores = defaultdict(list)

        for doc_id, cluster_id in self.doc_clusters.items():
            cluster_scores[cluster_id].append(self.doc_embeddings[doc_id])

        best_cluster = None
        best_sim = -1

        for cluster_id, vecs in cluster_scores.items():
            centroid = np.mean(vecs, axis=0)
            q_norm = np.linalg.norm(query_vec)
            c_norm = np.linalg.norm(centroid)

            if q_norm > 0 and c_norm > 0:
                sim = np.dot(query_vec, centroid) / (q_norm * c_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster_id

        return best_cluster

    def _get_candidate_docs(self, query_cluster: int, use_clustering: bool):
        if not use_clustering or query_cluster is None:
            return self.all_docs
        candidates = self.cluster_to_docs.get(query_cluster, [])
        return candidates if candidates else self.all_docs

    def search_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75, top_k: int = 10, use_clustering: bool = False, query_cluster: int = None) -> List[Tuple[str, float]]:
        candidates = self._get_candidate_docs(query_cluster, use_clustering)
        candidate_set = set(candidates)

        scores: Dict[str, float] = defaultdict(float)
        query_counts = Counter(query_tokens)

        for term in query_counts:
            if term not in self.index:
                continue

            postings = self.index[term]
            df = len(postings)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, tf in postings.items():
                if use_clustering and doc_id not in candidate_set:
                    continue

                dl = self.doc_lengths.get(doc_id, self.avgdl)
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (dl / self.avgdl))
                scores[doc_id] += idf * (numerator / denominator)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def search_tfidf(self, query_tokens: List[str], top_k: int = 10, use_clustering: bool = False, query_cluster: int = None) -> List[Tuple[str, float]]:
        candidates = set(self._get_candidate_docs(query_cluster, use_clustering))
        query_counts = Counter(query_tokens)
        query_weights = {}
        query_len = 0

        for t, f in query_counts.items():
            if t in self.idf:
                w = (1 + math.log10(f)) * self.idf[t]
                query_weights[t] = w
                query_len += w ** 2

        if query_len == 0:
            return []

        q_norm = math.sqrt(query_len)
        scores = Counter()

        for t, q_w in query_weights.items():
            if t not in self.index:
                continue
            for doc_id, tf in self.index[t].items():
                if use_clustering and doc_id not in candidates:
                    continue
                d_w = (1 + math.log10(tf)) * self.idf[t]
                scores[doc_id] += q_w * d_w

        ranked = []
        for d, s in scores.items():
            norm_val = math.sqrt(self.doc_lengths.get(d, 1))
            ranked.append((d, s / (q_norm * norm_val) if norm_val > 0 else 0))

        return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_k]

    def search_embeddings(self, query_text: str, top_k: int = 10, use_clustering: bool = False, query_cluster: int = None) -> List[Tuple[str, float]]:
        if not self.doc_embeddings:
            return []
        
        candidates = set(self._get_candidate_docs(query_cluster, use_clustering))
        query_embedding = self.embedding_model.encode([query_text])[0]
        q_norm = norm(query_embedding)
        
        scores: Dict[str, float] = {}

        for doc_id, doc_emb in self.doc_embeddings.items():
            if use_clustering and doc_id not in candidates:
                continue
            d_norm = norm(doc_emb)
            if q_norm > 0 and d_norm > 0:
                similarity = np.dot(query_embedding, doc_emb) / (q_norm * d_norm)
                scores[doc_id] = float(similarity)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def search_hybrid_parallel(self, query_text: str, query_tokens: List[str], alpha: float = 0.5, k1: float = 1.5, b: float = 0.75, top_k: int = 10, use_clustering: bool = False, query_cluster: int = None) -> List[Tuple[str, float]]:
        bm25_res = self.search_bm25(query_tokens, k1=k1, b=b, top_k=top_k * 5, use_clustering=use_clustering, query_cluster=query_cluster)
        bert_res = self.search_embeddings(query_text, top_k=top_k * 5, use_clustering=use_clustering, query_cluster=query_cluster)
        
        bm25_dict = dict(bm25_res)
        bert_dict = dict(bert_res)

        max_bm25 = max(bm25_dict.values()) if bm25_dict else 1.0
        all_docs = set(bm25_dict.keys()).union(set(bert_dict.keys()))
        
        hybrid_scores: Dict[str, float] = {}

        for doc_id in all_docs:
            bm25_score = bm25_dict.get(doc_id, 0.0)
            bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            bert_score = bert_dict.get(doc_id, 0.0)
            
            hybrid_scores[doc_id] = (alpha * bert_score) + ((1.0 - alpha) * bm25_norm)

        return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def search_hybrid_serial(self, query_text: str, query_tokens: List[str], k1: float = 1.5, b: float = 0.75, top_k: int = 10, use_clustering: bool = False, query_cluster: int = None) -> List[Tuple[str, float]]:
        initial_results = self.search_bm25(query_tokens, k1=k1, b=b, top_k=top_k * 5, use_clustering=use_clustering, query_cluster=query_cluster)
        if not initial_results:
            return []
            
        candidate_doc_ids = [doc_id for doc_id, _ in initial_results]
        query_embedding = self.embedding_model.encode([query_text])[0]
        query_norm = norm(query_embedding)
        
        scores: Dict[str, float] = {}
        
        for doc_id in candidate_doc_ids:
            if doc_id in self.doc_embeddings:
                doc_emb = self.doc_embeddings[doc_id]
                doc_norm = norm(doc_emb)
                if query_norm > 0 and doc_norm > 0:
                    similarity = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)
                    scores[doc_id] = float(similarity)
                    
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]