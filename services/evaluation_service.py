import math
from typing import Dict, List

class EvaluationService:

    def __init__(self) -> None:
        pass


    def precision_at_k(self, retrieved: List[str], qrels: Dict[str, int], k: int = 10) -> float:
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0
            

        relevant_retrieved = sum(1 for doc_id in retrieved_k if qrels.get(doc_id, 0) > 0)
        return relevant_retrieved / k


    def recall(self, retrieved: List[str], qrels: Dict[str, int]) -> float:
        total_relevant = sum(1 for rel in qrels.values() if rel > 0)
        if total_relevant == 0:
            return 0.0
            
        relevant_retrieved = sum(1 for doc_id in retrieved if qrels.get(doc_id, 0) > 0)
        return relevant_retrieved / total_relevant


    def average_precision(self, retrieved: List[str], qrels: Dict[str, int]) -> float:
        total_relevant = sum(1 for rel in qrels.values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        
        ap: float = 0.0
        relevant_retrieved: int = 0
        
        for i, doc_id in enumerate(retrieved):
            if qrels.get(doc_id, 0) > 0:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                ap += precision_at_i
                
        return ap / total_relevant


    def ndcg_at_k(self, retrieved: List[str], qrels: Dict[str, int], k: int = 10) -> float:
        retrieved_k = retrieved[:k]
        dcg: float = 0.0
        
        for i, doc_id in enumerate(retrieved_k):
            rel = qrels.get(doc_id, 0)

            dcg += (2**rel - 1) / math.log2(i + 2)
            

        ideal_rels = sorted(qrels.values(), reverse=True)[:k]
        idcg: float = 0.0
        
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1) / math.log2(i + 2)
            
        if idcg == 0.0:
            return 0.0
            
        return dcg / idcg


    def evaluate_system(self, system_results: Dict[str, List[str]], all_qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict[str, float]:
        sum_ap: float = 0.0
        sum_recall: float = 0.0
        sum_p10: float = 0.0
        sum_ndcg: float = 0.0
        
        valid_queries: int = 0
        
        for query_id, retrieved in system_results.items():
            if query_id not in all_qrels:
                continue
                
            qrels = all_qrels[query_id]

            if sum(1 for rel in qrels.values() if rel > 0) == 0:
                continue
                
            valid_queries += 1
            sum_ap += self.average_precision(retrieved, qrels)
            sum_recall += self.recall(retrieved, qrels)
            sum_p10 += self.precision_at_k(retrieved, qrels, k)
            sum_ndcg += self.ndcg_at_k(retrieved, qrels, k)
            
        if valid_queries == 0:
            return {"MAP": 0.0, "Recall": 0.0, "P@10": 0.0, "nDCG": 0.0}
            
        return {
            "MAP": sum_ap / valid_queries,
            "Recall": sum_recall / valid_queries,
            "P@10": sum_p10 / valid_queries,
            "nDCG": sum_ndcg / valid_queries
        }