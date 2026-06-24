from textblob import TextBlob
from typing import List, Set, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np


from services.preprocessing_service import preprocess

class QueryRefinementService:
   
   

    def __init__(self, dataset_vocabulary: Optional[Set[str]] = None):
        self.dataset_vocabulary = dataset_vocabulary
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vocab_embeddings_cache: Dict[str, Any] = {}

    def _get_or_compute_embedding(self, word: str) -> Any:
        lower_word = word.lower()
        if lower_word not in self.vocab_embeddings_cache:
            self.vocab_embeddings_cache[lower_word] = self.embedding_model.encode([lower_word])[0]
        return self.vocab_embeddings_cache[lower_word]

    def correct_spelling(self, query: str) -> str:
       
       
        words = query.split()
        corrected_words = []

        for word in words:

            if not word.isalpha() or word.isupper() or len(word) < 4:
                corrected_words.append(word)
                continue


            processed_tokens = preprocess(word)
            
            is_valid_medical_term = False
            if self.dataset_vocabulary is not None and processed_tokens:

                if processed_tokens[0] in self.dataset_vocabulary:
                    is_valid_medical_term = True

            if is_valid_medical_term:
                corrected_words.append(word)
                continue


            blob = TextBlob(word)
            corrected_word = str(blob.correct())
            corrected_words.append(corrected_word)

        corrected_query = " ".join(corrected_words)
        
        if query != corrected_query:
            print(f"[Query Refinement] تم تصحيح الاستعلام من: '{query}' إلى: '{corrected_query}'")
            
        return corrected_query

    def expand_query_with_synonyms(self, query: str, max_synonyms_per_word: int = 1) -> str:
        if not self.dataset_vocabulary:
            return query 
            
        words = query.split()
        original_query = query 
        new_synonyms: List[str] = [] 

        for word in words:
            lower_word = word.lower()
            
            if not lower_word.isalpha():
                continue

            word_emb = self._get_or_compute_embedding(lower_word)
            word_norm = norm(word_emb)
            
            if word_norm == 0:
                 continue

            from nltk.corpus import wordnet
            synonyms = []
            for syn in wordnet.synsets(lower_word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != lower_word and synonym.isalpha():
                        synonyms.append(synonym)
            
            medical_context_emb = self._get_or_compute_embedding("medical disease virus")
            med_norm = norm(medical_context_emb)
            
            valid_synonyms = []
            for syn in set(synonyms):

                 syn_processed = preprocess(syn)
                 if syn_processed and syn_processed[0] in self.dataset_vocabulary:
                     syn_emb = self._get_or_compute_embedding(syn)
                     syn_norm = norm(syn_emb)
                     if syn_norm > 0:
                         medical_similarity = np.dot(syn_emb, medical_context_emb) / (syn_norm * med_norm)
                         word_similarity = np.dot(syn_emb, word_emb) / (syn_norm * word_norm)
                         
                         if medical_similarity > 0.15 and word_similarity > 0.4:
                             valid_synonyms.append((syn, word_similarity))
            
            valid_synonyms.sort(key=lambda x: x[1], reverse=True)
            for syn, _ in valid_synonyms[:max_synonyms_per_word]:
                 if syn not in words and syn not in new_synonyms:
                     new_synonyms.append(syn)

        if new_synonyms:
            expanded_query = original_query + " " + " ".join(new_synonyms)
            print(f"[Query Refinement] تم التوسيع (بإضافة المرادفات في النهاية): '{expanded_query}'")
            return expanded_query
            
        return query

    def refine_query(self, query: str, use_spell_check: bool = True, use_synonyms: bool = True) -> str:
        refined_query = query
        if use_spell_check:
            refined_query = self.correct_spelling(refined_query)
        if use_synonyms:
            refined_query = self.expand_query_with_synonyms(refined_query)
        return refined_query