import re
from typing import List, Set, Optional, Dict, Any
from functools import lru_cache
import numpy as np
from numpy.linalg import norm
from textblob import TextBlob
from sentence_transformers import SentenceTransformer

import nltk
from nltk.corpus import wordnet
from services.preprocessing_service import preprocess

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


@lru_cache(maxsize=2000)
def _cached_spell_correct(word: str) -> str:
    return str(TextBlob(word).correct())


class QueryRefinementService:
    def __init__(
        self,
        dataset_vocabulary: Optional[Set[str]] = None,
        enable_spellcheck: bool = True,
        enable_synonyms: bool = True,
        max_synonyms_per_query: int = 4,
        max_synonyms_per_word: int = 1
    ):
        self.dataset_vocabulary = dataset_vocabulary
        self.enable_spellcheck = enable_spellcheck
        self.enable_synonyms = enable_synonyms
        self.max_synonyms_per_query = max_synonyms_per_query
        self.max_synonyms_per_word = max_synonyms_per_word
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vocab_embeddings_cache: Dict[str, Any] = {}

    def _get_embedding(self, text: str) -> Any:
        if text not in self.vocab_embeddings_cache:
            self.vocab_embeddings_cache[text] = self.embedding_model.encode(text)
        return self.vocab_embeddings_cache[text]

    def correct_spelling(self, query: str) -> str:
        if not self.enable_spellcheck:
            return query

        words = query.split()
        corrected_words = []

        for word in words:
            clean_word = re.sub(r"[^a-zA-Z0-9-]", "", word).lower()
            
            if word.isupper() or len(clean_word) <= 3:
                corrected_words.append(word)
                continue
                
            stemmed_word_list = preprocess(clean_word)
            if self.dataset_vocabulary and stemmed_word_list and stemmed_word_list[0] in self.dataset_vocabulary:
                corrected_words.append(word)
                continue

            corrected_word = _cached_spell_correct(word)
            clean_corrected = re.sub(r"[^a-zA-Z0-9-]", "", corrected_word).lower()

            stemmed_corrected_list = preprocess(clean_corrected)
            if self.dataset_vocabulary and stemmed_corrected_list and stemmed_corrected_list[0] not in self.dataset_vocabulary:
                corrected_words.append(word)
                continue

            if word.lower() != corrected_word.lower():
                print(f"[Spell Check Match] تم تصحيح خطأ بشري حقيقي: '{word}' -> '{corrected_word}'")
            
            corrected_words.append(corrected_word)

        return " ".join(corrected_words)

    def expand_query_with_synonyms(self, query: str) -> str:
        words = query.split()
        expanded = list(words) 

        for word in words:
            if self._should_skip_word(word):
                continue

            synonyms_added = 0
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ").lower()

                    if candidate == word.lower():
                        continue

                    if self._is_valid_synonym(word, candidate):
                        expanded.append(candidate)
                        synonyms_added += 1

                    if synonyms_added >= self.max_synonyms_per_word:
                        break
                if synonyms_added >= self.max_synonyms_per_word:
                    break

        expanded = expanded[:len(words) + self.max_synonyms_per_query]
        expanded_query = " ".join(expanded)
        return expanded_query

    def _should_skip_word(self, word: str) -> bool:
        return bool(re.match(r"^\d+$", word)) or len(word) <= 2

    def _is_valid_synonym(self, word: str, synonym: str) -> bool:
        if len(synonym.split()) > 2:
            return False
            
        if self.dataset_vocabulary:
            syn_processed = preprocess(synonym)
            if not syn_processed or syn_processed[0] not in self.dataset_vocabulary:
                return False

        try:
            word_emb = self._get_embedding(word.lower())
            syn_emb = self._get_embedding(synonym.lower())
            
            w_norm = norm(word_emb)
            s_norm = norm(syn_emb)
            
            if w_norm == 0 or s_norm == 0:
                return False
                
            similarity = np.dot(word_emb, syn_emb) / (w_norm * s_norm)
            
            return bool(similarity > 0.5)
        except Exception:
            return False

    def refine_query(self, query: str, use_spell_check: bool = True, use_synonyms: bool = True) -> str:
        refined = query
        if use_spell_check and self.enable_spellcheck:
            refined = self.correct_spelling(refined)
        if use_synonyms and self.enable_synonyms:
            refined = self.expand_query_with_synonyms(refined)
        return refined