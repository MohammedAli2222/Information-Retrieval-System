import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
from typing import List, Set

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class QueryRefinementService:
   

    def __init__(self):

        pass

    def correct_spelling(self, query: str) -> str:
    
        blob = TextBlob(query)

        corrected_query = str(blob.correct())
        
        if query != corrected_query:
            print(f"[Query Refinement] تم تصحيح الاستعلام من: '{query}' إلى: '{corrected_query}'")
            
        return corrected_query

    def expand_query_with_synonyms(self, query: str, max_synonyms_per_word: int = 2) -> str:
        
        
        words = query.split()
        expanded_terms: Set[str] = set(words) 

        for word in words:
            synonyms = []

            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ') 
                    if synonym.lower() != word.lower():
                        synonyms.append(synonym.lower())
            

            unique_synonyms = list(set(synonyms))[:max_synonyms_per_word]
            for syn in unique_synonyms:
                expanded_terms.add(syn)

        expanded_query = " ".join(expanded_terms)
        
        if query != expanded_query:
             print(f"[Query Refinement] تم توسيع الاستعلام إلى: '{expanded_query}'")
             
        return expanded_query

    def refine_query(self, query: str, use_spell_check: bool = True, use_synonyms: bool = True) -> str:
       
       
        refined_query = query

        if use_spell_check:
            refined_query = self.correct_spelling(refined_query)

        if use_synonyms:
            refined_query = self.expand_query_with_synonyms(refined_query)

        return refined_query