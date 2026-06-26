import os
import json
from typing import Dict, Tuple, Callable, Any, List
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from services.preprocessing_service import extract_text
from services.database_service import DatabaseService

class IndexingService:

    def __init__(self, db_service: DatabaseService) -> None:
        self.db_service = db_service
        self.embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    def build_inverted_index(self, dataset: Any, preprocess_fn: Callable[[str], List[str]]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        existing_index, doc_lengths = self.db_service.get_index()
        
        if existing_index:
            print("Loaded index and document lengths from SQLite Database.")
            return existing_index, doc_lengths

        print("Building index and saving to SQLite Database using High-Performance Batch Insertion...")
        inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        docs_batch: List[Tuple[str, str, str, str]] = []
        batch_size: int = 2500 
        processed_count: int = 0
        
        for doc in dataset.docs_iter():
            title: str = getattr(doc, "title", "")
            abstract: str = getattr(doc, "abstract", "")
            text: str = extract_text(doc)
            
            tokens: List[str] = preprocess_fn(text)
            doc_lengths[doc.doc_id] = len(tokens)
            
            preprocessed_text: str = " ".join(tokens)
            docs_batch.append((doc.doc_id, title, abstract, preprocessed_text))
            
            term_counts: Counter[str] = Counter(tokens)
            for term, freq in term_counts.items():
                inverted_index[term][doc.doc_id] = freq

            processed_count += 1
            
            if processed_count % batch_size == 0:
                self.db_service.save_documents_batch(docs_batch)
                print(f"[Indexing] Processed and saved {processed_count} documents...")
                docs_batch.clear()

        if docs_batch:
            self.db_service.save_documents_batch(docs_batch)
            print(f"[Indexing] Finished saving all {processed_count} documents.")

        print("Saving Inverted Index to SQLite in batches...")
        index_batch: List[Tuple[str, str]] = []
        term_count: int = 0
        
        for term, postings in inverted_index.items():
            index_batch.append((term, json.dumps(postings)))
            term_count += 1
            
            if term_count % 10000 == 0:
                self.db_service.save_index_batch(index_batch)
                index_batch.clear()
                
        if index_batch:
            self.db_service.save_index_batch(index_batch)

        print("Saved complete Inverted Index to SQLite Database.")
        return dict(inverted_index), doc_lengths

    def compute_documents_embeddings(self, dataset: Any, cache_name: str) -> Dict[str, Any]:

        existing_embeddings = self.db_service.get_all_embeddings()
        

        if len(existing_embeddings) > 1000:
            print(f"Loaded {len(existing_embeddings)} embeddings from SQLite Database.")
            return existing_embeddings
            
        print("جاري توليد الـ Embeddings بشكل مجمع (Batch Processing) لتفادي اختناق الذاكرة...")
        
        batch_size = 1000
        doc_ids_batch: List[str] = []
        texts_batch: List[str] = []
        total_processed = 0
        
        for doc in dataset.docs_iter():

            if doc.doc_id in existing_embeddings:
                continue

            text: str = getattr(doc, "title", "") + " " + getattr(doc, "abstract", "")
            if text.strip():
                doc_ids_batch.append(doc.doc_id)
                texts_batch.append(text)
                total_processed += 1
                

            if len(texts_batch) >= batch_size:
                print(f"Processing batch of {batch_size} documents. Total processed: {total_processed}...")
                embeddings = self.embedding_model.encode(texts_batch, show_progress_bar=False)
                
                db_batch = []
                for i, doc_id in enumerate(doc_ids_batch):

                    db_batch.append((doc_id, embeddings[i].tobytes()))
                
                self.db_service.save_embeddings_batch(db_batch)
                

                doc_ids_batch.clear()
                texts_batch.clear()
                

        if texts_batch:
            embeddings = self.embedding_model.encode(texts_batch, show_progress_bar=False)
            db_batch = []
            for i, doc_id in enumerate(doc_ids_batch):
                db_batch.append((doc_id, embeddings[i].tobytes()))
            self.db_service.save_embeddings_batch(db_batch)
            
        print("تم توليد وحفظ جميع الـ Embeddings في قاعدة البيانات بنجاح!")
        

        return self.db_service.get_all_embeddings()