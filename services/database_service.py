import sqlite3
import json
import numpy as np
from typing import Dict, List, Tuple, Any

class DatabaseService:
    def __init__(self, db_name: str = "ir_system.db") -> None:

        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_tables()

    def create_tables(self) -> None:
        cursor = self.conn.cursor()
        

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                content TEXT,
                preprocessed_text TEXT
            )
        """)
        

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inverted_index (
                term TEXT PRIMARY KEY,
                postings TEXT
            )
        """)


        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding_blob BLOB
            )
        """)
        
        self.conn.commit()

    def save_documents_batch(self, docs_batch: List[Tuple[str, str, str, str]]) -> None:
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO documents (doc_id, title, abstract, content, preprocessed_text)
            VALUES (?, ?, ?, '', ?)
        """, docs_batch)
        self.conn.commit()

    def save_index_batch(self, index_batch: List[Tuple[str, str]]) -> None:
        cursor = self.conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO inverted_index (term, postings) VALUES (?, ?)", index_batch)
        self.conn.commit()

    def save_embeddings_batch(self, embeddings_batch: List[Tuple[str, bytes]]) -> None:

        cursor = self.conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO embeddings (doc_id, embedding_blob) VALUES (?, ?)", embeddings_batch)
        self.conn.commit()

    def get_index(self) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT term, postings FROM inverted_index")
        rows = cursor.fetchall()
        
        existing_index: Dict[str, Dict[str, int]] = {}
        if rows:
            existing_index = {row[0]: json.loads(row[1]) for row in rows}
            
        doc_lengths: Dict[str, int] = {}
        cursor.execute("SELECT doc_id, preprocessed_text FROM documents")
        for doc_row in cursor.fetchall():
            preprocessed_text: str = doc_row[1] if doc_row[1] else ""
            doc_lengths[doc_row[0]] = len(preprocessed_text.split())
            
        return existing_index, doc_lengths

    def get_all_embeddings(self) -> Dict[str, Any]:

        cursor = self.conn.cursor()
        cursor.execute("SELECT doc_id, embedding_blob FROM embeddings")
        rows = cursor.fetchall()
        
        embeddings_dict: Dict[str, Any] = {}
        for row in rows:
            doc_id = row[0]
            emb_bytes = row[1]
            embeddings_dict[doc_id] = np.frombuffer(emb_bytes, dtype=np.float32)
            
        return embeddings_dict