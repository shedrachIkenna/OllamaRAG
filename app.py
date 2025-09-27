import os 
import requests
import json 
import numpy as np 
from typing import List, Dict, Any 
import sqlite3
from pathlib import Path 
import hashlib
from datetime import datetime 

class OllamaRAG:
    def __init__(self, model_name="llama3.2:latest", embedding_model="nomic_embed_text"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_url = "http://localhost:11434"
        self.db_path = "rag_database.db"
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for stroring documents and embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )        
        ''')

        conn.commit()
        conn.close() 

    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_embedding(self, text:str) -> List[float]:
        """Get embeddings for text using Ollama"""
        # First try with nomic-embed-text if available 
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            if response.status_code == 200:
                return response.json()["embedding"]
        except:
            pass

        # Fallback: Use the main model for embeddings (less optimal but works)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )

            if response.status_code == 200:
                return response.json()["embeddings"]
        except:
            # if embeddings API doesn't work, create a simple hash-based embedding
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text:str) -> List[float]:
        """Create a simple hash-based embedding as fallback"""
        # This is a very basic fallback - in production, use proper embeddings 
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Convert to a simple 768-dimentional vector 
        np.random.seed(hash_int % (2**32))
        embedding = np.random.normal(0, 1, 768).tolist()
        return embedding

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the RAG database"""
        doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        embedding = self.get_embedding(content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor

        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, contents, metadata, embedding)
            VALUES(?, ?, ?, ?)
            ''', (
                doc_id, 
                content,
                json.dumps(metadata or {}),
                json.dumps(embedding)
            ))
        
        conn.commit()
        conn.close()

        return doc_id
