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
    
    def add_documents_from_directory(self, directory_path: str):
        """ Add all text files from a directory (including PDFs)"""
        directory = Path(directory_path)
        text_extentions = ['.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml']
        pdf_extentions = ['.pdf']

        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue

            file_ext = file_path.suffix.lower()

            try:
                if file_ext in text_extentions:
                    # Handle text based files 
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                elif file_ext in pdf_extentions:
                    # Handle PDF files 
                    content = self._extract_pdf_text(file_path)
                    if not content.strip():
                        print(f"Warning: No text extracted from {file_path.name}")
                        continue
                
                else:
                    continue # skip unsupported file types 

                metadata = {
                    'filename': file_path.name,
                    'filepath': str(file_path),
                    'extention': file_path.suffix,
                    'file_type': 'pdf' if file_ext == 'pdf' else 'text'
                }

                doc_id = self.add_document(content, metadata)
                print(f"Added Document: {file_path.name} (ID: {doc_id})")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file using multiple fallback methods"""
        try:
            # Method 1: Use PyMuPDF (fitz) 
            try: 
                import fitz 
                doc = fitz.open(str(pdf_path))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                pass

            # Method 2: Use PyPDF2 
            try: 
                import PyPDF2 
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                pass

            # Method 3: Use pdfplumber 
            try:
                import pdfplumber
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
                
                return text
            except ImportError:
                pass

            # Method 4: Use Pdfminer 
            try: 
                from pdfminer.high_level import extract_text
                return extract_text(str(pdf_path))
            except ImportError:
                pass

            print(f"No PDF libraries available. Install one of: pymupdf, PyPDF2, pdfplumber, or pdfminer.six")
            return ""
        except Exception as e:
            print(f"Error extracting PDF text from {pdf_path.name}: {e}")
            return ""
        
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Cosine Similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        magnitude1 = np.linalg.norm(vec1_np)
        magnitude2 = np.linalg.norm(vec2_np)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0 
        
        return dot_product / (magnitude1 * magnitude2)

    
    def search_document(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity"""
        query_embedding = self.get_embedding(query)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id, content, metadata, embedding FROM documents')
        documents = cursor.fetchall()
        conn.close()

        results = []
        for doc_id, content, metadata, embedding_json in documents:
            doc_embedding = json.loads(embedding_json)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)

            results.append({
                'id': doc_id,
                'content': content,
                'metadata': json.loads(metadata),
                'similarity': similarity,
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Ollama with retrieved context"""
        # Prepare context from retrieved documents 
        context = "\n\n".join([
            f"Document {i+1}: \n{doc['content'][:1000]}..." for i, doc in enumerate(context_docs)
        ])

        prompt = f"""
            Based on the following context documents, Answer the question(s).
            Context: {context}
            Question: {query}
            Answer:
        """

        try: 
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temparature': 0.7,
                        'top_p': 0.9
                    }
                }
            )

            if response.status_code == 200:
                return response.json()["response"]
            
            else:
                return f"Error generating response: {response.status_code}"
            
        except Exception as e:
            return f"Error connecting to Ollama: {e}"
        

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Main RAG query function"""
        # Retrieve relevant documents 
        relevant_docs = self.search_document(question, top_k)

        if not relevant_docs:
            return {
                "Answer": "No relevant documents found in the database",
                "Sources": []
            }
        
        # Generate response 
        answer = self.generate_response(question, relevant_docs)

        return {
            "answer": answer,
            "sources": [
                {
                    "id": doc["id"],
                    "similarity": doc["similarity"],
                    "metadata": doc["metadata"],
                    "preview": doc["content"][:250] + "..."
                } for doc in relevant_docs
            ]
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT id, metadata, created_at FROM documents')
        documents = cursor.fetchall()
        conn.close()

        return [
            {
                "id": doc_id,
                "metadata": json.loads(metadata),
                "created_at": created_at
            } for doc_id, metadata, created_at in documents
        ]
    

