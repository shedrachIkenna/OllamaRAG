import os
import json
import requests
import numpy as np
import sqlite3
import re
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

class OllamaRAG:
    def __init__(self, model_name="llama3.2:latest", embedding_model="nomic_embed_text"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_url = "http://localhost:11434"
        self.db_path = "rag_database.db"
        self.embedding_dim = 348
        self.init_database()
        self.setup_embedding_model()

    def setup_embedding_model(self):
        """Check for available embedding models on the system"""
        embedding_models = [
            "nomic-embed-text",
            "mxbai-embed-large", 
            "all-minilm"
        ]

        self.embedding_model = None 

        try: 
            # Check what models are already installed 
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                installed_models = response.json().get("models", [])
                installed_model_names = [model.get("name", "").split(":")[0] for model in installed_models]
                print(f"Found installed models: {', '.join(installed_model_names)}")

                # Check if any of our preferred embedding models are installed 
                for model in embedding_models:
                    if model in installed_model_names:
                        self.embedding_model = model
                        print(f"Using embedding model: {model}")
                        return
                    
                # Check for any model with "embed" in the name exists 
                for model_name in installed_model_names:
                    if "embed" in model_name.lower():
                        self.embedding_model = model_name
                        print(f"Using embedding model: {model_name}")
                        return 
                    
        except Exception as e:
            print(f"Error Checking for embedding models: {e}")

        if not self.embedding_model:
            print("No embedding model found. Install one with: ")
            print("  ollama pull nomic-embed-text")
            print("  \n Using text similarity fallback instead...")


    def init_database(self):
        """Initialize SQLite database for stroring documents and embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
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
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        # Clean the text 
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0

        while start <= len(text):
            # Finding a good break point (end of sentence or paragraph)
            end = start + chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            # Break at sentence end 
            break_point = text.rfind('.', start, end)
            if break_point == -1:
                break_point = text.rfind(' ', start, end)
            if break_point == -1: 
                break_point = end 

            chunk = text[start:break_point + 1].strip()
            if chunk:
                chunks.append(chunk)

            # Move starting point with overlap
            start = break_point + 1 - overlap
            if start <= 0:
                start = break_point + 1
        
        return [chunk for chunk in chunks if chunk.strip()]
        
    
    def get_embedding(self, text:str) -> List[float]:
        """Get embeddings for text using Ollama with better fallback"""
        if self.embedding_model:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    print(f"Got embedding of size {len(embedding)}")
                    return embedding
                    
            except Exception as e:
                print(f"Embedding API error: {e}")

        # Fallback using TF-IDF style 
        return self._create_text_embedding(text)
    
    def _create_text_embedding(self, text: str) -> List[float]:
        """Text-based embedding using words frequency"""
        # Clean and tokenize test
        words = re.findall(r'\b\w+\b', text.lower())

        # Create a vocabulary from common English words + document words 
        vocab = set(words)
        vocab_list = sorted(list(vocab))

        # Create simple term frequency vector 
        embedding = np.zeros(self.embedding_dim)

        for i, word in enumerate(vocab_list[:self.embedding_dim]):
            # Term frequency 
            tf = words.count(word) / len(words) if words else 0
            embedding[i] = tf

        # Add some semantic features 
        if len(words) > 0:
            avg_word_len = np.mean([len(word) for word in words])
            embedding[-1] = avg_word_len / 10 #normalize 

        # Normalize the vector 
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()


    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> List[str]:
        """Add a document to the RAG database with chunking"""
        document_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        chunks = self.chunk_text(content)

        chunk_ids = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print(f"Processing document into {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            embedding = self.get_embedding(chunk)

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_id': document_id
            })

            cursor.execute('''
                INSERT OR REPLACE INTO document_chunks (id, document_id, chunk_index, content, metadata, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                chunk_id,
                document_id,
                i,
                chunk,
                json.dumps(chunk_metadata),
                json.dumps(embedding)
            ))
            
            chunk_ids.append(chunk_id)
            print(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
        
        conn.commit()
        conn.close()

        return chunk_ids


    def add_documents_from_directory(self, directory_path: str):
        """ Add all supported files from a directory (including PDFs)"""
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

        cursor.execute('SELECT id, content, metadata, embedding FROM document_chunks')
        chunks = cursor.fetchall()
        conn.close()

        results = []
        for chunk_id, content, metadata, embedding_json in chunks:
            chunk_embedding = json.loads(embedding_json)
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)

            results.append({
                'id': chunk_id,
                'content': content,
                'metadata': json.loads(metadata),
                'similarity': similarity,
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Ollama with retrieved context"""
        # Prepare context from retrieved documents 
        context_parts = []
        for i, doc in enumerate(context_docs):
            context_parts.append(f"Context {i+1} (similarity: {doc['similarity']:.3f}):\n{doc['content']}")
        context = "\n\n".join(context_parts)

        prompt = f"""
            Based on the following context documents, Answer the question(s). The context includes similarity scores - higher scores mean more relevant information 
            Context: {context}
            Question: {query}
            Please provide a detailed answer based on the context above. If information is not sufficient, please say so.
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
        relevant_chunks = self.search_document(question, top_k)

        if not relevant_chunks:
            return {
                "Answer": "No relevant documents found in the database",
                "Sources": []
            }
        
        # Filter out chunks with very low similarity (likely noise)
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk['similarity'] > -0.5]

        if not relevant_chunks:
            return {
                "Answer": "No relevant documents found in the database",
                "Sources": []
            }

        answer = self.generate_response(question, relevant_chunks)

        return {
            "answer": answer,
            "sources": [
                {
                    "id": chunk["id"],
                    "similarity": chunk["similarity"],
                    "metadata": chunk["metadata"],
                    "preview": chunk["content"][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
                } for chunk in relevant_chunks
            ]
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT document_id, metadata, COUNT(*) as chunk_count, MIN(created_at) as created_at
            FROM document_chunks
            GROUP BY document_id
        ''')
        documents = cursor.fetchall()
        conn.close()

        return [
            {
                "document_id": doc_id,
                "metadata": json.loads(metadata),
                "chunk_count": chunk_count,
                "created_at": created_at
            } for doc_id, metadata, chunk_count, created_at in documents
        ]
    

def main():
    print(f"Initializing Ollama")

    rag = OllamaRAG(model_name="llama3.2:latest")

    if not rag.check_ollama_connection():
        print("Cannot connect to Ollama. Make sure it's running on localhost:11434")
        return 
    
    print("Connected to Ollama successfully")

    while True: 
        print("\n" + "="*60)
        print("MENU OPTIONS:")
        print("1. Add PDF file")
        print("2. Add text file (.txt, .md, .py, .js, .json, .csv, .html, .xml)")
        print("3. Add entire directory")
        print("4. List all documents")
        print("5. Ask a question")
        print("6. Quit")
        print("="*60)

        choice = input("Select option (1-6): ").strip()

        if choice == "1":
            # Add PDF file 
            pdf_path = input("Enter PDF file path: ").strip()
            if not pdf_path:
                print("No file path provided")
                continue

            try:
                pdf_file = Path(pdf_path)
                if not pdf_file.exists():
                    print(f"File not found at: {pdf_path}")
                    continue

                if pdf_file.suffix.lower() != '.pdf':
                    print("File is not a PDF")
                    continue

                print("Extracting text from pdf...")
                content = rag._extract_pdf_text(pdf_file)
                if not content.strip():
                    print("No text could be extracted from the PDF")
                    continue

                metadata = {
                    'filename': pdf_file.name,
                    'filepath': str(pdf_file),
                    'extension': '.pdf',
                    'file_type': 'pdf'
                }

                chunk_ids = rag.add_document(content, metadata)
                print(f"Added PDF: {pdf_file.name} ({len(chunk_ids)} chunks)")

            except Exception as e:
                print(f"Error adding PDF: {e}")
        
        elif choice == "2":
            # Add text file 
            file_path = input("Enter text file path: ").strip()
            if not file_path:
                print("No file path provided")
                continue

            try:
                text_file = Path(file_path)
                if not text_file.exists():
                    print(f"File not found at {file_path}")
                    continue

                supported_extensions = ['.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml']
                if text_file.suffix.lower() not in supported_extensions:
                    print(f"Unsupported file type. Supported: {', '.join(supported_extensions)}")
                    continue

                print("Read text file...")
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metadata = {
                    'filename': text_file.name,
                    'filepath': str(text_file),
                    'extension': text_file.suffix,
                    'file_type': 'text'
                }

                chunk_ids = rag.add_document(content, metadata)
                print(f"Added text file: {text_file.name} ({len(chunk_ids)} chunks)")

            except Exception as e:
                print(f"Error adding text file: {e}")

        elif choice == "3":
            # Add entire directory 
            directory = input("Enter directory path: ").strip()
            if not directory:
                print("No directory provided")
                continue

            try:
                dir_path = Path(directory)
                if not dir_path.exists() or not dir_path.is_dir():
                    print(f"Directory not found at: {directory}")
                    continue

                print(f"Processing all files in: {directory}")
                rag.add_documents_from_directory(directory)
                print("Finished processing files in directory")
            
            except Exception as e:
                print(f"Error processing directory: {e}")

        elif choice == "4":
            # List documents 
            docs = rag.list_documents()
            if not docs:
                print("No documents in database")
            else:
                print(f"\n Documents in database: {len(docs)}")
                for doc in docs:
                    print(f"     {doc['metadata'].get('filename', 'Unknown')}")
                    print(f"     Chunks: {doc['chunk_count']}, Created: {doc['created_at']}")
                    print()

        
        elif choice == "5":
            # Ask a question 
            question = input("Enter your question: ").strip()
            if not question:
                print("No question provided")
                continue

            print(f"\nSearching for relevant information...")
            result = rag.query(question, top_k=5)

            print(f"\n Answer: \n{result['answer']}")

            if result['sources']:
                print(f"\n Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. Similarity: {source['similarity']:.3f}")
                    print(f"     File: {source['metadata'].get('filename', 'Unknown')}")
                    print(f"     Preview: {source['preview']}")
                    print()
        
        elif choice == "6":
            # End while loop 
            break
        
        else:
            print("Invalid option. Please select 1-6")
    
    print("\n See you next time mate!")



if __name__ == "__main__":
    main()
