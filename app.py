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
    

def main():
    print(f"Initialize Ollama")

    rag = OllamaRAG(model_name="llama3.2:latest")

    if not rag.check_ollama_connection():
        print("Cannot connect to Ollama. Make sure it's running on localhost:11434")
        return 
    
    print("Connected to Ollama successfully")

    sample_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "metadata": {"topic": "AI", "subtopic": "machine learning"}
        },
        {
            "content": "RAG (Retrieval-Augmented Generation) combines the power of large language models with external knowledge retrieval to provide more accurate and contextual responses.",
            "metadata": {"topic": "AI", "subtopic": "RAG"}
        }
    ]

    print("\n Adding sample documents...")
    for doc in sample_docs:
        doc_id = rag.add_document(doc["content"], doc["metadata"])
        print(f"Added document: {doc_id}")
    
    # Interactive menu loop
    print("\n🔍 RAG System Ready!")

    while True: 
        print("\n" + "="*50)
        print("MENU OPTIONS:")
        print("1. Add PDF file")
        print("2. Add text file (.txt, .md, .py, .js, .json, .csv, .html, .xml)")
        print("3. Add entire directory")
        print("4. List all documents")
        print("5. Ask a question")
        print("6. Quit")
        print("="*50)

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

                doc_id = rag.add_document(content, metadata)
                print(f"Added PDF document: {pdf_file.name} (ID: {doc_id})")

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

                print("Reader text file...")
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metadata = {
                    'filename': text_file.name,
                    'filepath': str(text_file),
                    'extension': text_file.suffix,
                    'file_type': 'text'
                }

                doc_id = rag.add_document(content, metadata)
                print(f"Added text document: {text_file.name} (ID: {doc_id})")

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
                for i, doc in enumerate(docs, 1):
                    print(f"  {i}. ID: {doc['id']}")
                    print(f"     Metadata: {doc['metadata']}")
                    print(f"     Created: {doc['created_at']}")
                    print()

        
        elif choice == "5":
            # Ask a question 
            question = input("Enter your question: ").strip()
            if not question:
                print("No question provided")
                continue

            print(f"\nSearching for relevant information...")
            result = rag.query(question)

            print(f"\n Answer: \n{result['answer']}")

            if result['sources']:
                print(f"\n Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. Similarity: {source['similarity']:.3f}")
                    print(f"     Metadata: {source['metadata']}")
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
