"""
Vector store module for managing ChromaDB vector database.
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from .embedding import EmbeddingPipeline
from config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    VECTOR_STORE_SEARCH_TYPE
)


class ChromaVectorStore:
    """
    Wrapper class for ChromaDB vector store operations.
    
    Handles document storage, retrieval, and similarity search
    using ChromaDB as the underlying vector database.
    """
    
    def __init__(
        self, 
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> None:
        """
        Initialize the ChromaDB vector store.
        
        Args:
            persist_dir: Directory path for persisting the database.
                        If None, uses CHROMA_DB_DIR from config.
            embedding_model: Name of the embedding model to use.
                            If None, uses EMBEDDING_MODEL_NAME from config.
        """
        self.persist_dir = Path(persist_dir) if persist_dir else CHROMA_DB_DIR
        self.persist_dir = self.persist_dir.resolve()
        
        # Initialize embedding pipeline
        model_name = embedding_model or EMBEDDING_MODEL_NAME
        self.emb_pipeline = EmbeddingPipeline(model_name=model_name)
        
        self.vector_db: Optional[Chroma] = None
        
        # Load existing database if it exists
        if self.persist_dir.exists():
            self.load()
        else:
            print(f"[INFO] Vector database directory does not exist: {self.persist_dir}, please build the database first(python -m src.ingest)")
    
    def build_from_documents(self, documents: List[Document]) -> None:
        """
        Build a new vector database from documents.
        
        This method will delete any existing database at the persist_dir
        location before creating a new one.
        
        Args:
            documents: List of Document objects to index.
            
        Raises:
            ValueError: If documents list is empty.
            RuntimeError: If database creation fails.
        """
        if not documents:
            raise ValueError("Cannot build database from empty document list")
        
        # Delete existing database
        if self.persist_dir.exists():
            try:
                shutil.rmtree(self.persist_dir)
                print(f"[INFO] Deleted existing database at {self.persist_dir}")
            except Exception as e:
                print(f"[WARN] Could not delete existing database: {e}")
        
        print(f"[INFO] Building ChromaDB from {len(documents)} document(s)...")
        
        try:
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.emb_pipeline.model,
                persist_directory=str(self.persist_dir),
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print("[INFO] ChromaDB built and saved successfully")
        except Exception as e:
            print(f"[ERROR] Failed to build ChromaDB: {e}")
            raise RuntimeError(f"Database creation failed: {e}") from e
    
    def load(self) -> None:
        """
        Load an existing vector database.
        
        Raises:
            FileNotFoundError: If database directory does not exist.
            RuntimeError: If database loading fails.
        """
        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"Vector database directory not found: {self.persist_dir}"
            )
        
        try:
            self.vector_db = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.emb_pipeline.model,
                collection_name=COLLECTION_NAME
            )
            print("[INFO] Vector database loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load vector database: {e}")
            raise RuntimeError(f"Database loading failed: {e}") from e
    
    def get_retriever(self, k: int = None) -> VectorStoreRetriever:
        """
        Get a LangChain retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve. Defaults to DEFAULT_TOP_K from config.
            
        Returns:
            VectorStoreRetriever instance.
            
        Raises:
            RuntimeError: If vector database is not initialized.
        """
        if not self.vector_db:
            self.load()
        
        k = k or DEFAULT_TOP_K
        return self.vector_db.as_retriever(
            search_type=VECTOR_STORE_SEARCH_TYPE,
            search_kwargs={"k": k}
        )
    
    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query_text: Query string to search for.
            top_k: Number of results to return. Defaults to DEFAULT_TOP_K from config.
            
        Returns:
            List of dictionaries containing:
            - metadata: Document metadata
            - content: Document content
            - score: Similarity score (0.0 to 1.0)
            
        Raises:
            RuntimeError: If vector database is not initialized.
            ValueError: If query_text is empty.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if not self.vector_db:
            print("[ERROR] Vector database is not initialized")
            raise RuntimeError("Vector database not initialized. Call load() or build_from_documents() first.")
        
        top_k = top_k or DEFAULT_TOP_K
        print(f"[INFO] Querying vector store: '{query_text}' (top_k={top_k})")
        
        try:
            results = self.vector_db.similarity_search_with_score(query_text, k=top_k)
            
            formatted_results: List[Dict[str, Any]] = []
            for doc, distance in results:
                # Convert distance to similarity score (0.0 to 1.0)
                # ChromaDB uses cosine distance, so we normalize it
                similarity_score = 1.0 - (distance / 2.0)
                similarity_score = max(0.0, min(1.0, similarity_score))
                
                formatted_results.append({
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                    "score": similarity_score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            raise
