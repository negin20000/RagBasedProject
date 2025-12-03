"""
Embedding pipeline module for generating vector embeddings.
"""
from typing import Optional
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from config import EMBEDDING_MODEL_NAME


class EmbeddingPipeline:
    """
    Wrapper class for Ollama embedding models.
    
    This class initializes and manages the embedding model used for
    converting text documents into vector representations.
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: Name of the Ollama embedding model to use.
                       Defaults to EMBEDDING_MODEL_NAME from config.
        
        Raises:
            RuntimeError: If the model fails to initialize.
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        try:
            self.model: Embeddings = OllamaEmbeddings(model=self.model_name)
            print(f"[INFO] Initialized Ollama embedding model: {self.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Ollama model '{self.model_name}': {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}") from e
