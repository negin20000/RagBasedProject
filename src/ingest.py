"""
Data ingestion module for building the vector database.
"""
from .data_loader import load_all_documents
from .vectorstore import ChromaVectorStore
from config import DATA_DIR, EMBEDDING_MODEL_NAME


def create_database(data_folder: str = None, embedding_model: str = None) -> None:
    """
    Create and build the vector database from CSV files.
    
    This function:
    1. Loads all CSV files from the data directory
    2. Initializes the vector store
    3. Builds the database from loaded documents
    
    Args:
        data_folder: Path to the data directory containing CSV files.
                    If None, uses DATA_DIR from config.
        embedding_model: Name of the embedding model to use.
                        If None, uses EMBEDDING_MODEL_NAME from config.
    
    Raises:
        FileNotFoundError: If data folder does not exist.
        ValueError: If no documents are found.
        RuntimeError: If database creation fails.
    """
    data_path = data_folder or str(DATA_DIR)
    model_name = embedding_model or EMBEDDING_MODEL_NAME
    
    print("[INFO] Starting database creation process...")
    
    # Load documents
    docs = load_all_documents(data_path)
    
    if not docs:
        error_msg = f"No documents found in {data_path}. Database creation cancelled."
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    print(f"[INFO] Loaded {len(docs)} document(s) for indexing")
    
    # Initialize vector store
    try:
        store = ChromaVectorStore(embedding_model=model_name)
        print("[INFO] Vector store initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector store: {e}")
        raise RuntimeError(f"Vector store initialization failed: {e}") from e
    
    # Build database
    try:
        store.build_from_documents(docs)
        print("[INFO] Database creation completed successfully")
    except Exception as e:
        print(f"[ERROR] Failed to build database: {e}")
        raise RuntimeError(f"Database build failed: {e}") from e


if __name__ == "__main__":
    try:
        create_database()
    except Exception as e:
        print(f"[ERROR] Database creation failed: {e}")
        raise
