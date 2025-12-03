"""
Data loader module for loading CSV documents.
"""
from pathlib import Path
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from config import DATA_DIR, CSV_ENCODING, CSV_METADATA_COLUMNS


def load_all_documents(data_dir: str = None) -> List[Document]:
    """
    Load all CSV files from the specified directory.
    
    Uses LangChain's CSVLoader with UTF-8-sig encoding for Excel compatibility
    and extracts metadata columns (name, category).
    
    Args:
        data_dir: Directory path containing CSV files. 
                 If None, uses DATA_DIR from config.
                 
    Returns:
        List of Document objects loaded from CSV files.
        Returns empty list if no CSV files found or on error.
    """
    data_path = Path(data_dir) if data_dir else DATA_DIR
    data_path = data_path.resolve()
    
    if not data_path.exists():
        print(f"[WARN] Data directory does not exist: {data_path}")
        return []
    
    documents: List[Document] = []
    csv_files = list(data_path.glob("**/*.csv"))
    
    if not csv_files:
        print(f"[WARN] No CSV files found in {data_path}")
        return []
    
    print(f"[INFO] Found {len(csv_files)} CSV file(s) to process")
    
    for csv_file in csv_files:
        try:
            loader = CSVLoader(
                file_path=str(csv_file),
                encoding=CSV_ENCODING,
                metadata_columns=CSV_METADATA_COLUMNS
            )
            
            loaded = loader.load()
            print(f"[INFO] Loaded {len(loaded)} document(s) from {csv_file.name}")
            documents.extend(loaded)
            
        except Exception as e:
            print(f"[ERROR] Failed to load CSV file '{csv_file.name}': {e}")
            continue
    
    print(f"[INFO] Total documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    # Simple test
    docs = load_all_documents()
    if docs:
        print(f"[INFO] Sample metadata: {docs[0].metadata}")
