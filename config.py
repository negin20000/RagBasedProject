"""
Configuration file for RAG Project.
All settings can be overridden via environment variables.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Model configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding:0.6b")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-flash-latest")

# LLM settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Vector Store settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products_collection")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
VECTOR_STORE_SEARCH_TYPE = os.getenv("VECTOR_STORE_SEARCH_TYPE", "similarity")

# API Keys (should be in .env file)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Proxy settings
PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:3067")
NO_PROXY = os.getenv("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

# CSV Loader settings
CSV_ENCODING = os.getenv("CSV_ENCODING", "utf-8-sig")
CSV_METADATA_COLUMNS = ["name", "category"]
