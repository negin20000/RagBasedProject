# RAG Project - Retrieval-Augmented Generation System (phase1)

## Project Structure

```
RAGproject/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Load documents from CSV
│   ├── embedding.py        # Manage embedding models
│   ├── vectorstore.py      # Manage ChromaDB
│   ├── search.py           # Main RAG system
│   └── ingest.py           # Build vector database
├── data/
│   ├── *.csv               # Data files
│   └── chroma_db/          # Vector database (generated)
├── config.py               # Project configuration
├── requirements.txt        # Dependencies
├── env.example             # Environment variables template
└── README.md               # This file
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp env.example .env
```

Then edit the `.env` file and add your API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Setup Ollama (for Embedding)


```bash
ollama pull qwen3-embedding:0.6b
```

### 4. Build Vector Database

Before using the system, you need to build the vector database:

```bash
python -m src.ingest
```

Or:

```python
from src.ingest import create_database
create_database()
```

## Usage

### Simple Example

```python
from src.search import RAGSystem

# Create RAG system
rag = RAGSystem()

# Ask a question
answer = rag.ask("I have dry and flaky skin")
print(answer)
```

### Advanced Usage

```python
from src.search import RAGSystem

# With custom settings
rag = RAGSystem(
    google_api_key="your_key",  # or read from .env
)

# Option 1: Just ask a question (recommended)
# The system automatically:
# 1. Searches your documents in the vector database
# 2. Extracts relevant context
# 3. Sends context + question to LLM
# 4. Returns the answer
answer = rag.ask("بهترین محصول برای پوست چرب چیه ؟")
print(answer)

# Option 2: Get context only (for debugging/inspection)
# This searches the vector database and returns matching documents
# You can see what documents were found before sending to LLM
context = rag.retrieve_context("moisturizing cream", k=5)
print(context)  # Shows the documents that match your query
```

(or you could just run test.py)

## Configuration

All settings can be modified in `config.py`. You can also use environment variables:

### Important Settings

- `EMBEDDING_MODEL_NAME`: Embedding model name (default: `qwen3-embedding:0.6b`)
- `LLM_MODEL_NAME`: LLM model name (default: `gemini-2.0-flash`)
- `LLM_TEMPERATURE`: Model temperature (default: `0.7`)
- `DEFAULT_TOP_K`: Default number of results (default: `3`)
- `COLLECTION_NAME`: Collection name in ChromaDB
- `PROXY_URL`: Proxy address (optional)

## Code Structure

### Modules

#### `data_loader.py`
Loads CSV files and converts them to Document objects.

#### `embedding.py`
Manages embedding models using Ollama.

#### `vectorstore.py`
Manages ChromaDB operations including:
- Database creation
- Loading existing database
- Similarity search
- Converting to retriever

#### `search.py`
Main RAG system that:
- Retrieves context from vector store
- Combines with LLM
- Generates final response

#### `ingest.py`
Script for building database from CSV files.

## Error Handling

All modules use proper exception handling:
- `ValueError`: For invalid inputs
- `RuntimeError`: For runtime errors
- `FileNotFoundError`: For missing files

## Important Notes

1. **API Key**: Never hardcode API keys in your code. Use the `.env` file.
2. **Database**: Vector database is stored in `data/chroma_db/`. This folder is in `.gitignore`.
3. **Encoding**: CSV files should use `utf-8-sig` encoding (for Persian/Farsi and Excel support).
4. **Metadata**: CSV files must have `name` and `category` columns.

## Development

### Adding New Files

To add a new CSV file, place it in the `data/` folder and rebuild the database:

```bash
python -m src.ingest
```

### Changing Embedding Model

In `config.py` or `.env`:

```python
EMBEDDING_MODEL_NAME=your-model-name
```

### Changing LLM Model

```python
LLM_MODEL_NAME=your-llm-model
```

## Common Issues

### Error: "Vector database not initialized"
- Make sure you've built the database with `python -m src.ingest`.

### Error: "Google API key is required"
- Create a `.env` file and set `GOOGLE_API_KEY`.

### Error: "No CSV files found"
- Make sure CSV files are in the `data/` folder.


