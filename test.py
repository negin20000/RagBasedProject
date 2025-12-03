from src.search import RAGSystem

# With custom settings
rag = RAGSystem()
    

# Option 1: Just ask a question (recommended)
# The system automatically:
# 1. Searches your documents in the vector database
# 2. Extracts relevant context
# 3. Sends context + question to LLM
# 4. Returns the answer
answer = rag.ask("بهترین محصول برای پوست چرب چیست؟")
print(answer)

# Option 2: Get context only (for debugging/inspection)
# This searches the vector database and returns matching documents
# You can see what documents were found before sending to LLM
sample_context = rag.retrieve_context("پوست خشک و پوسته پوسته", k=5)
print(sample_context)  # Shows the documents that match your query