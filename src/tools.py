# src/tools.py

from langchain_core.tools import tool
from src.vectorstore import ChromaVectorStore
from config import EMBEDDING_MODEL_NAME

# 1. اتصال به دیتابیس (یک بار لود می‌شود تا سریع باشد)
# نکته: ما اینجا فقط VectorStore را لود می‌کنیم، بقیه منطق با LLM در گراف هندل می‌شود.
try:
    vector_db = ChromaVectorStore(embedding_model=EMBEDDING_MODEL_NAME)
    vector_db.load()
    print("[INFO] Tools: Vector DB loaded successfully.")
except Exception as e:
    print(f"[WARN] Tools: Could not load Vector DB. Make sure to run ingest.py first. Error: {e}")
    vector_db = None

@tool
def search_products(query: str) -> str:
    """
    Search for beauty/skincare products in the inventory based on user requirements.
    Useful for answering questions about product availability, features, or recommendations.
    Input should be a descriptive search query (e.g., 'anti-aging cream for dry skin').
    """
    if not vector_db:
        return "Error: Database not loaded."
    
    # استفاده از همان متد query که در vectorstore.py دارید
    results = vector_db.query(query, top_k=3)
    
    if not results:
        return "No relevant products found."
        
    # فرمت کردن خروجی برای فهمیدن LLM
    formatted_results = []
    for doc in results:
        meta = doc['metadata']
        content = doc['content']
        formatted_results.append(
            f"Product Name: {meta.get('name', 'Unknown')}\n"
            f"Category: {meta.get('category', 'General')}\n"
            f"Details: {content}\n"
        )
    
    return "\n---\n".join(formatted_results)

@tool
def get_shipping_policy(query: str) -> str:
    """
    Retrieve information about shipping, returns, and general store policies.
    Useful when user asks about delivery time, return rules, or shipping costs.
    """
    # اینجا برای دمو یک متن ثابت برمی‌گردانیم. 
    # در نسخه واقعی می‌توانید این را هم از دیتابیس بخوانید.
    return """
    Policy Information:
    - Shipping Cost: Free for orders over 700,000 Toman, otherwise 50,000 Toman flat rate.
    - Delivery Time: 2-3 business days for Tehran, 3-5 days for other cities.
    - Returns: Accepted within 7 days if the package is unopened.
    - Guarantee: All products are 100% original.
    - reply user in persian
    """
