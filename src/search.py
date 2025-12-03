"""
RAG (Retrieval-Augmented Generation) system module.
Combines vector search with LLM for intelligent question answering.
"""
import os
import getpass
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from .vectorstore import ChromaVectorStore
from config import (
    GOOGLE_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_RETRIES,
    EMBEDDING_MODEL_NAME,
    DEFAULT_TOP_K,
    PROXY_URL,
    NO_PROXY
)


def setup_proxy() -> None:
    """
    Configure proxy settings for network requests.
    
    Local traffic (Ollama) should not go through proxy.
    This must be called before any network-dependent imports.
    """
    if PROXY_URL:
        os.environ["HTTP_PROXY"] = PROXY_URL
        os.environ["HTTPS_PROXY"] = PROXY_URL
    
    os.environ["NO_PROXY"] = NO_PROXY
    os.environ["no_proxy"] = NO_PROXY


# Setup proxy before imports that use network
setup_proxy()


class RAGSystem:
    """
    Retrieval-Augmented Generation system.
    
    Combines vector similarity search with LLM to provide
    context-aware responses based on document retrieval.
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> None:
        """
        Initialize the RAG system.
        
        Args:
            google_api_key: Google API key for Gemini. If None, reads from
                           environment variable or prompts user.
            llm_model: LLM model name. Defaults to LLM_MODEL_NAME from config.
            embedding_model: Embedding model name. Defaults to EMBEDDING_MODEL_NAME from config.
        
        Raises:
            ValueError: If API key cannot be obtained.
            RuntimeError: If vector store or LLM initialization fails.
        """
        # Handle API key
        api_key = google_api_key or GOOGLE_API_KEY
        if not api_key:
            api_key = getpass.getpass("Google API Key: ")
            if not api_key:
                raise ValueError("Google API key is required")
        
        os.environ["GOOGLE_API_KEY"] = api_key
        print("[INFO] Google API key configured")
        
        # Initialize vector store
        model_name = embedding_model or EMBEDDING_MODEL_NAME
        print("[INFO] Connecting to vector store...")
        try:
            self.vectorstore = ChromaVectorStore(embedding_model=model_name)
            print("[INFO] Vector store connected successfully")
        except Exception as e:
            print(f"[ERROR] Failed to connect to vector store: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e
        
        # Initialize LLM
        llm_name = llm_model or LLM_MODEL_NAME
        try:
            self.llm: BaseChatModel = ChatGoogleGenerativeAI(
                model=llm_name,
                temperature=LLM_TEMPERATURE,
                max_retries=LLM_MAX_RETRIES
            )
            print(f"[INFO] RAG system initialized with model: {llm_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
    
    def retrieve_context(self, query: str, k: int = None) -> Optional[str]:
        """
        Retrieve relevant context from vector store for a given query.
        
        Args:
            query: Search query string.
            k: Number of documents to retrieve. Defaults to DEFAULT_TOP_K from config.
            
        Returns:
            Formatted context string with product information, or None if no results found.
            
        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        k = k or DEFAULT_TOP_K
        results = self.vectorstore.query(query, top_k=k)
        
        if not results:
            print(f"[WARN] No results found for query: '{query}'")
            return None
        
        context_parts = []
        for i, res in enumerate(results, 1):
            meta = res.get("metadata", {})
            content = res.get("content", "")
            
            info = (
                f"--- پیشنهاد {i} ---\n"
                f"نام محصول: {meta.get('name', 'ناشناس')}\n"
                f"دسته: {meta.get('category', 'عمومی')}\n"
                f"توضیحات: {content}\n"
            )
            context_parts.append(info)
        
        context = "\n".join(context_parts)
        return context
    
    def ask(self, query: str) -> str:
        """
        Ask a question and get an AI-generated response based on retrieved context.
        
        Args:
            query: User's question or query string.
            
        Returns:
            AI-generated response string.
            
        Raises:
            ValueError: If query is empty.
            RuntimeError: If LLM invocation fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        print(f"\n[USER] {query}")
        
        # Retrieve context
        context = self.retrieve_context(query)
        
        if not context:
            no_result_msg = "متاسفانه در حال حاضر محصولی مرتبط با درخواست شما در انبار موجود نیست."
            print("[INFO] No context found, returning default message")
            return no_result_msg
        
        # System instruction for the LLM
        system_instruction = """
        تو یک مشاور زیبایی و فروشنده حرفه‌ای و دلسوز هستی.
        وظیفه تو کمک به کاربر برای رفع نیاز اوست، حتی اگر محصول دقیقی که می‌خواهد را نداشته باشیم.
        
        لیست محصولات موجود در انبار:
        {context}

        دستورالعمل پاسخگویی:
        1. بر اساس سوال کاربر و محصولات موجود در لیست بالا، مستقیماً پاسخ بده. از کاربر سوال نپرس.
        2. نیاز کاربر را از سوالش تشخیص بده (مثلاً کنترل چربی، پوشش‌دهی، آبرسانی) و محصولات مناسب را پیشنهاد کن.
        3. اگر محصول دقیقی که کاربر خواست (مثل کرم پودر) در لیست نبود، فوراً نگو "نداریم". بلکه بگو "برای هدفی که داری، من این محصولات جایگزین فوق‌العاده را پیشنهاد می‌کنم" و توضیح بده چرا این محصولات به کارش می‌آیند.
        4. از دانش خودت برای ربط دادن محصولات موجود (مثل پرایمر یا پنکیک) به نیاز کاربر استفاده کن.
        5. لحن پاسخ باید گرم، صمیمی و کاملاً فارسی باشد.
        6. قیمت و موجودی را نگو مگر اینکه در متن محصولات باشد.
        7. مهم: همیشه مستقیماً پاسخ بده و از کاربر سوال نپرس. از context استفاده کن و محصولات مناسب را پیشنهاد کن.
        """
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{question}"),
        ])
        
        # Create chain
        chain: Runnable = prompt | self.llm
        
        try:
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            answer = response.content if hasattr(response, "content") else str(response)
            return answer
            
        except Exception as e:
            error_msg = f"خطا در دریافت پاسخ از هوش مصنوعی: {e}"
            print(f"[ERROR] LLM invocation failed: {e}")
            raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    try:
        rag = RAGSystem(
            llm_model=LLM_MODEL_NAME
        )
        
        # Test query
        test_query = "من پوستم خشک هستم و پوسته پوسته میشه"
        answer = rag.ask(test_query)
        
        print("\n" + "=" * 40)
        print(answer)
        print("=" * 40)
        
    except Exception as e:
        print(f"[ERROR] Application error: {e}")
        raise
