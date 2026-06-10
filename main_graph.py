# main_graph.py

import os
import json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END

# ایمپورت مدل‌ها
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# ایمپورت ابزارها و کانفیگ
from src.tools import search_products, get_shipping_policy
from config import GOOGLE_API_KEY, LLM_MODEL_NAME

from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
load_dotenv()

# تنظیم پروکسی اگر لازم است (طبق فایل search.py شما)
# os.environ["HTTP_PROXY"] = ... 

# ====================================================
# 1. تعریف وضعیت (State)
# ====================================================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

# ====================================================
# 2. تعریف مدل‌ها (LLMs)
# ====================================================

# مدل سوپروایزر (Local - Llama3)
# نکته: format="json" بسیار مهم است برای اینکه خروجی دقیق بگیریم
supervisor_llm = ChatOllama(
    model="qwen2.5:0.5b",  # مطمئن شوید llama3 را دارید: ollama pull llama3
    format="json",
    temperature=0
)

# خواندن کلید از env
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

# مدل ورکرها (Google Gemini)
worker_llm = ChatOpenAI(
    # مدل انتخاب شده از لیست موجود OpenRouter
    model="google/gemini-2.0-flash-001", 
    
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    default_headers={
        "HTTP-Referer": "http://localhost:8501", 
        "X-Title": "Beauty Marketplace Bot"
    }
)

# ====================================================
# 3. تعریف نودها (Nodes)
# ====================================================
def supervisor_node(state: AgentState):
    messages = state['messages']
    
    # 1. آماده‌سازی تاریخچه (حتماً قبل از پرامپت باشد)
    recent_history = messages[-3:] 
    
    # هندل کردن اینکه msg.type ممکن است همیشه رشته نباشد (محکم‌کاری)
    history_text = "\n".join([f"{getattr(msg, 'type', 'unknown')}: {msg.content}" for msg in recent_history])
    
    # 2. پرامپت (حتماً f-string باشد)
    system_prompt = f"""
    You are the router for a beauty marketplace chatbot.
    
    RECENT CONVERSATION HISTORY:
    {history_text}
    
    Based on the above history, decide which worker should act next.

    WORKER DEFINITIONS:
    - "PRODUCT_AGENT": For shopping, buying, product availability, prices, or product advice.
    - "POLICY_AGENT": For shipping costs, delivery time, returns, refunds, rules.
    - "GENERAL_AGENT": For greetings, chit-chat, thanks, or irrelevant inputs.

    INSTRUCTION:
    Return ONLY a raw JSON object with a single key "next".
    Example: {{"next": "PRODUCT_AGENT"}}
    """
    
    # نکته: در f-string باید آکولادهای خود جیسون را دوبل {{ }} بگذارید تا با متغیر اشتباه نشود.
    # اما چون ما مثال جیسون را ساده نوشتیم، بهتر است آن را دستی اضافه کنیم تا پیچیده نشود.
    
    # نسخه ساده‌تر و امن‌تر برای f-string:
    prompt_text = system_prompt + "\n\nJSON Output:"
    
    try:
        # درخواست از جمنای
        response = worker_llm.invoke(prompt_text)
        
        content = response.content
        # هندل کردن خروجی چندبخشی جمنای (محکم‌کاری عالی بود که اضافه کردید)
        if isinstance(content, list):
             content = "".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
             
        content = content.strip()
        
        # پاکسازی مارک‌داون
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        
        import json
        decision = json.loads(content)
        next_node = decision.get("next", "GENERAL_AGENT")
        
    except Exception as e:
        print(f"[WARN] Supervisor Error: {e}")
        next_node = "GENERAL_AGENT"
        
    return {"next": next_node}



def product_node(state: AgentState):
    # گرفتن کل تاریخچه پیام‌ها
    messages = state['messages']
    
    # 1. بایند کردن ابزار
    tools = [search_products]
    llm_with_tools = worker_llm.bind_tools(tools)
    
    # 2. ساخت دستورالعمل (System Prompt)
    system_msg = SystemMessage(content="""
    You are a professional beauty consultant.
    Your goal is to help the user find the right product.
    
    MEMORY INSTRUCTION:
    - You have access to the full conversation history.
    - If the user refers to previous info (e.g., "how much is *that* cream?"), look at the history to find which cream they mean.
    
    DECISION RULES:
    1. If you need more info (skin type, budget), ASK the user.
    2. If you have enough info, USE the 'search_products' tool.
    
    Speak in Persian.
    """)
    
    # 3. ترکیب: دستور سیستم + کل تاریخچه چت
    # نکته: ورودی invoke باید لیستی از پیام‌ها باشد
    full_history = [system_msg] + messages
    
    # 4. اجرا
    response = llm_with_tools.invoke(full_history)
    
    # 5. هندل کردن ابزار (مثل قبل)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_output = search_products.invoke(tool_call['args']['query'])
        
        # برای جواب نهایی هم باید تاریخچه را بدانی تا بدانی چه پرسیده بود
        final_response = worker_llm.invoke(
            full_history + [response] + [
                # اینجا نتیجه ابزار را به عنوان یک پیام ToolMessage اضافه می‌کنیم (استاندارد لنگ‌چین)
                # اما برای سادگی فعلاً متن می‌کنیم:
                HumanMessage(content=f"TOOL RESULT: {tool_output}\nNow answer the user.")
            ]
        )
        return {"messages": [final_response]}
    else:
        return {"messages": [response]}

def policy_node(state: AgentState):
    """
    ایجنت قوانین
    """
    last_user_msg = state['messages'][-1].content
    policy_info = get_shipping_policy.invoke(last_user_msg)
    
       
    prompt = f"""
    You are a customer support agent.
    User Question: {last_user_msg}
    Policy Info: {policy_info}
    
    Answer the user's question accurately based on the policy info.
    
    CRITICAL INSTRUCTION: Answer ONLY in Persian (Farsi).
    """
    
    response = worker_llm.invoke(prompt)
    return {"messages": [response]}

def general_node(state: AgentState):
    """
    ایجنت عمومی (برای سلام و احوالپرسی)
    """
    last_user_msg = state['messages'][-1].content
     
    prompt = f"""
    You are a helpful assistant for a beauty shop.
    The user said: "{last_user_msg}"
    
    Reply politely to the greeting or chat.
    
    CRITICAL INSTRUCTION: Answer ONLY in Persian (Farsi).
    Example: If user says "Hi", say "سلام! خوش آمدید".
    """
    
    response = worker_llm.invoke(prompt)
    return {"messages": [response]}

# ====================================================
# 4. ساخت گراف (Workflow)
# ====================================================

workflow = StateGraph(AgentState)

# افزودن نودها
workflow.add_node("SUPERVISOR", supervisor_node)
workflow.add_node("PRODUCT_AGENT", product_node)
workflow.add_node("POLICY_AGENT", policy_node)
workflow.add_node("GENERAL_AGENT", general_node)

# نقطه شروع
workflow.set_entry_point("SUPERVISOR")

# مسیردهی شرطی (از سوپروایزر به بقیه)
workflow.add_conditional_edges(
    "SUPERVISOR",
    lambda state: state['next'], # تابعی که تعیین می‌کند کجا برویم
    {
        "PRODUCT_AGENT": "PRODUCT_AGENT",
        "POLICY_AGENT": "POLICY_AGENT",
        "GENERAL_AGENT": "GENERAL_AGENT"
    }
)

# مسیر برگشت: در این دمو ساده، بعد از هر پاسخ، کار تمام می‌شود (END)
# در سیستم واقعی‌تر، ممکن است بخواهید دوباره منتظر ورودی کاربر بمانید.
workflow.add_edge("PRODUCT_AGENT", END)
workflow.add_edge("POLICY_AGENT", END)
workflow.add_edge("GENERAL_AGENT", END)

# کامپایل گراف
app = workflow.compile()

# ====================================================
# 5. اجرای تست (Main Loop)
# ====================================================
if __name__ == "__main__":
    print("--- Beauty Marketplace AI (LangGraph) ---")
    print("Type 'exit' to quit.")
    
    # حافظه موقت برای این نشست
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # اضافه کردن پیام کاربر به وضعیت
        chat_history.append(HumanMessage(content=user_input))
        
        # اجرای گراف
        # ما کل تاریخچه را می‌فرستیم تا سوپروایزر کانتکست داشته باشد
        result = app.invoke({"messages": chat_history})
        
        # گرفتن آخرین پیام (پاسخ ربات)
        bot_response = result['messages'][-1]
        
        content = bot_response.content
        if isinstance(content, list):
            # اگر محتوا به صورت لیست باشد (ساختار جدیدتر)، متن را استخراج می‌کنیم
            content = "".join([item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"])
        
        print(f"Bot: {content}")
        
        # اضافه کردن پاسخ ربات به تاریخچه برای دور بعد
        chat_history.append(bot_response)
