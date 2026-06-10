import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time

try:
    from main_graph import app as graph_app
except ImportError:
    st.error("Error: Could not import 'app' from 'main_graph.py'. Make sure the file exists.")
    st.stop()

# --- تنظیمات صفحه ---
st.set_page_config(
    page_title="Beauty AI Assistant",
    page_icon="💄",
    layout="centered"
)

st.markdown("""
<style>
    /* فونت فارسی (اختیاری) */
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Vazirmatn', sans-serif;
    }
    
    /* حباب پیام کاربر */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #E3F2FD;
        border-radius: 15px;
        text-align: right;
        direction: rtl;
    }
    
    /* حباب پیام ربات */
    .stChatMessage[data-testid="stChatMessageAvatarAssistant"] {
        background-color: #FCE4EC;
    }
    
    /* عنوان اصلی */
    h1 {
        color: #E91E63;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# header
st.title("💄 دستیار هوشمند زیبایی")
st.markdown("---")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history 
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        # showing rtl Persian texts
        st.markdown(f'<div style="direction: rtl; text-align: right;">{message.content}</div>', unsafe_allow_html=True)

# recieve input
if prompt := st.chat_input("سوال خود را بپرسید... (مثلاً: کرم ضد آفتاب چی داری؟)"):
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(f'<div style="direction: rtl; text-align: right;">{prompt}</div>', unsafe_allow_html=True)

    # processing with ai
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner('در حال بررسی و جستجو...'):
            try:
                # فراخوانی گراف اصلی (LangGraph)
                # نکته: کل تاریخچه سشن را به گراف می‌دهیم تا کانتکست داشته باشد
                result = graph_app.invoke({"messages": st.session_state.messages})
                
                # دریافت آخرین پیام (پاسخ ربات)
                bot_message = result['messages'][-1]
                
                # استخراج متن پاسخ
                if isinstance(bot_message.content, str):
                    full_response = bot_message.content
                else:
                    # اگر خروجی چندبخشی بود
                    full_response = "".join([part.get('text', '') for part in bot_message.content if isinstance(part, dict)])
                
                # نمایش افکت تایپ شدن (Typing Effect)
                # این فقط نمایشی است و متن یکجا تولید شده
                displayed_text = ""
                for char in full_response.split():
                    displayed_text += char + " "
                    message_placeholder.markdown(f'<div style="direction: rtl; text-align: right;">{displayed_text}▌</div>', unsafe_allow_html=True)
                    time.sleep(0.05) # سرعت تایپ
                
                # نمایش نهایی بدون نشانگر تایپ
                message_placeholder.markdown(f'<div style="direction: rtl; text-align: right;">{full_response}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"خطایی رخ داد: {str(e)}")
                full_response = "متاسفانه مشکلی پیش آمد. لطفاً دوباره تلاش کنید."

    # 3. ذخیره پاسخ ربات در حافظه
    st.session_state.messages.append(AIMessage(content=full_response))
