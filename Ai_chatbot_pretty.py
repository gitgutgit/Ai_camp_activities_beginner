import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

# -------------------------
# 0) OpenAI 클라이언트 설정
# -------------------------
load_dotenv()  # .env 파일 로드
our_api_key = os.getenv("OPENAI_API_KEY_AICAMP", None)

client = OpenAI(api_key=our_api_key)

# -------------------------
# A) 페이지 & CSS 설정
# -------------------------
st.set_page_config(
    page_title="My Custom ChatGPT Chatbot",
    layout="centered"
)

# 배경 이미지 삽입 (여기서는 예시 이미지를 사용, 원하시는 이미지 경로/URL로 교체하세요)
page_bg_img = """
<style>
body {
    background-image: url("https://img.freepik.com/free-vector/pastel-color-background_23-2148737420.jpg?w=2000"); 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 채팅 버블 스타일
custom_css = """
<style>
.chat-container {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 16px;
    border-radius: 8px;
}

.user-message {
    background-color: #E0F7FA; /* 연한 하늘색 */
    color: #000000;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 80%;
}

.assistant-message {
    background-color: #FFF3E0; /* 연한 오렌지색 */
    color: #000000;
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 80%;
}

.system-message {
    background-color: #ECECEC; 
    color: #666666;
    border-radius: 6px;
    padding: 6px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    font-style: italic;
    width: fit-content;
    max-width: 80%;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------
# B) 타이틀 영역
# -------------------------
# 이모지(emoji) 사용 예시: 🤖, 🌟, 💬 등 원하는 대로 추가 가능
st.title("나만의 ChatGPT 챗봇 🤖")

# -------------------------
# C) 세션 스테이트 초기화
# -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# -------------------------
# D) 사용자 입력 받기
# -------------------------
st.markdown("### ⚙️ 시스템 프롬프트 설정")
custom_prompt = st.text_area(
    "시스템 프롬프트(System Prompt)를 직접 입력해 보세요.", 
    value="You are a helpful assistant. 답변은 친절하고 간결하게 해주세요."
)

user_input = st.text_input("💬 질문을 입력해주세요.", "")

# -------------------------
# E) 시스템 프롬프트 변경 적용
# -------------------------
if st.button("시스템 프롬프트 적용 ✅"):
    if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "system":
        st.session_state["messages"][0]["content"] = custom_prompt
    st.success("시스템 프롬프트가 변경되었습니다!")

# -------------------------
# F) 사용자 질문 처리
# -------------------------
if st.button("질문 전송 🚀"):
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state["messages"]
        )
        reply = response.choices[0].message.content

        st.session_state["messages"].append({"role": "assistant", "content": reply})

# -------------------------
# G) 대화 히스토리 표시
# -------------------------
st.write("---")
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]

    if role == "system":
        st.markdown(
            f"<div class='system-message'>[시스템] {content}</div>", 
            unsafe_allow_html=True
        )
    elif role == "user":
        st.markdown(
            f"<div class='user-message'><strong>You:</strong> {content}</div>", 
            unsafe_allow_html=True
        )
    elif role == "assistant":
        st.markdown(
            f"<div class='assistant-message'><strong>ChatGPT:</strong> {content}</div>", 
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
