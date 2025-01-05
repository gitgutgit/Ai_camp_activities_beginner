import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI


################################
# 1. OpenAI API 키 설정
################################

load_dotenv()  # .env 파일 로드
our_api_key = os.getenv("OPENAI_API_KEY_AICAMP", None)
client = OpenAI(api_key=our_api_key)

################################
# 2. 웹앱 기본 설정
################################
st.title("나만의 간단 챗봇 만들기")
st.write("""
이곳에 메시지를 입력하고 **전송** 버튼을 누르면, 
ChatGPT API를 통해 만들어진 챗봇과 대화를 할 수 있습니다.
""")

# 이전 대화내용을 저장하기 위해 Session State 사용
if "messages" not in st.session_state:
    st.session_state["messages"] = []

################################
# 3. 사용자 입력 받기
################################
user_input = st.text_input("메시지를 입력하세요:", value="", max_chars=100)

# 전송 버튼 생성
if st.button("전송"):
    # 사용자가 입력한 내용(messages 리스트에 추가)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # ChatGPT API를 사용하여 응답 생성
    response = client.chat.completions.create(model="gpt-4o",  
    messages=st.session_state["messages"])

    # 응답 받아오기
    assistant_content = response.choices[0].message.content

    # 챗봇 응답도 messages 리스트에 추가
    st.session_state["messages"].append({"role": "assistant", "content": assistant_content})

################################
# 4. 지금까지의 대화 내용 출력
################################
st.write("### 대화 내용")
for i, msg in enumerate(st.session_state["messages"]):
    # 사용자의 메시지
    if msg["role"] == "user":
        st.write(f"**사용자**: {msg['content']}")
    # 챗봇의 메시지
    else:
        st.write(f"**챗봇**: {msg['content']}")
