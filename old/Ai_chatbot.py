import os
from dotenv import load_dotenv
from openai import OpenAI





import streamlit as st

# ---------------------------------
# 1) 기본 설정
# ---------------------------------
load_dotenv()  # .env 파일 로드
our_api_key = os.getenv("OPENAI_API_KEY_AICAMP", None)
client = OpenAI(api_key=our_api_key)

st.set_page_config(
    page_title="My Custom ChatGPT Chatbot",
    layout="centered"
)

st.title("나만의 ChatGPT 챗봇")

# ChatGPT에게 전달할 메시지들의 히스토리를 세션 스테이트에 저장
if "messages" not in st.session_state:
    # system 역할의 '시스템 프롬프트'를 미리 설정할 수도 있습니다.
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# ---------------------------------
# 2) 사용자 입력 받기
# ---------------------------------
custom_prompt = st.text_area("시스템 프롬프트(System Prompt) 직접 입력", 
                             value="You are a helpful assistant. 답변은 친절하고 간결하게 해주세요.")

user_input = st.text_input("질문을 입력해주세요.", "")

# ---------------------------------
# 3) 프롬프트 변경 적용하기
# ---------------------------------
# 시스템 프롬프트가 바뀌면 세션 스테이트의 첫 번째 system 메시지를 수정
if st.button("시스템 프롬프트 적용"):
    if st.session_state["messages"] and st.session_state["messages"][0]["role"] == "system":
        st.session_state["messages"][0]["content"] = custom_prompt
    st.success("시스템 프롬프트가 변경되었습니다.")

# ---------------------------------
# 4) 사용자 질문 처리하기
# ---------------------------------
if st.button("질문 전송"):
    if user_input.strip():
        # 유저가 입력한 메시지를 대화 히스토리에 추가
        st.session_state["messages"].append(
            {"role": "user", "content": user_input}
        )

        # OpenAI ChatCompletion API 호출
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=st.session_state["messages"])

        # API로부터 받은 어시스턴트 답변을 대화 히스토리에 추가
        reply = response.choices[0].message.content
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )

# ---------------------------------
# 5) 대화 히스토리 출력
# ---------------------------------
st.write("---")
for idx, msg in enumerate(st.session_state["messages"]):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(f"**You:** {content}")
    elif role == "assistant":
        st.markdown(f"**ChatGPT:** {content}")
    else:
        # system or other roles
        st.markdown(f"**{role.capitalize()}**: {content}")
