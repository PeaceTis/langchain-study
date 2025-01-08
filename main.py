import glob

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI

load_dotenv()


st.set_page_config(page_title="나만의 ChatGPT 💬", page_icon="💬")
st.title("나만의 ChatGPT 💬")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인을 생성합니다.
def create_chain(prompt_filepath, model, task=""):
    # 프롬프트 적용
    prompt = load_prompt(prompt_filepath)

    if task:
        prompt = prompt.partial(task=task)

    llm = ChatOpenAI(model_name=model)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화내용 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")

if clear_btn:
    retriever = st.session_state["messages"].clear()

print_history()

if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    add_message("user", user_input)
    st.chat_message("user").write(user_input)
    chain = create_chain(selected_prompt, "gpt-4o-mini", task_input)

    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        add_message("ai", ai_answer)
