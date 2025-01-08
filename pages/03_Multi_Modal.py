import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.title("이미지 인식 기반 챗봇 💬")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 탭 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    system_prompt = st.text_area("시스템 프롬프트",
                                 """당신은 공장을 도형으로 추상화하는 산업공학 전문가 입니다. 주어진 이미지를 바탕으로 공장에 대해 알려주세요.
도형들을 감싸고 있는 네모는 공장입니다.
세모는 창고입니다.
점선 네모는 플로우입니다.
실선 네모는 공정입니다.
동그라미는 제품입니다.
다이아몬드는 설비입니다.
플로우 내부에 여러 공정들이 있습니다.
창고(삼각형)와 제품(원), 공정(실선 네모)과 설비(다이아몬드)의 내부에 있는 문자열은 아이디입니다.
공장(도형들을 감싸는 큰 네모)과 Flow(점선 네모)의 위에 있는 문자열은 각각 공장, 플로우 아이디입니다.
즉, 점선 네모 위의 문자열은 플로우 아이디, 내부의 실선 네모 안의 문자열은 공정 아이디입니다.
제품의 경우 창고의 바로 오른쪽에 있는 제품이 창고에 적재된 제품이며, 창고의 왼쪽의 플로우와 플로우 내부의 공정들도 해당 제품이 가공됩니다.
제품끼리의 연결구조를 PS(Product Structure)라고 합니다.
화살표대로 생산이 진행됩니다.
이 도형들의 정의, 관계 및 흐름을 통틀어 NEMOSYN(네모신)이라고 합니다. 
                                 """,
                                 height=200)


    clear_btn = st.button("대화내용 초기화")

    selected_prompt = "prompts/pdf-rag.yaml"


def print_history():
    for msg in st.session_state["messages"]:
        main_tab2.chat_message(msg.role).write(msg.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 파일을 캐시 저장
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path

# 체인을 생성합니다.
def generate_answer(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.

    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name=model_name,
    )

    # 멀티모달 객체 생성
    multimodal = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    # 이미지 파일로 부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)

    return answer


if clear_btn:
    retriever = st.session_state["messages"].clear()

user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = main_tab2.empty()

# 이미지가 업로드가 된다면
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)

print_history()

if user_input:
    if not uploaded_file:
        warning_msg = main_tab2.empty()
    else:
        # 파일 업로드 후 retriever 생성
        add_message("user", user_input)
        main_tab2.chat_message("user").write(user_input)
        image_filepath = process_imagefile(uploaded_file)
        response = generate_answer(image_filepath=image_filepath, model_name=selected_model, system_prompt=system_prompt, user_prompt=user_input)

        with main_tab2.chat_message("assistant"):
            container = main_tab2.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)
            add_message("ai", ai_answer)