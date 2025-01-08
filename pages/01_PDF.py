import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.title("PDF 기반 QA 💬")

if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf", "txt"])
    clear_btn = st.button("대화내용 초기화")

    selected_prompt = "prompts/pdf-rag.yaml"

def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 파일을 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    return retriever


# 체인을 생성합니다.
def create_chain(retriever, model_name="gpt-4o"):
    prompt = load_prompt("prompts/pdf-rag.yaml")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file)
    st.session_state["chain"] = create_chain(retriever=retriever, model_name=selected_model)


if clear_btn:
    retriever = st.session_state["messages"].clear()

print_history()

if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    if not st.session_state["chain"]:
        # 경고 메시지를 띄우기 위한 빈 영역
        st.empty().error("파일을 업로드 해주세요.")
    else:
        add_message("user", user_input)
        st.chat_message("user").write(user_input)
        response = st.session_state["chain"].stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
            add_message("ai", ai_answer)