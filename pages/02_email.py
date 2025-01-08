import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


st.set_page_config(page_title="나만의 ChatGPT 💬", page_icon="💬")
st.title("Email 요약기 💬")


if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []


def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    company: str = Field(description="메일을 보낸 사람의 회사")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일 제목")
    summary: str = Field(description="메일 본문을 요약한 텍스트")
    date: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")


# 체인을 생성합니다.
def create_email_parsing_chain(model):
    # PydanticOutputParser 생성
    output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

    llm = ChatOpenAI(temperature=0, model_name=model)

    prompt = PromptTemplate.from_template(
        """
    You are a helpful assistant. Please answer the following questions in KOREAN.

    #QUESTION:
    다음의 이메일 내용 중에서 주요 내용을 추출해 주세요.

    #EMAIL CONVERSATION:
    {email_conversation}

    #FORMAT:
    {format}
    """
    )

    # format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
    prompt = prompt.partial(format=output_parser.get_format_instructions())

    # 체인 생성
    chain = prompt | llm | output_parser

    return chain

# 체인을 생성합니다.
def create_report_chain(model):
    # PydanticOutputParser 생성
    llm = ChatOpenAI(temperature=0, model_name=model)

    report_prompt = load_prompt("../prompts/email.yaml")

    # 체인 생성
    report_chain = report_prompt | llm | StrOutputParser()

    return report_chain


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화내용 초기화")

if clear_btn:
    retriever = st.session_state["messages"].clear()

print_history()

if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    add_message("user", user_input)
    st.chat_message("user").write(user_input)
    # 이메일을 파싱하는 chain 생성 및 실행
    email_chain = create_email_parsing_chain("gpt-4o-mini")
    answer = email_chain.invoke({"email_conversation": user_input})

    from langchain_community.utilities import SerpAPIWrapper
    params = {"gl": "kr", "hl": "ko", "num": "3"}
    search = SerpAPIWrapper(params=params, search_engine="google")
    search_query = f"{answer.person} {answer.company} {answer.email}"
    search_result = eval(search.run(search_query))
    search_result_string = '\n'.join(search_result)

    report_chain = create_report_chain("gpt-4o-mini")
    report_chain_input = {
        "sender": answer.person,
        "additional_information": search_result_string,
        "company": answer.company,
        "email": answer.email,
        "subject": answer.subject,
        "summary": answer.summary,
        "date": answer.date,
    }

    response = report_chain.stream(report_chain_input)
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)
        add_message("ai", ai_answer)
