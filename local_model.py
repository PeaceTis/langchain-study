import os

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer:"""

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

import os
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# 사용할 모델의 저장소 ID를 설정합니다.
# repo_id = "microsoft/Phi-3-mini-4k-instruct"
# repo_id = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"
repo_id = "Saxo/Linkbricks-Horizon-AI-Korean-Superb-22B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,  # 모델 저장소 ID를 지정합니다.
    max_new_tokens=256,  # 생성할 최대 토큰 길이를 설정합니다.
    temperature=0.1,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # 허깅페이스 토큰
)

# LLMChain을 초기화하고 프롬프트와 언어 모델을 전달합니다.
chain = prompt | llm | StrOutputParser()
# 질문을 전달하여 LLMChain을 실행하고 결과를 출력합니다.
response = chain.invoke({"question": "생산관리에서 제일 중요한 요소는 무엇일까?"})
print(response)