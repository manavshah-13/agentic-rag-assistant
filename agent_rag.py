import os
from typing import Literal
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

def clean_env_var(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().strip('"').strip("'")

groq_api_key = clean_env_var(os.getenv("GROQ_API_KEY", ""))
groq_model = clean_env_var(os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))
raw_qdrant_url = clean_env_var(os.getenv("QDRANT_URL", ""))
qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY", ""))

# 1. Initialize Clients & Models
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(
    url=raw_qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None
)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=groq_model,
    temperature=0
)

# 2. Define Pydantic Schema for Router Node
class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource."""
    datasource: Literal["vectorstore", "direct_chat"] = Field(
        ...,
        description="Given a user question, choose whether to route it to vectorstore or direct_chat."
    )

# Structured Output Router LLM
structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = """You are an expert at routing a user question to a vectorstore or direct_chat.
The vectorstore contains documents about AI frameworks, Groq LPUs, and Qdrant DB.
Use vectorstore for questions on these topics. Otherwise, use direct_chat."""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

question_router = route_prompt | structured_llm_router

# 3. Test Routing Logic
test_q1 = "How does Groq achieve high speed?"
test_q2 = "What is the capital of France?"

res1 = question_router.invoke({"question": test_q1})
res2 = question_router.invoke({"question": test_q2})

print(f"Question: '{test_q1}' --> Routed to: {res1.datasource}")
print(f"Question: '{test_q2}' --> Routed to: {res2.datasource}")