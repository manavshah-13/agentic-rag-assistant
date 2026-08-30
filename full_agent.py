import os
import re
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

load_dotenv()

# ==========================================
# 0. HELPER FUNCTIONS & ENVIRONMENT CLEANING
# ==========================================
def clean_env_var(val: str | None) -> str:
    """Sanitize environment variables by removing single/double quotes, spaces, and escapes."""
    if not val:
        return ""
    return val.strip().strip('"').strip("'")

def clean_llm_output(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    return cleaned.strip()

SUMMARY_KEYWORDS = [
    "summary", "summarize", "what is in", "overview", "this file",
    "the document", "the pdf", "about the document", "explain the file",
    "main points", "table of contents", "describe the document", "content of"
]

def is_summary_query(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in SUMMARY_KEYWORDS)

# ==========================================
# 1. INITIALIZE CLIENTS & MODELS
# ==========================================
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("BAAI/bge-reranker-base")

raw_qdrant_url = clean_env_var(os.getenv("QDRANT_URL", ""))
qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY", ""))

qdrant_client = QdrantClient(
    url=raw_qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None
)
COLLECTION_NAME = "demo_collection"

tavily_api_key = clean_env_var(os.getenv("TAVILY_API_KEY", ""))
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

groq_api_key = clean_env_var(os.getenv("GROQ_API_KEY", ""))
groq_model = clean_env_var(os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=groq_model,
    temperature=0
)

# ==========================================
# 2. DEFINE AGENT STATE
# ==========================================
class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    is_summary_query: bool

# ==========================================
# 3. DEFINE NODES
# ==========================================

# --- Node A: Retrieve Candidate Documents from Qdrant ---
def retrieve(state: AgentState):
    print("\n--- 🔎 NODE: RETRIEVING CANDIDATES FROM QDRANT ---")
    question = state["question"]
    summary_flag = is_summary_query(question)
    
    if summary_flag:
        print("  -> Summary intent detected. Retrieving broad document sample...")
        try:
            records, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10,
                with_payload=True
            )
            docs = [point.payload["text"] for point in records if point.payload and "text" in point.payload]
        except Exception as e:
            print(f"  -> Qdrant scroll error: {e}")
            docs = []
        return {"documents": docs, "question": question, "is_summary_query": True}
    else:
        try:
            query_vector = embedder.encode(question).tolist()
            search_response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=6
            )
            docs = [point.payload["text"] for point in search_response.points if point.payload and "text" in point.payload]
        except Exception as e:
            print(f"  -> Qdrant search error: {e}")
            docs = []
        return {"documents": docs, "question": question, "is_summary_query": False}

# --- Node B: Cross-Encoder Semantic Reranker ---
def rerank(state: AgentState):
    print("--- 🎯 NODE: SEMANTIC RERANKING ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"documents": [], "question": question}
    
    # If summary query, bypass reranking to preserve broad context
    if state.get("is_summary_query"):
        return {"documents": documents[:6], "question": question}
    
    pairs = [[question, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    top_docs = [doc for doc, score in scored_docs[:3]]
    return {"documents": top_docs, "question": question}

# --- Node C: Grade Retrieved Documents ---
class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocument)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the retrieved document is useful for answering the question. Grade 'yes' or 'no'."),
    ("human", "Document:\n\n{document}\n\nQuestion: {question}")
])
doc_grader = grader_prompt | structured_llm_grader

def grade_documents(state: AgentState):
    print("--- ⚖️ NODE: CHECKING DOCUMENT RELEVANCE ---")
    question = state["question"]
    documents = state["documents"]
    
    # Summary queries bypass LLM relevance grading to prevent false rejections
    if state.get("is_summary_query"):
        return {"documents": documents, "question": question}
    
    filtered_docs = []
    for doc in documents:
        try:
            score = doc_grader.invoke({"question": question, "document": doc})
            if score.binary_score == "yes":
                filtered_docs.append(doc)
        except Exception as e:
            print(f"  -> Grader fallback due to parsing/invoke error: {e}")
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs, "question": question}

# --- Node D: Web Search Fallback ---
def web_search(state: AgentState):
    print("--- 🌐 NODE: FALLBACK WEB SEARCH ---")
    question = state["question"]
    if not tavily_client:
        return {"documents": ["Tavily API key not configured."], "question": question}
    try:
        search_response = tavily_client.search(query=question, max_results=2)
        web_results = [res["content"] for res in search_response.get("results", [])]
    except Exception as e:
        print(f"  -> Tavily search error: {e}")
        web_results = ["Web search unavailable."]
    return {"documents": web_results, "question": question}

# --- Node E: Generate Answer ---
def generate(state: AgentState):
    print("--- 🤖 NODE: GENERATING FINAL ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    if documents:
        context = "\n\n".join(documents)
    else:
        if state.get("is_summary_query"):
            context = "No documents found in knowledge base. Please upload a document to generate a summary."
        else:
            context = "No relevant context found."
        
    prompt = f"""You are a helpful assistant. Use the provided context to answer the user request accurately and comprehensively.

Context:
{context}

User Question: {question}
Answer:"""
    
    response = llm.invoke(prompt)
    return {"generation": clean_llm_output(response.content)}

# --- Node F: Direct Chat ---
def direct_chat(state: AgentState):
    print("\n--- 💬 NODE: DIRECT CHAT ---")
    question = state["question"]
    response = llm.invoke(question)
    return {"generation": clean_llm_output(response.content)}

# ==========================================
# 4. CONDITIONAL ROUTING
# ==========================================
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "direct_chat"] = Field(..., description="Route query")

structured_llm_router = llm.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing a user question to either 'vectorstore' or 'direct_chat'.

    - Route to 'vectorstore' if the user asks about an uploaded document, file, summary, technical topics (such as LPUs, GPUs, Groq, Qdrant, compilers), code, or specific domain knowledge.
    - Route to 'direct_chat' for general world knowledge (e.g., capitals, basic math, geography, history), greetings, or general conversational questions."""),
    ("human", "{question}")
])
question_router = route_prompt | structured_llm_router

def route_question(state: AgentState):
    print("\n--- 🚦 ROUTING INITIAL QUESTION ---")
    question = state["question"]
    
    # Deterministic override for document/summary queries
    if is_summary_query(question):
        print("  -> Intercepted summary/document intent. Routing to vectorstore.")
        return "vectorstore"
        
    try:
        source = question_router.invoke({"question": question})
        return "vectorstore" if source.datasource == "vectorstore" else "direct_chat"
    except Exception as e:
        print(f"  -> Router exception fallback to vectorstore: {e}")
        return "vectorstore"

def decide_to_generate(state: AgentState):
    print("--- 🔍 EVALUATING GRADED DOCUMENTS ---")
    if state.get("is_summary_query"):
        print("  -> Summary query. Routing directly to GENERATE 🤖")
        return "generate"
        
    if not state.get("documents"):
        print("  -> Routing to WEB SEARCH 🌐")
        return "websearch"
    else:
        print("  -> Routing to GENERATE 🤖")
        return "generate"

# ==========================================
# 5. BUILD & COMPILE GRAPH
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("direct_chat", direct_chat)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
        "direct_chat": "direct_chat"
    }
)

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("direct_chat", END)

app = workflow.compile()