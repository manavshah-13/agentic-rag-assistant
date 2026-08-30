import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

def clean_env_var(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().strip('"').strip("'")

groq_api_key = clean_env_var(os.getenv("GROQ_API_KEY", ""))
groq_model = clean_env_var(os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))
raw_qdrant_url = clean_env_var(os.getenv("QDRANT_URL", ""))
qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY", ""))

# 1. Initialize Clients
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
groq_client = Groq(api_key=groq_api_key)
qdrant_client = QdrantClient(
    url=raw_qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None
)

COLLECTION_NAME = "demo_collection"

# 2. Define User Query
user_query = "How does Groq achieve high speed?"
print(f"Searching for: '{user_query}'...")

# 3. Embed Query
query_vector = embedder.encode(user_query).tolist()

# 4. Search Qdrant using query_points
search_response = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1
)

if search_response.points:
    top_match = search_response.points[0]
    retrieved_text = top_match.payload.get("text", "")
    score = top_match.score

    print(f"Top Match Found (Similarity Score: {score:.4f}):\n   -> \"{retrieved_text}\"\n")

    # 5. Send Context + Prompt to Groq LLM
    prompt = f"""Use the following context to answer the question.

Context: {retrieved_text}

Question: {user_query}
Answer:"""

    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}]
    )

    print("LLM Response:")
    print(response.choices[0].message.content)
else:
    print("No matches found in collection.")