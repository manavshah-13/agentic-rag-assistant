import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

load_dotenv()

def clean_env_var(val: str | None) -> str:
    if not val:
        return ""
    return val.strip().strip('"').strip("'")

groq_api_key = clean_env_var(os.getenv("GROQ_API_KEY", ""))
groq_model = clean_env_var(os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"))
raw_qdrant_url = clean_env_var(os.getenv("QDRANT_URL", ""))
qdrant_api_key = clean_env_var(os.getenv("QDRANT_API_KEY", ""))

# 1. Test Groq Connection
try:
    groq_client = Groq(api_key=groq_api_key)
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": "Hello! Say system ready."}]
    )
    print("Groq Status:", response.choices[0].message.content)
except Exception as e:
    print("Groq Error:", e)

# 2. Test Qdrant Connection
try:
    qdrant_client = QdrantClient(
        url=raw_qdrant_url,
        api_key=qdrant_api_key if qdrant_api_key else None
    )
    collections = qdrant_client.get_collections()
    print("Qdrant Connected! Current Collections:", collections.collections)
except Exception as e:
    print("Qdrant Error:", e)