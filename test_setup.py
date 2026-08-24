import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

# Load variables from .env file
load_dotenv()

# 1. Test Groq Connection
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello! Say system ready."}]
    )
    print("✅ Groq Status:", response.choices[0].message.content)
except Exception as e:
    print("❌ Groq Error:", e)

# 2. Test Qdrant Connection
try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    collections = qdrant_client.get_collections()
    print("✅ Qdrant Connected! Current Collections:", collections.collections)
except Exception as e:
    print("❌ Qdrant Error:", e)