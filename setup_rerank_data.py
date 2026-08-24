import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "demo_collection"

# Recreate collection cleanly
if qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.delete_collection(COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

docs = [
    {"id": 1, "text": "Groq provides ultra-fast LLM inference using custom LPU chips designed specifically for sequential computing."},
    {"id": 2, "text": "GPUs from NVIDIA excel at massive parallel matrix multiplications, making them ideal for model training."},
    {"id": 3, "text": "The Groq compiler reorganizes model execution to take advantage of deterministic SRAM access patterns on the LPU."},
    {"id": 4, "text": "Qdrant is an open-source vector search engine with extended filtering support."},
    {"id": 5, "text": "A comparison between LPU and GPU architectures shows LPUs reduce latency for batch size 1 inference workloads."}
]

points = [
    PointStruct(id=doc["id"], vector=embedder.encode(doc["text"]).tolist(), payload={"text": doc["text"]})
    for doc in docs
]

qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
print("✅ Ingested 5 technical documents into Qdrant for reranker testing.")