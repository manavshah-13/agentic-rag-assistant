import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

# 1. Initialize Clients & Local Embedding Model (Runs free on CPU)
print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "demo_collection"
VECTOR_SIZE = 384  # bge-small-en-v1.5 outputs 384-dimensional vectors

# 2. Create Collection in Qdrant if it doesn't exist
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection: '{COLLECTION_NAME}'")

# 3. Sample Document Chunks to Embed
documents = [
    {"id": 1, "text": "Agentic RAG uses reasoning loops like ReAct to evaluate retrieved information."},
    {"id": 2, "text": "Groq provides ultra-fast LLM inference using custom LPU chips."},
    {"id": 3, "text": "Qdrant is a high-performance vector database built for similarity search."}
]

# 4. Generate Embeddings & Prepare Points
points = []
for doc in documents:
    vector = embedder.encode(doc["text"]).tolist()
    points.append(
        PointStruct(
            id=doc["id"],
            vector=vector,
            payload={"text": doc["text"]}  # Store original text alongside the vector
        )
    )

# 5. Upsert into Qdrant
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("✅ Successfully embedded and stored 3 sample documents in Qdrant!")