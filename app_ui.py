import os
import uuid
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct, VectorParams, Distance

from full_agent import (
    app,
    embedder,
    qdrant_client,
    COLLECTION_NAME,
    clean_llm_output
)

st.set_page_config(
    page_title="Adaptive Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# SIDEBAR: KNOWLEDGE BASE INGESTION & CONFIG
# ==========================================
with st.sidebar:
    st.header("📄 Knowledge Base Ingestion")
    st.write("Upload custom documents to dynamically index them into Qdrant.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            all_text_chunks = []
            
            with st.spinner("Parsing and chunking documents..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                for uploaded_file in uploaded_files:
                    file_text = ""
                    try:
                        # 1. Parse PDF
                        if uploaded_file.name.endswith(".pdf"):
                            pdf_reader = PdfReader(uploaded_file)
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                if text:
                                    file_text += text + "\n"
                        # 2. Parse Plain Text / Markdown
                        else:
                            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")
                        continue
                        
                    if file_text.strip():
                        chunks = text_splitter.split_text(file_text)
                        for chunk in chunks:
                            all_text_chunks.append({
                                "source": uploaded_file.name,
                                "text": chunk
                            })
            
            if all_text_chunks:
                with st.spinner(f"Embedding and upserting {len(all_text_chunks)} chunks to Qdrant..."):
                    try:
                        # Ensure collection exists before upserting
                        if not qdrant_client.collection_exists(COLLECTION_NAME):
                            qdrant_client.create_collection(
                                collection_name=COLLECTION_NAME,
                                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                            )
                            
                        points = []
                        for item in all_text_chunks:
                            vector = embedder.encode(item["text"]).tolist()
                            points.append(
                                PointStruct(
                                    id=str(uuid.uuid4()),
                                    vector=vector,
                                    payload={
                                        "text": item["text"],
                                        "source": item["source"]
                                    }
                                )
                            )
                        
                        qdrant_client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=points
                        )
                        st.success(f"✅ Successfully indexed {len(points)} chunks from {len(uploaded_files)} file(s)!")
                    except Exception as e:
                        st.error(f"Failed to upsert to Qdrant: {e}")
            else:
                st.error("Could not extract any readable text from the uploaded files.")

    st.markdown("---")
    st.header("🗑️ Collection Management")
    if st.button("Clear Vector Database", type="secondary"):
        try:
            if qdrant_client.collection_exists(COLLECTION_NAME):
                qdrant_client.delete_collection(COLLECTION_NAME)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            st.success("Vector database wiped clean and recreated!")
        except Exception as e:
            st.error(f"Error resetting database: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ System Configuration")
    st.write(f"• **Collection:** `{COLLECTION_NAME}`")
    st.write("• **Embedding:** `BAAI/bge-small-en-v1.5`")
    st.write("• **Reranker:** `BAAI/bge-reranker-base`")
    st.write("• **LLM:** `Groq (Qwen / Llama)`")

# ==========================================
# MAIN INTERACTION: AGENTIC CHAT
# ==========================================
st.title("🤖 Adaptive Agentic RAG Assistant")
st.caption("Autonomous State Graph • Self-Correcting • Cross-Encoder Reranked")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg and msg["details"]:
            with st.expander("🔍 View Agent Execution Trace"):
                st.json(msg["details"])

# Handle User Query
if prompt := st.chat_input("Ask a question about your uploaded documents or AI topics..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt, "details": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run Agent State Graph
    with st.chat_message("assistant"):
        with st.status("Agent reasoning in progress...", expanded=True) as status:
            st.write("🚦 Evaluating routing decision...")
            try:
                result = app.invoke({"question": prompt})
                docs = result.get("documents", [])
                st.write(f"📄 Documents retrieved & graded: {len(docs)}")
                status.update(label="Response ready!", state="complete", expanded=False)
                
                answer = clean_llm_output(result.get("generation", "No response generated."))
            except Exception as e:
                status.update(label="Error processing query", state="error", expanded=True)
                answer = f"An error occurred while running the agent graph: {e}"
                docs = []

        st.markdown(answer)

        trace_info = {
            "Is Summary Query": result.get("is_summary_query", False) if 'result' in locals() else False,
            "Total Graded Documents": len(docs),
            "Retrieved Payloads": docs if docs else "Bypassed / Direct Chat / Web Search"
        }
        
        with st.expander("🔍 View Agent Execution Trace"):
            st.json(trace_info)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "details": trace_info
        })