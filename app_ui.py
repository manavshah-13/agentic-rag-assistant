import streamlit as st
from full_agent import app, clean_llm_output

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Adaptive Agentic RAG System")
st.caption("Powered by LangGraph • Groq (Qwen/Llama) • Qdrant • BGE Reranker • Tavily")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander("🔍 View Agent Execution Trace"):
                st.write(msg["details"])

# Handle New User Input
if prompt := st.chat_input("Ask a question about AI hardware, Groq, or anything else..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute LangGraph Pipeline
    with st.chat_message("assistant"):
        with st.status("Agent reasoning in progress...", expanded=True) as status:
            st.write("🚦 Evaluating routing decision...")
            result = app.invoke({"question": prompt})
            
            docs = result.get("documents", [])
            st.write(f"📄 Documents processed: {len(docs)}")
            status.update(label="Response generated!", state="complete", expanded=False)

        answer = clean_llm_output(result["generation"])
        st.markdown(answer)

        # Build execution trace details
        trace_info = {
            "Total Documents Retained": len(docs),
            "Retrieved Content Preview": docs if docs else "Bypassed / Direct Generation"
        }
        
        with st.expander("🔍 View Agent Execution Trace"):
            st.json(trace_info)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "details": trace_info
        })