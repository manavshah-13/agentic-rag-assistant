import os
import re
import json
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from full_agent import app

load_dotenv()

# 1. Initialize Evaluator Judge LLM
judge_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b", # Use strong reasoning model as judge
    temperature=0
)

# Helper to clean reasoning tags
def clean_text(raw_text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

# ==========================================
# 2. DEFINE STRUCTURED EVALUATION SCHEMAS
# ==========================================
class FaithfulnessScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0. 1.0 means every claim in the answer is completely grounded in the retrieved context.")
    reasoning: str = Field(description="Brief explanation of whether any hallucinated facts were introduced.")

class RelevanceScore(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0. 1.0 means the answer directly answers the query without fluff.")
    reasoning: str = Field(description="Brief explanation of relevance.")

structured_judge_faithfulness = judge_llm.with_structured_output(FaithfulnessScore)
structured_judge_relevance = judge_llm.with_structured_output(RelevanceScore)

# ==========================================
# 3. EVALUATION PROMPTS
# ==========================================
faithfulness_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial judge evaluating the FAITHFULNESS of an AI generated response.\n"
               "Determine if every fact in the response is directly supported by the provided context.\n"
               "Give a score from 0.0 (completely made up) to 1.0 (100% faithful to context)."),
    ("human", "Context:\n{context}\n\nGenerated Response:\n{response}")
])

relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial judge evaluating the RELEVANCE of an answer to a question.\n"
               "Determine if the answer directly addresses what the user asked.\n"
               "Give a score from 0.0 (irrelevant) to 1.0 (perfectly relevant and direct)."),
    ("human", "User Question:\n{question}\n\nGenerated Response:\n{response}")
])

eval_faithfulness_chain = faithfulness_prompt | structured_judge_faithfulness
eval_relevance_chain = relevance_prompt | structured_judge_relevance

# ==========================================
# 4. BENCHMARK TEST DATASET
# ==========================================
test_dataset = [
    {
        "question": "How do LPUs compare to GPUs for batch size 1 inference workloads?",
        "type": "Vector DB Hit"
    },
    {
        "question": "How does the Groq compiler optimize model execution on SRAM?",
        "type": "Vector DB Hit"
    },
    {
        "question": "What is the capital of Japan?",
        "type": "Direct Chat Bypass"
    },
    {
        "question": "Who won the ICC T20 World Cup in 2024?",
        "type": "Fallback Web Search"
    }
]

# ==========================================
# 5. RUN EVALUATION PIPELINE
# ==========================================
def run_evaluation():
    print("🚀 Starting Agentic RAG Evaluation Benchmark...\n")
    results = []

    for idx, item in enumerate(test_dataset, 1):
        q = item["question"]
        q_type = item["type"]
        print(f"[{idx}/{len(test_dataset)}] Running Query: '{q}' ({q_type})")
        
        # Invoke our compiled LangGraph Agent
        state = app.invoke({"question": q})
        answer = clean_text(state["generation"])
        docs = state.get("documents", [])
        context_str = "\n".join(docs) if docs else "N/A (Direct Chat or No DB Context)"

        # 1. Evaluate Faithfulness (if context was used)
        if docs:
            f_eval = eval_faithfulness_chain.invoke({"context": context_str, "response": answer})
            faith_score = f_eval.score
            faith_reason = f_eval.reasoning
        else:
            faith_score = 1.0  # Direct chat has no context to contradict
            faith_reason = "Bypassed retrieval; direct generation."

        # 2. Evaluate Answer Relevance
        r_eval = eval_relevance_chain.invoke({"question": q, "response": answer})
        rel_score = r_eval.score
        rel_reason = r_eval.reasoning

        results.append({
            "Question": q[:40] + "...",
            "Type": q_type,
            "Faithfulness": round(faith_score, 2),
            "Relevance": round(rel_score, 2),
            "Faithfulness Reason": faith_reason[:60] + "...",
            "Relevance Reason": rel_reason[:60] + "..."
        })

    # Summary Report
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 EVALUATION BENCHMARK REPORT")
    print("="*80)
    print(df[["Question", "Type", "Faithfulness", "Relevance"]].to_markdown(index=False))
    
    avg_faith = df["Faithfulness"].mean()
    avg_rel = df["Relevance"].mean()
    print("\n📈 Aggregate Metrics:")
    print(f"  • Mean Faithfulness : {avg_faith:.2f} / 1.00")
    print(f"  • Mean Relevance    : {avg_rel:.2f} / 1.00")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()