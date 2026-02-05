from __future__ import annotations

import os

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.core.agent import build_agent_graph
from app.core.retrieval import SecureRetriever


test_questions = [
    {
        "question": "What is the CEO's salary?",
        "ground_truth": "The CEO earns $500k.",
        "role": "admin",
    },
    {
        "question": "How do I reset my password?",
        "ground_truth": "Go to settings > security.",
        "role": "intern",
    },
]


def run_eval() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for Ragas evaluation.")

    retriever = SecureRetriever()
    workflow = build_agent_graph(retriever)

    records = []
    for item in test_questions:
        state = {
            "question": item["question"],
            "generation": "",
            "documents": [],
            "user_roles": [item["role"]],
            "steps": [],
            "rewrite_query": False,
            "retries": 0,
            "route": "",
            "grounded": False,
            "hallucination_retries": 0,
            "force_grounded": False,
        }
        result = workflow.invoke(state)
        documents = result.get("documents", [])
        contexts = [doc.page_content for doc in documents]
        records.append(
            {
                "question": item["question"],
                "answer": result.get("generation", ""),
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )

    dataset = Dataset.from_list(records)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("Evaluation results:")
    print(results)

    results_df = results.to_pandas()
    results_df.to_csv("results.csv", index=False)
    print("Saved results to results.csv")


if __name__ == "__main__":
    run_eval()
