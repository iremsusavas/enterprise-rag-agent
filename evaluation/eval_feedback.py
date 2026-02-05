from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.core.agent import build_agent_graph
from app.core.retrieval import SecureRetriever


def run_feedback_eval(
    feedback_path: str | None = None,
    min_rating: str = "down",
) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for Ragas evaluation.")

    feedback_file = Path(
        feedback_path
        or os.getenv(
            "FEEDBACK_PATH",
            str(Path(__file__).parent / "feedback.jsonl"),
        )
    )
    if not feedback_file.exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_file}")

    negatives = _load_negative_examples(feedback_file)
    if not negatives:
        print("No negative feedback examples found.")
        return

    retriever = SecureRetriever()
    workflow = build_agent_graph(retriever)

    records = []
    for item in negatives:
        question = item.get("question") or ""
        role = item.get("user_role") or "intern"
        if not question:
            continue
        state = {
            "question": question,
            "generation": "",
            "documents": [],
            "user_roles": [role],
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
                "question": question,
                "answer": result.get("generation", ""),
                "contexts": contexts,
                "ground_truth": item.get("answer", ""),
            }
        )

    dataset = Dataset.from_list(records)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("Evaluation results for negative feedback set:")
    print(results)

    results_df = results.to_pandas()
    results_df.to_csv("feedback_results.csv", index=False)
    print("Saved results to feedback_results.csv")


def _load_negative_examples(path: Path) -> list[dict]:
    negatives = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("rating") == "down":
                negatives.append(record)
    return negatives


if __name__ == "__main__":
    run_feedback_eval()
