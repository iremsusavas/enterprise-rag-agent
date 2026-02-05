from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.agent import build_agent_graph  # noqa: E402
from app.core.retrieval import SecureRetriever  # noqa: E402


API_URL = os.getenv("API_URL", "http://localhost:8000")
FEEDBACK_PATH = os.getenv(
    "FEEDBACK_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "feedback.jsonl"),
)


def main():
    st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
    st.title("Enterprise Knowledge & Decision Assistant")

    with st.sidebar:
        st.header("Access Controls")
        role = st.selectbox("User Role", ["admin", "hr", "intern", "employee"])
        user_id = st.text_input("User ID", value="demo-user")
        st.divider()
        st.header("Ingestion")
        upload_file = st.file_uploader("Upload PDF/Markdown", type=["pdf", "md", "txt"])
        if st.button("Upload") and upload_file:
            _upload_document(upload_file, user_id)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("steps"):
                with st.expander("Thought Process"):
                    for step in message["steps"]:
                        st.write(step)
            if message.get("citations"):
                st.caption("Sources: " + ", ".join(message["citations"]))
            if message["role"] == "assistant":
                _render_feedback_buttons(message, idx)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            meta_container = st.container()
            response = _stream_chat(prompt, user_id, role, placeholder)
            with meta_container:
                with st.expander("Thought Process"):
                    for step in response.get("steps", []):
                        st.write(step)
                if response.get("citations"):
                    st.caption("Sources: " + ", ".join(response["citations"]))
                _render_feedback_buttons(
                    {
                        "role": "assistant",
                        "content": response.get("answer", ""),
                        "steps": response.get("steps", []),
                        "citations": response.get("citations", []),
                        "question": prompt,
                        "user_id": user_id,
                        "user_role": role,
                    },
                    len(st.session_state.messages),
                )
        assistant_message = {
            "role": "assistant",
            "content": response.get("answer", ""),
            "steps": response.get("steps", []),
            "citations": response.get("citations", []),
            "question": prompt,
            "user_id": user_id,
            "user_role": role,
        }
        st.session_state.messages.append(assistant_message)


def _upload_document(upload_file, user_id: str):
    try:
        files = {"file": (upload_file.name, upload_file.getvalue())}
        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=30,
        )
        if response.ok:
            st.success("Document queued for ingestion.")
        else:
            st.error(f"Upload failed: {response.text}")
    except requests.RequestException as exc:
        st.error(f"Upload error: {exc}")


def _chat(prompt: str, user_id: str, role: str) -> dict:
    payload = {
        "query": prompt,
        "user_id": user_id,
        "role": role,
    }
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=60,
        )
        if response.ok:
            return response.json()
        return {"answer": f"Request failed: {response.text}", "steps": []}
    except requests.RequestException as exc:
        return {"answer": f"Request error: {exc}", "steps": []}


def _get_workflow():
    if "workflow" not in st.session_state:
        retriever = SecureRetriever()
        st.session_state.workflow = build_agent_graph(retriever)
    return st.session_state.workflow


def _stream_chat(prompt: str, user_id: str, role: str, placeholder) -> dict:
    workflow = _get_workflow()
    state = {
        "question": prompt,
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
    full_response = ""
    last_state = None
    for event in workflow.stream(state, stream_mode="values"):
        last_state = event
        generation = event.get("generation") if isinstance(event, dict) else None
        if generation and generation != full_response:
            full_response = generation
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)
    if not last_state:
        return {"answer": "No response.", "steps": []}
    documents = last_state.get("documents", [])
    citations = [
        doc.metadata.get("source")
        for doc in documents
        if doc.metadata.get("source")
    ]
    return {
        "answer": last_state.get("generation", ""),
        "steps": last_state.get("steps", []),
        "citations": citations,
    }


def _render_feedback_buttons(message: dict, index: int) -> None:
    col_up, col_down, col_spacer = st.columns([1, 1, 8])
    with col_up:
        if st.button("👍", key=f"thumbs_up_{index}"):
            _save_feedback(message, "up")
            st.toast("Thanks for the feedback!", icon="✅")
    with col_down:
        if st.button("👎", key=f"thumbs_down_{index}"):
            _save_feedback(message, "down")
            st.toast("Feedback logged.", icon="📝")
    with col_spacer:
        st.caption("Give feedback to improve evaluation.")


def _save_feedback(message: dict, rating: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "question": message.get("question"),
        "answer": message.get("content"),
        "user_id": message.get("user_id"),
        "user_role": message.get("user_role"),
        "citations": message.get("citations", []),
        "steps": message.get("steps", []),
    }
    feedback_dir = os.path.dirname(FEEDBACK_PATH)
    os.makedirs(feedback_dir, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
