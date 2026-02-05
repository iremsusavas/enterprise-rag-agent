from __future__ import annotations

import os
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.core.grader import HallucinationGrader, RetrievalGrader
from app.core.retrieval import SecureRetriever


class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    user_roles: List[str]
    steps: List[str]
    rewrite_query: bool
    retries: int
    route: str
    grounded: bool
    hallucination_retries: int
    force_grounded: bool


class RouterDecision(BaseModel):
    route: Literal["retrieve", "direct"] = Field(
        description="Routing decision for the question."
    )


def build_agent_graph(
    retriever: SecureRetriever,
    llm=None,
    max_retries: int = 2,
    max_hallucination_retries: int = 2,
    min_relevant: int = 2,
):
    fast_llm = _get_chat_model(
        model=os.getenv("FAST_LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    smart_llm = llm or _get_chat_model(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        temperature=0,
    )
    retrieval_grader = RetrievalGrader(fast_llm)
    hallucination_grader = HallucinationGrader(fast_llm)
    query_rewriter = _get_query_rewriter(fast_llm)
    generator = _get_generator(smart_llm)
    grounded_generator = _get_grounded_generator(smart_llm)
    router = _get_router(fast_llm)

    def route_node(state: AgentState) -> AgentState:
        decision = _route_question(router, state["question"])
        state["route"] = decision
        state["steps"].append(f"route:{decision}")
        return state

    def retrieve_node(state: AgentState) -> AgentState:
        documents = retriever.invoke(state["question"], state["user_roles"])
        state["documents"] = documents
        state["steps"].append(f"retrieve:{len(documents)}")
        return state

    def grade_documents_node(state: AgentState) -> AgentState:
        filtered = []
        for doc in state.get("documents", []):
            grade = retrieval_grader.grade(state["question"], doc)
            if grade.relevant:
                filtered.append(doc)
        state["documents"] = filtered
        state["rewrite_query"] = len(filtered) < min_relevant
        state["steps"].append(f"grade:{len(filtered)}")
        return state

    def transform_query_node(state: AgentState) -> AgentState:
        state["question"] = query_rewriter.invoke({"question": state["question"]})
        state["retries"] = state.get("retries", 0) + 1
        state["steps"].append("rewrite")
        return state

    def generate_node(state: AgentState) -> AgentState:
        docs_text = "\n\n".join(doc.page_content for doc in state.get("documents", []))
        chain = grounded_generator if state.get("force_grounded") else generator
        response = chain.invoke({"question": state["question"], "context": docs_text})
        state["generation"] = response
        state["steps"].append("generate")
        return state

    def check_hallucination_node(state: AgentState) -> AgentState:
        grade = hallucination_grader.grade(
            state.get("documents", []),
            state.get("generation", ""),
        )
        state["grounded"] = grade.grounded
        state["steps"].append(f"grounded:{grade.grounded}")
        if not grade.grounded:
            state["force_grounded"] = True
            state["hallucination_retries"] = state.get("hallucination_retries", 0) + 1
        return state

    graph = StateGraph(AgentState)
    graph.add_node("router", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_edge,
        {"retrieve": "retrieve", "direct": "generate"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: _grade_edge(state, max_retries),
        {"generate": "generate", "transform_query": "transform_query"},
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        lambda state: _hallucination_edge(state, max_hallucination_retries),
        {"end": END, "generate": "generate"},
    )

    return graph.compile()


def _route_edge(state: AgentState) -> str:
    return state.get("route", "retrieve")


def _grade_edge(state: AgentState, max_retries: int) -> str:
    if state.get("rewrite_query") and state.get("retries", 0) < max_retries:
        return "transform_query"
    return "generate"


def _hallucination_edge(state: AgentState, max_hallucination_retries: int) -> str:
    if state.get("grounded"):
        return "end"
    if state.get("hallucination_retries", 0) >= max_hallucination_retries:
        return "end"
    return "generate"


def _get_chat_model(model: str, temperature: float):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
    )


def _get_router(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a router for an enterprise assistant. Decide whether "
                    "the user question requires retrieval from documents or can be "
                    "answered as casual conversation."
                ),
            ),
            ("user", "Question:\n{question}\n\nReturn route: retrieve or direct."),
        ]
    )
    return prompt | llm.with_structured_output(RouterDecision)


def _route_question(router, question: str) -> str:
    try:
        decision = router.invoke({"question": question})
        return decision.get("route", "retrieve")
    except Exception:
        return "retrieve"


def _get_query_rewriter(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the question to improve document retrieval. Keep meaning.",
            ),
            ("user", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _get_generator(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a precise enterprise assistant. Use the provided "
                    "context to answer. If the context is empty or insufficient, "
                    "say you do not have enough information."
                ),
            ),
            ("user", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _get_grounded_generator(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a precise enterprise assistant. Only answer using the "
                    "provided context. If the context does not support the answer, "
                    "say you do not have enough information."
                ),
            ),
            ("user", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )
    return prompt | llm | StrOutputParser()
