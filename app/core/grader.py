from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RetrievalGrade(BaseModel):
    relevant: bool = Field(
        description="True if the document is relevant to the question."
    )
    rationale: str = Field(description="Short justification for the decision.")


class HallucinationGrade(BaseModel):
    grounded: bool = Field(
        description="True if the answer is grounded in the provided documents."
    )
    rationale: str = Field(description="Short justification for the decision.")


class RetrievalGrader:
    def __init__(self, llm) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a grader assessing relevance of a retrieved document "
                        "to a user question. If the document contains keyword matches "
                        "or semantic relatedness to the question, grade it as relevant. "
                        "It does not need to be a stringent test. The goal is to filter "
                        "out erroneous retrievals. If the document mentions the entities, "
                        "topics, or numbers asked in the question, grade it as 'yes'. "
                        "Give a binary score 'yes' or 'no' to indicate whether the "
                        "document is relevant to the question."
                    ),
                ),
                (
                    "user",
                    "Question:\n{question}\n\nDocument:\n{document}\n\n"
                    "Is the document relevant?",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(RetrievalGrade)

    def grade(self, question: str, document: Document) -> RetrievalGrade:
        return self._chain.invoke(
            {"question": question, "document": document.page_content}
        )


class HallucinationGrader:
    def __init__(self, llm) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a strict factuality judge. Determine whether the "
                        "answer is fully grounded in the provided documents."
                    ),
                ),
                (
                    "user",
                    "Documents:\n{documents}\n\nAnswer:\n{generation}\n\n"
                    "Is the answer grounded in the documents?",
                ),
            ]
        )
        self._chain = prompt | llm.with_structured_output(HallucinationGrade)

    def grade(self, documents: List[Document], generation: str) -> HallucinationGrade:
        docs_text = "\n\n".join(doc.page_content for doc in documents)
        return self._chain.invoke(
            {"documents": docs_text, "generation": generation}
        )
