from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from celery import Celery
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.ingestion.loader import load_documents


DEFAULT_ROLE_ACCESS = ["hr", "admin"]
ALL_ROLES = [
    role.strip()
    for role in os.getenv("DEFAULT_ROLE_ACCESS", "intern,employee,hr,admin").split(",")
    if role.strip()
]
CONFIDENTIAL_ROLES = [
    role.strip()
    for role in os.getenv("CONFIDENTIAL_ROLE_ACCESS", "admin").split(",")
    if role.strip()
]
LLM_ROLE_MODEL = os.getenv("RBAC_LLM_MODEL", "gpt-4o-mini")


class RoleDecision(BaseModel):
    roles: List[str] = Field(
        description="List of allowed roles for this chunk."
    )




@dataclass(frozen=True)
class IngestionConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_db_provider: str = os.getenv("VECTOR_DB_PROVIDER", "weaviate")


celery_app = Celery(
    "enterprise_rag_agent",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


@celery_app.task(name="ingestion.process_document")
def process_document(
    file_path: str,
    role_access: Optional[List[str]] = None,
    source: Optional[str] = None,
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        print(f"[INGESTION LOG] OPENAI_API_KEY loaded: {api_key[:5]}***")
    else:
        print("[INGESTION LOG] OPENAI_API_KEY missing")
    config = IngestionConfig()
    documents = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    source_name = source or os.path.basename(file_path)
    enriched_chunks = _apply_metadata(
        chunks,
        role_access=role_access,
        source=source_name,
    )

    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings, provider=config.vector_db_provider)
    _delete_existing_by_source(vector_store, source_name)
    vector_store.add_documents(enriched_chunks)

    return {
        "chunks_processed": len(enriched_chunks),
        "source": source_name,
        "role_access": role_access or "auto",
    }


def _apply_metadata(
    documents: Iterable[Document],
    role_access: Optional[List[str]],
    source: str,
) -> List[Document]:
    enriched = []
    for doc in documents:
        metadata = dict(doc.metadata or {})
        chunk_roles = role_access or _resolve_role_access(doc.page_content)
        _log_role_assignment(doc.page_content, chunk_roles)
        metadata.update(
            {
                "role_access": chunk_roles,
                "source": source,
            }
        )
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))
    return enriched


def _resolve_role_access(text: str) -> List[str]:
    try:
        return _resolve_role_access_llm(text)
    except Exception as e:
        print(f"❌ LLM ERROR: {e}")
        return ["admin"]


def _log_role_assignment(text: str, roles: List[str]) -> None:
    chunk_text = " ".join((text or "").split())
    snippet = chunk_text[:20] + ("..." if len(chunk_text) > 20 else "")
    print(f"Snippet: {snippet} -> ROLES: {roles}")


def _resolve_role_access_llm(text: str) -> List[str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM RBAC classification.")

    from langchain_openai import ChatOpenAI

    parser = PydanticOutputParser(pydantic_object=RoleDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Data Security Officer. Analyze the following text "
                    "snippet from a corporate document. Classify its sensitivity "
                    "into one of these roles:\n"
                    "- 'admin': If it contains sensitive financial data, salaries, "
                    "executive bonuses, acquisitions, or explicit 'Confidential' markers.\n"
                    "- 'employee': If it contains internal benefits, insurance details, "
                    "or internal operational procedures not for interns.\n"
                    "- 'intern': If it contains general policies, handbook introductions, "
                    "mission statements, password policies, or work arrangements.\n\n"
                    "Output ONLY a JSON object: {{\"roles\": [\"...list of allowed roles...\"]}}\n"
                    "Note: 'intern' implies ['intern', 'employee', 'admin']. "
                    "'employee' implies ['employee', 'admin']."
                ),
            ),
            ("user", "{chunk}\n\n{format_instructions}"),
        ]
    )
    llm = ChatOpenAI(model=LLM_ROLE_MODEL, temperature=0)
    chain = prompt | llm
    chunk = (text or "")[:3000]
    response = chain.invoke(
        {"chunk": chunk, "format_instructions": parser.get_format_instructions()}
    )
    response_content = getattr(response, "content", str(response))
    print(f"🔍 RAW LLM OUTPUT: {response_content}")
    result = parser.parse(response_content)
    roles = [role for role in result.roles if role in ALL_ROLES]
    return roles or ["admin"]




def _get_embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "openai")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        )

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    raise ValueError(f"Unsupported embeddings provider: {provider}")


def _get_vector_store(embeddings, provider: str):
    provider = provider.lower()

    if provider == "weaviate":
        return _get_weaviate_store(embeddings)

    if provider == "pinecone":
        return _get_pinecone_store(embeddings)

    raise ValueError(f"Unsupported vector DB provider: {provider}")


def _get_weaviate_store(embeddings):
    import weaviate
    from langchain_community.vectorstores import Weaviate

    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", ""),
        },
    )
    index_name = os.getenv("WEAVIATE_INDEX", "EnterpriseKnowledge")
    _ensure_weaviate_schema(client, index_name)
    return Weaviate(client, index_name, "text", embeddings)


def _ensure_weaviate_schema(client, index_name: str) -> None:
    if client.schema.exists(index_name):
        return

    schema = {
        "class": index_name,
        "description": "Enterprise knowledge chunks",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "role_access", "dataType": ["text[]"]},
        ],
    }
    client.schema.create_class(schema)


def _delete_existing_by_source(vector_store, source: str) -> None:
    client = getattr(vector_store, "_client", None)
    index_name = getattr(vector_store, "_index_name", None)
    if not client or not index_name:
        return
    try:
        where_filter = {
            "path": ["source"],
            "operator": "Equal",
            "valueText": source,
        }
        client.batch.delete_objects(
            class_name=index_name,
            where=where_filter,
            dry_run=False,
        )
    except Exception as exc:
        print(f"[INGESTION LOG] Cleanup failed for source={source}: {exc}")


def _get_pinecone_store(embeddings):
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "")
    index_name = os.getenv("PINECONE_INDEX", "enterprise-knowledge")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for Pinecone")

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)
