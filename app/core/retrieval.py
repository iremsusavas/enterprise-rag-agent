from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SecureRetriever:
    def __init__(
        self,
        vector_store=None,
        provider: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 25,
        rerank_k: int = 5,
        search_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._provider = (provider or os.getenv("VECTOR_DB_PROVIDER", "weaviate")).lower()
        self._alpha = alpha
        self._top_k = top_k
        self._rerank_k = rerank_k
        self._search_k = search_k

    def invoke(self, query: str, user_roles: list[str]) -> List[Document]:
        if not user_roles:
            return []
        candidates = self._retrieve_candidates(query, user_roles)
        if not candidates:
            return []
        reranked = _rerank_documents(query, candidates, self._rerank_k)
        return reranked

    def _retrieve_candidates(self, query: str, user_roles: list[str]) -> List[Document]:
        if self._provider == "weaviate":
            return self._hybrid_weaviate(query, user_roles)
        if self._provider == "pinecone":
            return self._hybrid_pinecone(query, user_roles)
        raise ValueError(f"Unsupported vector DB provider: {self._provider}")

    def _hybrid_weaviate(self, query: str, user_roles: list[str]) -> List[Document]:
        client, index_name, text_key = _get_weaviate_client(self._vector_store)
        where_filter = {
            "path": ["role_access"],
            "operator": "ContainsAny",
            "valueTextArray": user_roles,
        }
        limit = max(self._top_k, self._search_k)
        bm25_docs = _weaviate_bm25_query(
            client=client,
            index_name=index_name,
            text_key=text_key,
            query=query,
            where_filter=where_filter,
            limit=limit,
        )
        vector_docs = _weaviate_vector_query(
            client=client,
            index_name=index_name,
            text_key=text_key,
            query=query,
            where_filter=where_filter,
            limit=limit,
        )
        return _merge_documents(bm25_docs, vector_docs, self._top_k)

    def _hybrid_pinecone(self, query: str, user_roles: list[str]) -> List[Document]:
        filter_payload = {"role_access": {"$in": user_roles}}

        if self._vector_store is None:
            self._vector_store = _get_pinecone_store()

        if hasattr(self._vector_store, "_index") and os.getenv(
            "PINECONE_USE_SPARSE", "false"
        ).lower() in {"true", "1", "yes"}:
            return _pinecone_hybrid_query(
                self._vector_store._index,
                query,
                filter_payload,
                top_k=self._top_k,
            )

        if hasattr(self._vector_store, "similarity_search"):
            docs = self._vector_store.similarity_search(
                query, k=self._search_k, filter=filter_payload
            )
            logger.warning("Pinecone hybrid disabled; using dense-only search.")
            return docs

        logger.warning("Pinecone vector store does not support similarity_search.")
        return []


def _rerank_documents(
    query: str, documents: List[Document], top_k: int
) -> List[Document]:
    if not documents:
        return []

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        return _rerank_with_cohere(query, documents, top_k, cohere_api_key)

    try:
        return _rerank_with_flagembedding(query, documents, top_k)
    except Exception:
        logger.debug("FlagEmbedding not available; fallback to lexical score.", exc_info=True)

    scored = [(doc, _lexical_score(query, doc.page_content)) for doc in documents]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def _rerank_with_cohere(
    query: str, documents: List[Document], top_k: int, api_key: str
) -> List[Document]:
    import cohere

    client = cohere.Client(api_key)
    results = client.rerank(
        model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        query=query,
        documents=[doc.page_content for doc in documents],
        top_n=min(top_k, len(documents)),
    )
    reranked = [documents[item.index] for item in results.results]
    return reranked


def _rerank_with_flagembedding(
    query: str, documents: List[Document], top_k: int
) -> List[Document]:
    from FlagEmbedding import FlagReranker

    model_name = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker = FlagReranker(model_name, use_fp16=True)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.compute_score(pairs)
    scored = sorted(
        zip(documents, scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [doc for doc, _ in scored[:top_k]]


def _lexical_score(query: str, content: str) -> float:
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return 0.0
    content_terms = content.lower().split()
    hits = sum(1 for term in content_terms if term in query_terms)
    return hits / max(1, len(content_terms))


def _get_weaviate_client(vector_store=None):
    if vector_store and hasattr(vector_store, "_client"):
        client = vector_store._client
        index_name = getattr(vector_store, "_index_name", None)
        text_key = getattr(vector_store, "_text_key", "text")
        if index_name:
            return client, index_name, text_key

    import weaviate

    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", ""),
        },
    )
    index_name = os.getenv("WEAVIATE_INDEX", "EnterpriseKnowledge")
    text_key = os.getenv("WEAVIATE_TEXT_KEY", "text")
    return client, index_name, text_key


def _weaviate_bm25_query(
    client,
    index_name: str,
    text_key: str,
    query: str,
    where_filter: dict,
    limit: int,
) -> List[Document]:
    response = (
        client.query.get(index_name, [text_key, "source", "role_access"])
        .with_where(where_filter)
        .with_bm25(query=query)
        .with_limit(limit)
        .do()
    )
    if response.get("errors"):
        logger.warning("Weaviate BM25 error: %s", response.get("errors"))
        return []
    items = response.get("data", {}).get("Get", {}).get(index_name) or []
    return [_weaviate_item_to_doc(item, text_key) for item in items]


def _weaviate_vector_query(
    client,
    index_name: str,
    text_key: str,
    query: str,
    where_filter: dict,
    limit: int,
) -> List[Document]:
    embeddings = _get_embeddings()
    vector = embeddings.embed_query(query)
    response = (
        client.query.get(index_name, [text_key, "source", "role_access"])
        .with_where(where_filter)
        .with_near_vector({"vector": vector})
        .with_limit(limit)
        .do()
    )
    if response.get("errors"):
        logger.warning("Weaviate vector error: %s", response.get("errors"))
        return []
    items = response.get("data", {}).get("Get", {}).get(index_name) or []
    return [_weaviate_item_to_doc(item, text_key) for item in items]


def _weaviate_item_to_doc(item: dict, text_key: str) -> Document:
    return Document(
        page_content=item.get(text_key, ""),
        metadata={
            "source": item.get("source"),
            "role_access": item.get("role_access"),
        },
    )


def _merge_documents(
    bm25_docs: List[Document],
    vector_docs: List[Document],
    limit: int,
) -> List[Document]:
    merged = []
    seen = set()
    for doc in bm25_docs + vector_docs:
        key = (doc.page_content or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def _get_pinecone_store():
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "")
    index_name = os.getenv("PINECONE_INDEX", "enterprise-knowledge")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for Pinecone")

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    index = pc.Index(index_name)

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    return PineconeVectorStore(index=index, embedding=embeddings)


def _pinecone_hybrid_query(index, query: str, filter_payload: dict, top_k: int):
    embeddings = _get_embeddings()
    dense_vector = embeddings.embed_query(query)

    try:
        from pinecone_text.sparse import BM25Encoder
    except Exception as exc:
        logger.warning(
            "pinecone-text not available; falling back to dense-only.",
            exc_info=True,
        )
        return _pinecone_dense_query(index, dense_vector, filter_payload, top_k)

    encoder_path = os.getenv("PINECONE_BM25_ENCODER_PATH")
    if not encoder_path:
        logger.warning(
            "PINECONE_BM25_ENCODER_PATH not set; falling back to dense-only."
        )
        return _pinecone_dense_query(index, dense_vector, filter_payload, top_k)

    encoder = BM25Encoder().load(encoder_path)
    sparse_vector = encoder.encode_queries(query)

    response = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_payload,
    )
    return _pinecone_to_documents(response)


def _pinecone_dense_query(index, dense_vector: list[float], filter_payload: dict, top_k: int):
    response = index.query(
        vector=dense_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_payload,
    )
    return _pinecone_to_documents(response)


def _pinecone_to_documents(response) -> List[Document]:
    matches = getattr(response, "matches", None) or response.get("matches", [])
    docs = []
    for match in matches:
        metadata = getattr(match, "metadata", None) or match.get("metadata", {})
        page_content = metadata.get("text") or metadata.get("content") or ""
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


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
