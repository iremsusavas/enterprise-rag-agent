from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Iterable, Optional, Tuple

import redis


class SemanticCache:
    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.90,
        key_prefix: str = "semantic_cache",
    ) -> None:
        self._redis = redis.Redis.from_url(
            redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        self._ttl_seconds = ttl_seconds
        self._similarity_threshold = similarity_threshold
        self._key_prefix = key_prefix

    def check_cache(self, query: str) -> Optional[str]:
        query_embedding = self._embed_query(query)
        best_answer = None
        best_score = -1.0

        for key, answer, embedding in self._scan_cache_entries():
            score = _cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_answer = answer

        if best_score >= self._similarity_threshold:
            return best_answer
        return None

    def store_cache(self, query: str, answer: str) -> None:
        embedding = self._embed_query(query)
        cache_key = self._cache_key(query)
        payload = {
            "query": query,
            "answer": answer,
            "embedding": embedding,
        }
        self._redis.hset(cache_key, mapping={"payload": json.dumps(payload)})
        self._redis.expire(cache_key, self._ttl_seconds)

    def _cache_key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:{digest}"

    def _scan_cache_entries(self) -> Iterable[Tuple[str, str, list[float]]]:
        cursor = 0
        pattern = f"{self._key_prefix}:*"
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            for key in keys:
                payload = self._redis.hget(key, "payload")
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                embedding = data.get("embedding")
                answer = data.get("answer")
                if not embedding or not answer:
                    continue
                yield key, answer, embedding
            if cursor == 0:
                break

    def _embed_query(self, query: str) -> list[float]:
        provider = os.getenv("EMBEDDINGS_PROVIDER", "openai")
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
            )
        elif provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv(
                    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")

        return embeddings.embed_query(query)


def _cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector_a, vector_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0:
        return 0.0
    return dot / denom
