"""Embedding / reranking models shared by ingestion and retrieval.

Ingestion and querying MUST use the same dense model; keeping both behind this
module (and recording the model name in the collection metadata) prevents the
query/document model mismatch the original code had.
"""
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from lawrag.config import get_settings


@lru_cache
def dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().dense_model)


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=get_settings().sparse_model)


@lru_cache
def reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=get_settings().rerank_model)


def embed_query(text: str) -> list[float]:
    # query_embed adds the BGE retrieval instruction prefix for queries.
    return next(iter(dense_model().query_embed(text))).tolist()


def embed_documents(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return [v.tolist() for v in dense_model().embed(texts, batch_size=batch_size)]


def sparse_query(text: str):
    return next(iter(sparse_model().query_embed(text)))


def sparse_documents(texts: list[str], batch_size: int = 64):
    return list(sparse_model().embed(texts, batch_size=batch_size))


def rerank(query: str, texts: list[str]) -> list[float]:
    return list(reranker().rerank(query, texts))
