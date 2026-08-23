"""RAG pipeline: sanitization-aware ingestion, embeddings, vector store, and the engine.

Re-exports the pieces the composition root wires together, so callers can do
``from kuhaku.tools.rag import RAGEngine`` without knowing the module split.
"""

from .cache import QueryAnswerCache, compute_cache_key
from .config import RAGSettings
from .chunking import (
    Chunker,
    ParagraphChunker,
    StructuralChunker,
    build_chunker,
    chunk_document,
)
from .contradiction_detector import (
    ContradictionDetector,
    ContradictionPair,
    ContradictionResult,
)
from .embeddings import EmbeddingProvider, SentenceTransformerEmbeddings, build_embedding_provider
from .engine import RAGEngine
from .evaluation_target import RAGTargetAdapter
from .ingestion import (
    EmptyContent,
    UnsupportedFileType,
    UploadTooLarge,
    extract_text,
    ingest,
    ingest_single_document,
    load_corpus,
)
from .loader import DocumentLoader, FileSystemLoader
from .messages import DEFAULT_ENGINE_MESSAGES, EngineMessages
from .models import (
    ACCESS_TAG_INTERNAL,
    ACCESS_TAG_PUBLIC,
    ACCESS_TAG_RESTRICTED,
    Answer,
    Chunk,
    Citation,
    Document,
    RetrievedChunk,
)
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker, Reranker
from .retriever import (
    DenseRetriever,
    HybridRetriever,
    Retriever,
    SparseRetriever,
    is_entitled,
    reciprocal_rank_fusion,
)
from .sparse_retriever import BM25Retriever, StoreBackedBM25Retriever, build_bm25_from_store
from .vectorstore import ChromaVectorStore, VectorStore

__all__ = [
    "RAGSettings",
    # Domain models
    "Document",
    "Chunk",
    "RetrievedChunk",
    "Citation",
    "Answer",
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "build_embedding_provider",
    "RAGEngine",
    "RAGTargetAdapter",
    "Chunker",
    "ParagraphChunker",
    "StructuralChunker",
    "build_chunker",
    "chunk_document",
    "DocumentLoader",
    "FileSystemLoader",
    "ingest",
    "ingest_single_document",
    "extract_text",
    "UnsupportedFileType",
    "EmptyContent",
    "UploadTooLarge",
    "load_corpus",
    "ChromaVectorStore",
    "VectorStore",
    # retrieval layer
    "Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "SparseRetriever",
    "reciprocal_rank_fusion",
    "is_entitled",
    "BM25Retriever",
    "StoreBackedBM25Retriever",
    "build_bm25_from_store",
    "Reranker",
    "CrossEncoderReranker",
    # Category 2: query-answer cache
    "QueryAnswerCache",
    "compute_cache_key",
    # Query rewriting
    "QueryRewriter",
    # Contradiction detection
    "ContradictionDetector",
    "ContradictionPair",
    "ContradictionResult",
    # Domain constants externalization
    "EngineMessages",
    "DEFAULT_ENGINE_MESSAGES",
    # Document-level access filtering: default tag vocabulary (Feature 6, suggestions only)
    "ACCESS_TAG_PUBLIC",
    "ACCESS_TAG_INTERNAL",
    "ACCESS_TAG_RESTRICTED",
]
