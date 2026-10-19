"""Sparse (lexical) retrieval via a hand-rolled Okapi BM25.

Complements dense retrieval: embeddings capture meaning but blur exact tokens, while BM25
nails literal matches — error codes (``PAY-6006``), endpoint paths, and English technical
terms that appear verbatim inside otherwise-Turkish questions.

Implemented by hand (no ``rank_bm25``) to stay dependency-free and consistent with the
project's "minimal hand-rolled pipeline" decision (D4). The scoring is standard Okapi BM25:

    score(q, d) = Σ_t  IDF(t) · f(t,d)·(k1 + 1) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(t)      = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter

from kuhaku.core.auth import AuthContext
from kuhaku.tools.rag.models import Chunk, RetrievedChunk

from .retriever import is_entitled
from .vectorstore import VectorStore

logger = logging.getLogger("kuhaku.tools.rag.sparse_retriever")

# Unicode-aware word tokens, so Turkish diacritics (ı, ş, ğ, ü, ö, ç) survive.
# "PAY-5005" -> ["pay", "5005"]; the numeric part is highly discriminative.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into word tokens."""

    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over an in-memory list of chunks.

    Satisfies the ``Retriever`` protocol in :mod:`.retriever`.
    """

    strategy = "sparse"

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b

        # Inverted index: term -> {doc_index: term_frequency}
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_len: list[int] = []

        for idx, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.text)
            self._doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, {})[idx] = freq

        n_docs = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }
        logger.info("BM25 index built: %d chunks, %d terms", n_docs, len(self._postings))

    def count(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` highest-scoring chunks (score > 0 only)."""

        if not self._chunks or top_k <= 0:
            return []

        scores: dict[int, float] = {}
        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc_idx, freq in postings.items():
                norm = 1 - self._b + self._b * (self._doc_len[doc_idx] / self._avgdl)
                contribution = idf * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contribution

        # Entitlement, freshness and doc_type all narrow the candidate set *before*
        # ranking/truncation: a filtered-out high-scoring chunk must never crowd an
        # eligible lower-scoring one out of the top_k window. (Freshness and doc_type
        # used to be applied after truncation, so stale chunks could empty the result.)
        eligible = {
            idx: s
            for idx, s in scores.items()
            if is_entitled(self._chunks[idx], auth_context)
            and self._chunks[idx].is_fresh()
            and (doc_type is None or self._chunks[idx].doc_type == doc_type)
        }
        ranked = sorted(eligible.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=self._chunks[i], score=s) for i, s in ranked]


class StoreBackedBM25Retriever:
    """A :class:`BM25Retriever` sourced from a :class:`~.vectorstore.VectorStore`,
    rebuilt lazily after :meth:`refresh` rather than on every ingest.

    Both the dense and sparse sides now read the same ``Chunk`` objects (whatever the
    store holds), so the historical dense/sparse chunker-mismatch hazard -- two
    separately-chunked copies of the same corpus disagreeing under a shared chunk id --
    is structurally impossible: there is only ever one chunking of any given document,
    performed once at ingest time and written to the store.

    Satisfies the ``Retriever`` protocol in :mod:`.retriever`, exactly like the bare
    ``BM25Retriever`` it wraps.
    """

    strategy = "sparse"

    def __init__(self, store: VectorStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = store
        self._k1 = k1
        self._b = b
        self._bm25: BM25Retriever | None = None
        # A generation counter, not a dirty flag: with a flag, a refresh() landing while
        # a rebuild was reading the store was wiped out when the rebuild reset the flag,
        # and the new document was never indexed. The rebuild records the generation it
        # started from; any refresh() since then leaves it stale.
        self._lock = threading.Lock()
        self._generation = 0
        self._built_generation = -1
        # Writes that never call refresh() -- another process, a direct store.add/reset
        # -- are caught by the store's chunk count changing.
        self._built_count = -1

    def refresh(self) -> None:
        """Mark the cached index stale so the next :meth:`retrieve`/:meth:`count`
        rebuilds it from the store's current chunks.

        Deliberately lazy rather than rebuilding here: BM25's corpus-wide statistics
        (avgdl, IDF) depend on every chunk, so a rebuild is O(corpus size) regardless of
        how small the newly-ingested document was. Rebuilding immediately on every
        ``refresh()`` call would make a run of N ingests cost O(N * corpus_size) --
        quadratic in the number of documents ingested in a row (e.g.
        ``RAG.load_documents`` looping over a directory). Deferring the rebuild to the
        next actual query instead collapses any number of ingests that happen between
        two queries into a single rebuild.
        """

        with self._lock:
            self._generation += 1

    def _current(self) -> BM25Retriever:
        count = self._store.count()
        with self._lock:
            if (
                self._bm25 is not None
                and self._built_generation == self._generation
                and self._built_count == count
            ):
                return self._bm25
            generation = self._generation
        # Built outside the lock so a refresh() is never blocked behind a rebuild; two
        # racing rebuilds cost duplicate work, never a stale index.
        chunks = list(self._store.iter_chunks())
        bm25 = BM25Retriever(chunks, k1=self._k1, b=self._b)
        with self._lock:
            if generation >= self._built_generation:
                self._bm25 = bm25
                self._built_generation = generation
                self._built_count = len(chunks)
        return bm25

    def count(self) -> int:
        return self._current().count()

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        auth_context: AuthContext | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedChunk]:
        return self._current().retrieve(
            query, top_k, auth_context=auth_context, doc_type=doc_type,
        )


def build_bm25_from_store(
    store: VectorStore, *, k1: float = 1.5, b: float = 0.75
) -> StoreBackedBM25Retriever:
    """Build a BM25 index from the vector store's own chunks (the single source of
    truth for what is actually indexed), rather than re-reading and re-chunking source
    files from disk.

    The returned retriever rebuilds lazily on the next query after any :meth:`~
    StoreBackedBM25Retriever.refresh` call, so it reflects documents ingested into
    ``store`` after this factory ran -- see :class:`StoreBackedBM25Retriever`.
    """

    return StoreBackedBM25Retriever(store, k1=k1, b=b)
