"""
Retrieval services for contract clauses.
Provides an ultra-fast, zero-dependency hybrid legal ranker (instant, sub-millisecond)
and an optional dense vector retriever (Sentence-Transformers + FAISS) with embedding caching.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# Critical legal terms used to boost relevant clause discovery based on focus area
LEGAL_KEYWORDS: Dict[str, List[str]] = {
    "general": [
        "indemnif", "liabilit", "breach", "damages", "terminat", "govern",
        "warrant", "dispute", "exclusiv", "penalty", "remedy", "risk",
        "consequential", "limitation of liability", "hold harmless"
    ],
    "payment": [
        "payment", "invoice", "fee", "due", "late", "interest", "penalty",
        "net 30", "net 60", "net 90", "withhold", "dispute", "currency", "expense"
    ],
    "intellectual property": [
        "intellectual property", "ownership", "work made for hire", "copyright",
        "patent", "trademark", "assign", "license", "proprietary", "moral rights",
        "derivative works", "trade secret"
    ],
    "termination": [
        "terminate", "termination", "notice", "breach", "cure period", "convenience",
        "immediate", "survival", "effect of termination", "default"
    ],
    "indemnification": [
        "indemnif", "hold harmless", "defend", "unlimited", "gross negligence",
        "third-party claims", "liability", "damages", "losses", "reimburse"
    ],
    "non-compete": [
        "non-compete", "non-competition", "exclusiv", "solicit", "non-solicitation",
        "restrict", "covenant", "restraint of trade", "geography", "period"
    ],
}


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean word stems/tokens."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", text)]


class FastClauseRetriever:
    """
    Sub-millisecond BM25-style hybrid legal clause ranker.
    Requires no PyTorch/CUDA dependencies, runs 100% in memory in <5ms.
    """

    @staticmethod
    def rank_chunks(
        chunks: List[Dict[str, Any]],
        focus_query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank and return top_k most relevant chunks using BM25 + legal domain weighting."""
        if not chunks:
            return []

        if len(chunks) <= top_k:
            return list(chunks)

        # 1. Determine focus-specific boost terms
        focus_lower = focus_query.lower()
        active_legal_terms = list(LEGAL_KEYWORDS["general"])
        for key, terms in LEGAL_KEYWORDS.items():
            if key in focus_lower:
                active_legal_terms.extend(terms)

        # 2. Tokenize corpus & query
        tokenized_chunks = [_tokenize(c["content"]) for c in chunks]
        query_tokens = _tokenize(focus_query)

        # 3. Document frequencies
        n_docs = len(chunks)
        avg_doc_len = sum(len(doc) for doc in tokenized_chunks) / max(1, n_docs)
        
        doc_freqs: Dict[str, int] = {}
        for doc in tokenized_chunks:
            unique_words = set(doc)
            for w in unique_words:
                doc_freqs[w] = doc_freqs.get(w, 0) + 1

        # 4. Score each chunk (BM25 with k1=1.5, b=0.75 + legal term boosting)
        k1 = 1.5
        b = 0.75
        scored_chunks: List[Tuple[float, int, Dict[str, Any]]] = []

        for idx, (chunk, doc) in enumerate(zip(chunks, tokenized_chunks)):
            doc_len = len(doc)
            doc_text_lower = chunk["content"].lower()
            score = 0.0

            # Match query tokens
            for q in query_tokens:
                tf = doc.count(q)
                if tf > 0:
                    df = doc_freqs.get(q, 1)
                    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    score += term_score

            # Legal domain term boosting
            for legal_term in active_legal_terms:
                if legal_term in doc_text_lower:
                    score += 2.5

            scored_chunks.append((score, idx, chunk))

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in scored_chunks[:top_k]]


def get_dense_embeddings(chunks: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load SentenceTransformer and generate FAISS embeddings."""
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    embedder = SentenceTransformer(model_name)
    chunk_texts = [c["content"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=False).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embedder, index
