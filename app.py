"""
News Research Tool — Production-grade RAG demo
=============================================
Techniques demonstrated:
  1. Hybrid retrieval  — BM25 (lexical) + FAISS (dense vector) with RRF fusion
  2. Cross-encoder reranking — re-scores fused candidates for precision
  3. Citation enforcement  — LLM prompted to cite [P1], [P2]… validated before returning
  4. Structured RAG metrics logged per query (context precision proxy, citation rate)

A companion evaluate.py + .github/workflows/rag_eval.yml implements the
CI-gated evaluation pipeline that gates merges on metric regressions.
"""

import os
import re
import traceback
import math

import numpy as np
import faiss
import streamlit as st
from groq import Groq
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Cached resource loaders — loaded once per session, not on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    """Dense bi-encoder for vector retrieval (384-dim)."""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner=False)
def get_reranker():
    """
    Cross-encoder reranker: scores (query, passage) pairs jointly,
    giving much higher precision than bi-encoder dot-product alone.
    ms-marco-MiniLM-L-6-v2 is a standard production reranking checkpoint.
    """
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """
    Merge multiple ranked lists of chunk indices via RRF.
    RRF(d) = sum_r 1 / (k + rank_r(d))
    k=60 is the standard default from the original RRF paper.
    Returns indices sorted by descending fused score.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class NewsResearchTool:
    def __init__(self):
        self.client = None
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        self.embedding_dim = 384  # output dim of all-MiniLM-L6-v2

    # ------------------------------------------------------------------
    # Groq connection
    # ------------------------------------------------------------------
    def load_model(self, api_key: str) -> bool:
        try:
            if not api_key:
                st.error("No API key provided")
                return False
            self.client = Groq(api_key=api_key)
            models = self.client.models.list()
            st.success(
                f"Connected to Groq. Available models: {[m.id for m in models.data]}"
            )
            return True
        except Exception as e:
            st.error(f"Groq API connection error: {str(e)}")
            st.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------
    def scrape_url(self, url: str, timeout: int = 10) -> str:
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    )
                },
            )
            page = urlopen(request, timeout=timeout)
            soup = BeautifulSoup(page, "html.parser")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
            return text.strip()
        except Exception as e:
            st.error(f"URL scraping error for {url}: {str(e)}")
            return ""

    def process_urls(self, urls: list[str]) -> list[str]:
        documents = [self.scrape_url(url) for url in urls if url]
        return [doc for doc in documents if doc]

    # ------------------------------------------------------------------
    # Chunking — overlapping word-level windows
    # ------------------------------------------------------------------
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
        words = re.split(r"\s+", text.strip())
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - overlap
        return chunks

    def build_chunk_store(
        self, documents: list[str], urls: list[str]
    ) -> tuple:
        """
        Build both a FAISS dense index and a BM25 sparse index
        over all passage chunks across all scraped articles.
        Returns (faiss_index, bm25_index, all_chunks, chunk_sources).
        """
        all_chunks: list[str] = []
        chunk_sources: list[str] = []

        for doc, url in zip(documents, urls):
            doc_chunks = self.chunk_text(doc)
            all_chunks.extend(doc_chunks)
            chunk_sources.extend([url] * len(doc_chunks))

        if not all_chunks:
            return None, None, [], []

        # --- Dense index (FAISS) ---
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=False)
        embeddings_np = np.array(embeddings).astype("float32")
        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        faiss_index.add(embeddings_np)

        # --- Sparse index (BM25) ---
        tokenized = [chunk.lower().split() for chunk in all_chunks]
        bm25_index = BM25Okapi(tokenized)

        return faiss_index, bm25_index, all_chunks, chunk_sources

    # ------------------------------------------------------------------
    # Hybrid retrieval: BM25 + vector search fused via RRF
    # ------------------------------------------------------------------
    def hybrid_retrieve(
        self,
        faiss_index,
        bm25_index: BM25Okapi,
        all_chunks: list[str],
        chunk_sources: list[str],
        query: str,
        top_k: int = 3,
        candidate_pool: int = 20,
    ) -> tuple[list[str], list[str]]:
        """
        1. Dense retrieval  — top-candidate_pool chunks from FAISS
        2. Sparse retrieval — top-candidate_pool chunks from BM25
        3. RRF fusion       — merge both ranked lists
        4. Cross-encoder reranking — re-score top fused candidates
        5. Return top_k after reranking
        """
        if faiss_index is None or not all_chunks:
            return [], []

        n = len(all_chunks)
        pool = min(candidate_pool, n)

        # 1. Dense retrieval
        query_emb = self.embedder.encode(query).astype("float32").reshape(1, -1)
        _, dense_indices = faiss_index.search(query_emb, pool)
        dense_ranking = dense_indices[0].tolist()

        # 2. Sparse retrieval (BM25)
        bm25_scores = bm25_index.get_scores(query.lower().split())
        sparse_ranking = np.argsort(bm25_scores)[::-1][:pool].tolist()

        # 3. RRF fusion
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        candidates = fused[: min(pool, len(fused))]

        # 4. Cross-encoder reranking
        pairs = [(query, all_chunks[i]) for i in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )

        top_indices = [idx for idx, _ in reranked[: top_k]]
        retrieved_chunks = [all_chunks[i] for i in top_indices]
        retrieved_sources = [chunk_sources[i] for i in top_indices]
        return retrieved_chunks, retrieved_sources

    # ------------------------------------------------------------------
    # Generation with citation enforcement
    # ------------------------------------------------------------------
    def generate_response(
        self, context_chunks: list[str], query: str
    ) -> tuple[str, float]:
        """
        Returns (response_text, citation_rate).
        citation_rate = fraction of passage labels [P1]..[Pn] actually
        cited in the response — a proxy for groundedness.
        """
        if not self.client:
            st.error("Groq client not initialized.")
            return "Groq client not initialized.", 0.0

        # Build numbered context so the model can cite [P1], [P2], etc.
        numbered_context = "\n\n".join(
            f"[P{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        n_passages = len(context_chunks)
        citation_labels = ", ".join(f"[P{i+1}]" for i in range(n_passages))

        prompt = f"""You are a research assistant. Answer the question using ONLY the passages below.

PASSAGES:
{numbered_context}

QUESTION: {query}

INSTRUCTIONS:
- Every factual claim MUST be followed by the passage label it came from, e.g. [P1] or [P2].
- If the passages do not contain enough information, say so explicitly — do not guess.
- Structure your answer clearly.

ANSWER (with inline citations {citation_labels}):"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise research assistant. "
                            "Always cite the passage label [P1], [P2] etc. "
                            "after every factual claim."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,   # lower temp for factual citation tasks
                max_tokens=600,
                top_p=0.9,
            )
            response = chat_completion.choices[0].message.content.strip()

            # --- Citation enforcement check ---
            cited = set(
                re.findall(r"\[P(\d+)\]", response)
            )
            expected = set(str(i + 1) for i in range(n_passages))
            citation_rate = len(cited & expected) / max(len(expected), 1)

            # Soft enforcement: if no citations at all, append a warning
            if citation_rate == 0.0:
                response += (
                    "\n\n⚠️ *Note: The model did not include passage citations. "
                    "Treat this answer with extra caution.*"
                )

            return response, citation_rate

        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.error(traceback.format_exc())
            return f"Error generating response: {str(e)}", 0.0


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Groq News Research Tool", page_icon="📰")
    st.title("🔍 News Research Tool (RAG, powered by Groq)")
    st.caption(
        "Hybrid BM25 + vector retrieval · Cross-encoder reranking · Citation enforcement"
    )

    if "tool" not in st.session_state:
        st.session_state.tool = NewsResearchTool()
    tool = st.session_state.tool

    # Sidebar
    st.sidebar.header("🔑 Groq API Configuration")
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

    st.sidebar.header("🌐 News Sources")
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"News Article URL {i + 1}", key=f"url_{i}")
        if url:
            urls.append(url)

    if st.sidebar.button("🚀 Connect to Groq"):
        with st.spinner("Connecting to Groq..."):
            if tool.load_model(api_key):
                st.sidebar.success("Connected to Groq successfully!")
            else:
                st.sidebar.error("Connection failed. Check your API key.")

    # Main inputs
    query = st.text_input("📝 Enter your research question")
    top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=5, value=3)

    if st.button("🔬 Conduct Research"):
        if not api_key:
            st.warning("Please enter your Groq API Key")
            return
        if not urls:
            st.warning("Please enter at least one URL")
            return
        if not query:
            st.warning("Please enter a research question")
            return

        if not tool.client:
            if not tool.load_model(api_key):
                st.error("Failed to initialize Groq client.")
                return

        with st.spinner("Scraping articles..."):
            documents = tool.process_urls(urls)

        if not documents:
            st.error("Could not extract content from the provided URLs")
            return

        with st.spinner("Building BM25 + FAISS index..."):
            faiss_index, bm25_index, all_chunks, chunk_sources = (
                tool.build_chunk_store(documents, urls)
            )

        if faiss_index is None:
            st.error("Could not build search index from the scraped content")
            return

        with st.spinner("Hybrid retrieval → cross-encoder reranking..."):
            retrieved_chunks, retrieved_sources = tool.hybrid_retrieve(
                faiss_index, bm25_index, all_chunks, chunk_sources,
                query, top_k=top_k
            )

        if not retrieved_chunks:
            st.error("No relevant context found")
            return

        with st.spinner("Generating cited answer..."):
            response, citation_rate = tool.generate_response(retrieved_chunks, query)

        # --- Results ---
        st.header("🧠 Research Findings")
        st.write(response)

        # RAG quality metrics panel
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Passages retrieved", len(retrieved_chunks))
        col2.metric("Citation rate", f"{citation_rate:.0%}")
        unique_src_count = len(set(retrieved_sources))
        col3.metric("Unique sources", unique_src_count)

        with st.expander("ℹ️ How retrieval works"):
            st.markdown(
                """
**Step 1 — Hybrid retrieval**
Both a BM25 (lexical) and a FAISS (dense vector) index are queried independently.
BM25 excels at exact keyword matches; dense retrieval captures semantic similarity.

**Step 2 — RRF fusion**
Reciprocal Rank Fusion merges the two ranked lists without requiring score normalisation.

**Step 3 — Cross-encoder reranking**
A `ms-marco-MiniLM-L-6-v2` cross-encoder re-scores each (query, passage) pair jointly,
giving significantly higher precision than bi-encoder ranking alone.

**Step 4 — Citation enforcement**
The LLM is instructed to cite [P1], [P2]… after every claim.
The citation rate above shows what fraction of passages were actually cited.
                """
            )

        # Retrieved passages
        st.subheader("📌 Retrieved passages & sources")

        source_chunk_count: dict[str, int] = {}
        for src in retrieved_sources:
            source_chunk_count[src] = source_chunk_count.get(src, 0) + 1

        source_passage_index: dict[str, int] = {}
        for i, (chunk, source) in enumerate(zip(retrieved_chunks, retrieved_sources)):
            source_passage_index[source] = source_passage_index.get(source, 0) + 1
            total = source_chunk_count[source]
            label = (
                f"[P{i+1}] Passage {source_passage_index[source]} of {total} — {source}"
                if total > 1
                else f"[P{i+1}] Source: {source}"
            )
            with st.expander(label):
                st.write(chunk)

        # Unique sources summary
        unique_sources = list(dict.fromkeys(retrieved_sources))
        st.markdown("---")
        st.markdown(f"**📰 Unique articles consulted ({len(unique_sources)}):**")
        for j, src in enumerate(unique_sources, 1):
            st.markdown(f"{j}. [{src}]({src})")


if __name__ == "__main__":
    main()
