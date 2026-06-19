# 🔍 News Research Tool — Production-grade RAG

A Retrieval-Augmented Generation (RAG) system that scrapes news articles, retrieves the most relevant passages using **hybrid search**, reranks them with a **cross-encoder**, generates cited answers via **Groq's Llama 3.3**, and gates quality regressions with a **CI evaluation pipeline**.

Built to demonstrate production RAG techniques beyond basic vector search.

---

## 🖼️ Demo

> **Question:** "How do machine learning and large language models relate to artificial intelligence?"
>
> Sources: Wikipedia — Artificial Intelligence · Machine Learning · Large Language Models

##  Demo video
![Demo](assets/demo.gif)

**Results:**
- ✅ Passages retrieved from **3 unique sources**
- ✅ **100% citation rate** — every factual claim tagged [P1]–[P5]
- ✅ Answer honestly flags where context was insufficient (no hallucination)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Hybrid Retrieval              │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │  BM25 Index  │  │  FAISS (dense)  │  │
│  │  (lexical)   │  │  (semantic)     │  │
│  └──────┬───────┘  └────────┬────────┘  │
│         └────────┬──────────┘           │
│              RRF Fusion                 │
│         (top-20 candidates)             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│        Cross-Encoder Reranking           │
│   ms-marco-MiniLM-L-6-v2                │
│   scores (query, passage) pairs jointly  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│     Citation-Enforced Generation         │
│   Groq Llama-3.3-70b-versatile          │
│   Prompted to cite [P1], [P2]… inline   │
│   Citation rate validated post-response  │
└──────────────────┬───────────────────────┘
                   │
                   ▼
            Cited Answer
     + RAG Quality Metrics Panel
```

---

## ⚙️ Production RAG Techniques

### 1. Hybrid Retrieval (BM25 + Vector Search)
Pure vector search misses exact keyword matches; pure BM25 misses semantic similarity. This system runs both in parallel and merges results using **Reciprocal Rank Fusion (RRF)** — the standard fusion method that requires no score normalisation across retrieval systems.

```python
# Dense retrieval
_, dense_indices = faiss_index.search(query_emb, candidate_pool)

# Sparse retrieval
bm25_scores = bm25_index.get_scores(query.lower().split())
sparse_ranking = np.argsort(bm25_scores)[::-1][:candidate_pool]

# RRF fusion
fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
```

### 2. Cross-Encoder Reranking
Bi-encoder retrieval (FAISS) encodes query and passages independently — fast but imprecise. A cross-encoder reads the query and passage **together**, giving much higher relevance precision at the cost of speed. Applied on the fused top-20 candidates before returning top-k to the LLM.

```python
pairs = [(query, chunk) for chunk in candidates]
rerank_scores = cross_encoder.predict(pairs)
# Sort by rerank score, return top_k
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (standard production checkpoint).

### 3. Citation Enforcement
The LLM is prompted to tag every factual claim with the passage it came from ([P1], [P2]…). After generation, the system validates citation rate programmatically — if no citations appear, a visible warning is shown. This makes hallucinations traceable and auditable.

```python
cited = set(re.findall(r"\[P(\d+)\]", response))
citation_rate = len(cited & expected) / len(expected)
```

### 4. CI-Gated Evaluation Pipeline
A `evaluate.py` script runs a fixed golden test suite on every PR using **LLM-as-judge** scoring for three RAG metrics. A GitHub Actions workflow blocks merges if any metric falls below its threshold.

| Metric | Threshold | How measured |
|--------|-----------|--------------|
| Citation rate | ≥ 50% | Regex on model output |
| Context precision | ≥ 60% | LLM judges each passage for relevance |
| Faithfulness | ≥ 70% | LLM checks claims against source passages |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/news-research-tool.git
cd news-research-tool
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run the evaluation pipeline locally

```bash
export GROQ_API_KEY=your_key_here      # Mac/Linux
set GROQ_API_KEY=your_key_here         # Windows

python evaluate.py
```

---

## 📁 Project Structure

```
news-research-tool/
├── app.py                        # Streamlit app — full RAG pipeline
├── evaluate.py                   # CI evaluation script (LLM-as-judge)
├── requirements.txt              # Python dependencies
├── .github/
│   └── workflows/
│       └── rag_eval.yml          # GitHub Actions CI pipeline
└── assets/
    └── demo.png                  # Demo screenshot (add manually)
```

---

## 🧪 How the CI Pipeline Works

1. On every PR to `main`, GitHub Actions runs `evaluate.py`
2. Three golden test questions are run through the full RAG pipeline
3. Citation rate, context precision, and faithfulness are scored
4. A metric table is posted as a PR comment automatically
5. If any metric falls below threshold, the job fails and blocks the merge

To enable: add `GROQ_API_KEY` as a GitHub Actions repository secret under **Settings → Secrets and variables → Actions**.

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| Dense retrieval | FAISS (`faiss-cpu`) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Reranking | `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Generation | Groq API (`llama-3.3-70b-versatile`) |
| Web scraping | BeautifulSoup4 |

---

## ⚠️ Known Limitations

- **Paywalled / JS-rendered sites** (e.g. Moneycontrol, most Indian news portals) return 403 errors. Use Wikipedia or other open-access URLs for reliable scraping.
- **`IndexFlatL2`** (brute-force FAISS) is correct at this scale but would need to be replaced with `IndexIVFFlat` or `IndexHNSW` for production-scale document stores.
- The LLM-as-judge scores in `evaluate.py` use the same Groq API — in a stricter production setup you would use a separate, isolated judge model to avoid self-evaluation bias.

---

## 📄 License

MIT
