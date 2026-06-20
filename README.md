# 🔍 News Research Tool — Production-grade RAG

A Streamlit application that answers research questions grounded in news articles,
built using production RAG engineering patterns typically found in AI platform teams.

**Core techniques demonstrated:**
- Hybrid retrieval (BM25 + FAISS vector search fused via RRF)
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
- Citation enforcement with post-generation validation
- CI-gated LLM-as-judge evaluation pipeline (GitHub Actions)

---

##  Demo video
![Demo](assets/demo.gif)

> **Query:** *"How do machine learning and large language models relate to artificial intelligence?"*
> **Result:** 5 passages retrieved across 3 sources · 100% citation rate · answer correctly
> acknowledges what the context does not cover rather than hallucinating.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           Scraping + Chunking               │
│  BeautifulSoup · 180-word overlapping       │
│  windows · User-Agent spoofing              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           Hybrid Retrieval                  │
│                                             │
│  BM25 (lexical)  +  FAISS (dense vector)    │
│       │                    │                │
│       └──── RRF fusion ────┘                │
│         top-20 candidates                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│        Cross-Encoder Reranking              │
│  ms-marco-MiniLM-L-6-v2 scores each         │
│  (query, passage) pair jointly              │
│  → top-k passages selected                  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│     Citation-Enforced Generation            │
│  Passages numbered [P1]..[Pn]               │
│  LLM instructed to cite after every claim   │
│  Citation rate validated post-generation    │
└─────────────────────────────────────────────┘
    │
    ▼
  Answer + Citation Rate + Unique Sources
  + Traceable passage expanders
```

---

## ✨ Key Features

### 1. Hybrid Retrieval (BM25 + Vector Search)
Dense retrieval (`all-MiniLM-L6-v2` + FAISS `IndexFlatL2`) captures semantic
similarity. Sparse retrieval (`BM25Okapi`) captures exact keyword matches.
Both ranked lists are fused via **Reciprocal Rank Fusion (RRF, k=60)** —
no score normalisation required, robust to scale differences between retrievers.

### 2. Cross-Encoder Reranking
The top-20 fused candidates are re-scored by a
`cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder, which encodes
the query and passage **jointly** rather than independently.
This is the same pattern used by Cohere Rerank, Jina Reranker, and similar
production retrieval services — significantly higher precision than bi-encoder
dot-product ranking alone.

### 3. Citation Enforcement
Passages are numbered `[P1]`…`[Pn]` and the LLM is prompted to cite the
relevant label after every factual claim. After generation, a citation rate
(fraction of retrieved passages cited) is computed and displayed as a metric
tile. If the model cites nothing, a visible warning is appended — the system
never silently returns an uncited answer.

### 4. CI-Gated Evaluation Pipeline
`evaluate.py` runs a fixed golden test suite against three RAG metrics using
an **LLM-as-judge** pattern. `.github/workflows/rag_eval.yml` runs this on
every PR — a metric regression blocks the merge and posts a formatted metric
table as a PR comment.

**Evaluation results (actual run):**

| Test | Citation rate | Context precision | Faithfulness | Result |
|------|--------------|-------------------|--------------|--------|
| What is Python used for? | 100% | 100% | 75% | ✅ PASS |
| Who founded Wikipedia? | 66% | 66% | 66% | ✅ PASS |
| What is machine learning? | 100% | 100% | 66% | ✅ PASS |
| **Average** | **88.89%** | **88.89%** | **69.44%** | **✅ ALL PASS** |

**Metric thresholds:**

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Citation rate | ≥ 50% | Core groundedness signal |
| Context precision | ≥ 60% | Retrieval relevance quality |
| Faithfulness | ≥ 50% | Calibrated against observed judge variance* |

> *LLM-as-judge faithfulness scoring exhibits non-determinism even at
> `temperature=0.0` due to distributed inference floating-point variance.
> Thresholds are calibrated empirically — a known production challenge with
> LLM-based evaluation. A more robust setup would average N=3 judge runs per
> test to reduce variance before comparing against thresholds.

---

## 🛠️ Tech Stack

| Component | Library / Service |
|-----------|------------------|
| UI | Streamlit |
| LLM | Groq API · `llama-3.3-70b-versatile` |
| Dense retrieval | FAISS `IndexFlatL2` + `sentence-transformers` |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Rank fusion | Reciprocal Rank Fusion (custom implementation) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Scraping | BeautifulSoup + urllib |
| Evaluation | LLM-as-judge (context precision, faithfulness, citation rate) |
| CI | GitHub Actions |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/news-research-rag.git
cd news-research-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Running the Evaluation Pipeline Locally

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"
python evaluate.py
```

```bash
# Mac / Linux
export GROQ_API_KEY="your_key_here"
python evaluate.py
```

Results print to terminal and are saved to `rag_eval_report.json`.

---

## ⚙️ CI Pipeline Setup (GitHub Actions)

1. Push this repo to GitHub
2. Go to **Settings → Secrets and variables → Actions → New repository secret**
3. Add secret: name `GROQ_API_KEY`, value: your Groq key
4. Open any Pull Request — the workflow runs automatically, gates on metric
   thresholds, and posts a metric table as a PR comment

---

## 📁 Project Structure

```
├── app.py                        # Streamlit app — hybrid RAG pipeline
├── evaluate.py                   # LLM-as-judge evaluation suite
├── requirements.txt              # Python dependencies
├── rag_eval_report.json          # Auto-generated evaluation report
├── .github/
│   └── workflows/
│       └── rag_eval.yml          # CI workflow — gates merges on RAG metrics
└── assets/
    └── demo_video.mp4            # Demo video
    └── demo.gif                  # Demo Gif
```

---

## 💡 Tested Example Queries

These reliably trigger multi-source retrieval (unique sources = 3):

**AI / ML topic:**
```
URLs:
  https://en.wikipedia.org/wiki/Artificial_intelligence
  https://en.wikipedia.org/wiki/Machine_learning
  https://en.wikipedia.org/wiki/Large_language_model

Query: How do machine learning and large language models relate to artificial intelligence?
```

**AI companies:**
```
URLs:
  https://en.wikipedia.org/wiki/OpenAI
  https://en.wikipedia.org/wiki/Google_DeepMind
  https://en.wikipedia.org/wiki/Anthropic

Query: Compare the founding missions and key research areas of leading AI companies.
```

> **Scraping note:** Paywalled or JavaScript-rendered sites (Moneycontrol,
> Economic Times homepage) return HTTP 403 or empty content.
> Use direct article URLs or Wikipedia for reliable scraping.

---

## 🔮 Known Limitations & Future Work

- **Single-hop retrieval** — no multi-hop reasoning across documents
- **No persistent index** — FAISS rebuilt per query; production would cache with a vector DB (Pinecone, Qdrant, Weaviate)
- **Scraping reliability** — replace BeautifulSoup with a news API (NewsAPI.org) for production
- **Brute-force FAISS** — `IndexFlatL2` is fine at this scale; large corpora need `IndexHNSWFlat` or `IndexIVFPQ`
- **Judge variance** — faithfulness scores vary run-to-run; averaging N=3 judge calls per test would improve CI reliability

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built to demonstrate production RAG engineering patterns:
hybrid retrieval · reranking · citation enforcement · CI-gated evaluation.*
