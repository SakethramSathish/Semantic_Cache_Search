<div align="center">

# 🔍 Semantic Cache Search

**A production-ready semantic search system with an intelligent fuzzy-cluster-aware cache layer**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/sklearn-GMM%20Clustering-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

*Built on the 20 Newsgroups dataset (~18,000 documents across 20 topic categories)*

</div>

---

## ✨ What Makes This Different

Most semantic search systems answer every query from scratch. This system answers it **once**, then recognises when a semantically equivalent rephrasing arrives — and serves the result from an intelligent, cluster-aware cache in microseconds instead of milliseconds.

> "Which GPU should I buy for PC gaming?" is recognised as the same question as "What are the best graphics cards for gaming?" — and served instantly.

The cache is not a naive key-value store. It uses **fuzzy topic clusters** to partition the cache space, so lookup time stays near-constant even as the cache grows to thousands of entries.

---

## 🏗️ System Architecture

```
                          ┌─────────────────────────────────┐
                          │         FastAPI Service          │
                          │         (app/main.py)           │
                          └──────────────┬──────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────────┐
                          │      Query Embedding Layer       │
                          │   fastembed · BAII/bge-small     │
                          │   384-dim · ONNX · No PyTorch   │
                          └──────────────┬──────────────────┘
                                         │
                     ┌───────────────────┼───────────────────┐
                     ▼                                       ▼
       ┌─────────────────────────┐           ┌──────────────────────────┐
       │    Semantic Cache        │     MISS  │      FAISS Index          │
       │   (app/cache.py)        │◄──────────│   IndexFlatIP (cosine)    │
       │                         │           │   ~18K normalised vectors  │
       │  Cluster-partitioned    │           └──────────────────────────┘
       │  Cosine similarity      │                         │
       │  Threshold: 0.75        │                         ▼
       └─────────────────────────┘           ┌──────────────────────────┐
                     │                        │      GMM Fuzzy Clusters   │
                     │  HIT                   │   20 components · spherical│
                     ▼                        │   predict_proba() output  │
             Return cached result             └──────────────────────────┘
```

---

## 🧠 Technical Design Decisions

### 1. Embedding Model — `BAAI/bge-small-en-v1.5` via `fastembed`
Rather than loading a full PyTorch stack, the service uses [`fastembed`](https://github.com/qdrant/fastembed), which serves the BGE model via **ONNX Runtime**. This eliminates ~2GB of CUDA dependencies, cuts cold-start time in half, and removes platform-specific DLL issues entirely. The model produces **384-dimensional L2-normalised vectors**, identical in semantics to the widely-benchmarked `all-MiniLM-L6-v2`.

### 2. Vector Index — FAISS `IndexFlatIP`
FAISS Inner Product search on normalised vectors is mathematically equivalent to cosine similarity, with the benefit of C++ speed and near-zero overhead on import. The index is built offline (in Google Colab) and loaded at startup — meaning the service never re-embeds the corpus.

### 3. Fuzzy Clustering — Gaussian Mixture Model (GMM)
A GMM trained on the embedding space gives every document (and every incoming query) a **probability distribution across 20 clusters**, not a hard label. This is critical for cache efficiency:

| Approach | Cache lookup as N grows |
|---|---|
| Linear scan over all entries | O(N) — degrades |
| Cluster-partitioned (this system) | O(N/k) — near-constant |

When a query arrives, only the cache entries whose dominant cluster overlaps with the query's top clusters are compared. As the cache fills up, lookup stays fast.

### 4. Tunable Similarity Threshold
The cache decision threshold (`0.75` by default, configurable in `app/config.py`) exposes a precise lever:

| Value | Behaviour |
|---|---|
| `0.95+` | Near-exact wording only |
| `0.85` | Minor rephrasing |
| `0.75` | ✅ Paraphrases with different vocabulary |
| `0.65` | Broad topic match |

---

## 📁 Project Structure

```
semantic-cache-search/
│
├── app/
│   ├── main.py          # FastAPI application, endpoint definitions
│   ├── embedder.py      # fastembed wrapper — embed_query(), embed_texts()
│   ├── cache.py         # SemanticCache — cluster-partitioned lookup + store
│   ├── clustering.py    # FuzzyCluster — GMM wrapper, predict_proba()
│   ├── vector_store.py  # VectorStore — FAISS index wrapper
│   └── config.py        # Central config (threshold, model, clusters, top-k)
│
├── data/
│   ├── loader.py        # 20 Newsgroups loader + text cleaning
│   ├── faiss_index.bin  # ⚡ Pre-built FAISS index (generated via Colab)
│   ├── gmm_model.pkl    # 🔮 Trained GMM model (generated via Colab)
│   └── corpus.pkl       # 📄 Cleaned document texts + labels
│
├── Implementation.md    # Step-by-step Google Colab training notebook
├── Dockerfile           # Two-stage production Docker build
├── docker-compose.yml   # One-command deployment with volume management
├── .dockerignore        # Keeps image lean
└── requirements.txt     # Minimal Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- **Docker** installed, **or** Python 3.11+ for running locally
- The three data files generated by the Colab notebook (see [Training Pipeline](#-training-pipeline))

### Option A — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/SakethramSathish/Semantic_Cache_Search.git
cd Semantic_Cache_Search

# Place your Colab-generated files in data/
# (see Training Pipeline section below)

# Build and launch
docker compose up --build
```

The API will be live at **http://localhost:8000**

---

### Option B — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

---

## 🧪 Training Pipeline (Google Colab)

The embedding and clustering models are trained on Colab (free GPU) to avoid local hardware requirements. `Implementation.md` in the repo contains **copy-paste-ready cells** that:

1. Load and clean the 20 Newsgroups corpus (18,000+ docs)
2. Generate 384-dim sentence embeddings via `sentence-transformers`
3. Build and save the FAISS index (`faiss_index.bin`)
4. Train the 20-component GMM and save it (`gmm_model.pkl`)
5. Save the document corpus (`corpus.pkl`)
6. One-click download all three files to your machine

Place the downloaded files in the `data/` directory and start the server.

---

## 📡 API Reference

### `POST /query`
Submit a natural language query. Returns a cache hit (instant) or performs a live FAISS search and populates the cache.

**Request:**
```json
{ "query": "What are the best graphics cards for gaming?" }
```

**Response (cache miss):**
```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "text": "I've been looking at the new RTX series...",
      "label": "comp.graphics",
      "similarity": 0.871
    }
  ],
  "dominant_cluster": 4
}
```

**Response (cache hit — rephrased query):**
```json
{
  "query": "Which GPU should I buy for PC gaming?",
  "cache_hit": true,
  "matched_query": "What are the best graphics cards for gaming?",
  "similarity_score": 0.783,
  "result": "...",
  "dominant_cluster": 4
}
```

---

### `GET /cache/stats`
Live cache performance metrics.

```json
{
  "total_entries": 12,
  "hit_count": 8,
  "miss_count": 4,
  "hit_rate": 0.667
}
```

---

### `DELETE /cache`
Flush the cache and reset all statistics.

```json
{ "message": "Cache cleared successfully." }
```

---

### Interactive Docs
Visit **http://localhost:8000/docs** for the full Swagger UI — test all endpoints directly in the browser.

---

## 🔬 Evaluation Scenarios

| Test | Query 1 | Query 2 | Expected |
|---|---|---|---|
| **Cache hit — synonym substitution** | "best graphics cards for gaming" | "GPU for PC gaming" | `cache_hit: true` |
| **Cache miss — different topic** | "What is relativity?" | — | `cache_hit: false` |
| **Cross-cluster boundary** | "how zero gravity affects the body" | — | Interesting cluster split |
| **Stats accumulation** | (run 5+ queries) | `GET /cache/stats` | Watch hit_rate climb |
| **Flush & verify** | `DELETE /cache` | `GET /cache/stats` | All counters reset |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| API Framework | FastAPI | Auto-generated docs, async-ready, Pydantic validation |
| Embeddings | fastembed (ONNX) | No PyTorch, no GPU drivers, ~10ms per query |
| Vector Search | FAISS IndexFlatIP | Exact cosine search, C++ speed, serialisable |
| Fuzzy Clustering | scikit-learn GMM | `predict_proba()` gives true soft assignments |
| Containerisation | Docker + Compose | Reproducible, one-command deploy |
| Training | Google Colab | Free T4 GPU for embedding 18K documents |

---

## 📄 License

MIT License — feel free to use, modify and distribute.

---

<div align="center">

Built with 🔍 semantic search, 🧠 fuzzy clustering, and ⚡ intelligent caching.

</div>
