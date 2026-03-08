# Google Colab Implementation Guide
## Semantic Cache Search — Training & Model Export

Run each cell in order inside a Google Colab notebook.  
After the final cell, **download the files from the Colab file browser** (the folder icon on the left) and place them in your local `data/` folder.

---

## Cell 1 — Install Dependencies

```python
# Colab already has sklearn and numpy.
# We install sentence-transformers (GPU-accelerated on Colab's T4)
# and faiss-gpu for fast vector indexing.
!pip install -q sentence-transformers faiss-gpu
```

**Why this works on Colab but not Windows locally:**  
Colab runs on Ubuntu with proper CUDA/OpenMP libraries pre-installed, so `sentence-transformers` (which needs PyTorch) loads without any DLL errors.

---

## Cell 2 — Imports

```python
import re
import json
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer
import faiss

print("All imports OK ✅")
```

---

## Cell 3 — Load & Clean the 20 Newsgroups Dataset

```python
def clean_text(text: str) -> str:
    """
    Data cleaning decisions (same logic as data/loader.py):
    1. Remove email addresses  → they are metadata, not semantic content.
    2. Remove quoted lines (>) → repeated text from previous emails dilutes the embedding.
    3. Collapse multiple newlines → normalises whitespace for the tokeniser.
    4. Minimum length filter   → posts < 50 chars have no useful semantic signal.
    """
    text = re.sub(r'\S+@\S+', '', text)          # strip email addresses
    text = re.sub(r'>.*\n', '', text)             # strip quoted lines
    text = re.sub(r'\n+', ' ', text)              # collapse newlines
    return text.strip()

# fetch_20newsgroups removes headers, footers, quotes at source level too.
# We use subset="all" (train + test) to maximise corpus size.
raw = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes")
)

documents = []
labels    = []

for doc, target in zip(raw.data, raw.target):
    cleaned = clean_text(doc)
    if len(cleaned) >= 50:           # discard near-empty posts
        documents.append(cleaned)
        labels.append(raw.target_names[target])

print(f"Kept {len(documents):,} documents out of {len(raw.data):,}")
print(f"Example:\n{documents[0][:300]}")
```

---

## Cell 4 — Generate Embeddings

> ⏱️ **This cell takes ~5–10 minutes on Colab GPU (T4). Do NOT re-run it unnecessarily.**

```python
# Model choice: all-MiniLM-L6-v2
#   - 384-dimensional output: expressive but compact (fast FAISS search)
#   - Trained specifically for semantic similarity tasks
#   - 80 MB model size: quick to download in Colab
model = SentenceTransformer("all-MiniLM-L6-v2")

# encode() handles batching internally; show_progress_bar gives ETA
embeddings = model.encode(
    documents,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalise to unit length so that Inner Product == Cosine Similarity
faiss.normalize_L2(embeddings)

print(f"Embeddings shape: {embeddings.shape}")   # (N, 384)
print(f"dtype: {embeddings.dtype}")              # float32
```

---

## Cell 5 — Build & Save the FAISS Index

```python
dim   = embeddings.shape[1]   # 384
index = faiss.IndexFlatIP(dim) # Inner Product on normalised vectors = cosine similarity
index.add(embeddings)

print(f"FAISS index contains {index.ntotal:,} vectors")

# Save index
faiss.write_index(index, "faiss_index.bin")
print("Saved → faiss_index.bin ✅")
```

---

## Cell 6 — Train the Fuzzy Clustering Model (GMM) & Save

```python
# Why GMM (Gaussian Mixture Model)?
#   Unlike K-Means, GMM computes predict_proba() — a probability
#   DISTRIBUTION over clusters for each document. This is "fuzzy" clustering:
#   a post about "electric cars" can be 60% sci.autos + 40% sci.electronics.
#
# Why 20 components?
#   The dataset has exactly 20 categories. Aligning the GMM components with
#   the latent topic structure gives the cache the best routing signal.
#   In an unlabelled scenario you would sweep BIC scores to choose this.
#
# covariance_type="spherical":
#   Full covariance matrices in 384-d space are computationally prohibitive
#   and prone to singularity. Spherical (one variance per component) is a
#   stable, fast approximation that still captures cluster uncertainty well.

print("Training GMM — this may take 3–5 minutes …")

gmm = GaussianMixture(
    n_components=20,
    covariance_type="spherical",
    max_iter=200,
    random_state=42,
    verbose=2
)
gmm.fit(embeddings)

print(f"\nGMM converged: {gmm.converged_}")
print(f"Lower-bound log-likelihood: {gmm.lower_bound_:.4f}")

# Save to pickle
with open("gmm_model.pkl", "wb") as f:
    pickle.dump(gmm, f)

print("Saved → gmm_model.pkl ✅")
```

---

## Cell 7 — Save Documents & Metadata

```python
# Save document texts and their original labels so the local app
# can reconstruct human-readable search results from FAISS indices.
payload = {
    "documents": documents,
    "labels":    labels,
}

with open("corpus.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"Saved → corpus.pkl  ({len(documents):,} docs) ✅")
```

---

## Cell 8 — Quick Sanity Check

```python
# Verify everything loads and operates correctly before downloading.
test_query = "What are the best graphics cards for gaming?"

q_emb = model.encode([test_query], convert_to_numpy=True)
faiss.normalize_L2(q_emb)

# FAISS search
D, I = index.search(q_emb, k=3)
print("=== Top 3 FAISS matches ===")
for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
    print(f"\nRank {rank} (cosine={score:.4f}):")
    print(f"  Label : {labels[idx]}")
    print(f"  Snippet: {documents[idx][:200]}")

# GMM fuzzy cluster distribution
probs = gmm.predict_proba(q_emb)[0]
top3  = np.argsort(probs)[::-1][:3]
print("\n=== Fuzzy Cluster Distribution (top 3) ===")
for c in top3:
    print(f"  Cluster {c:2d}: {probs[c]:.4f}")
```

---

## Cell 9 — Download Files from Colab

```python
from google.colab import files

for fname in ["faiss_index.bin", "gmm_model.pkl", "corpus.pkl"]:
    files.download(fname)
    print(f"Downloading {fname} …")
```

**After downloading**, move the three files into your local project's `data/` folder:

```
Semantic Cache Search/
└── data/
    ├── faiss_index.bin   ← FAISS vector index
    ├── gmm_model.pkl     ← Fuzzy GMM clustering model
    └── corpus.pkl        ← documents + labels list
```

---

## Local Usage — Loading the Models

Once the files are in `data/`, the local app can load them like this. The `app/embedder.py` uses `fastembed` (ONNX-based, no PyTorch) for real-time query encoding, and these pre-trained artefacts for search and clustering.

### `data/loader.py` — update to load pre-built artefacts

```python
# Run this once locally to verify everything loads correctly:
import pickle, faiss, numpy as np

with open("data/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

index = faiss.read_index("data/faiss_index.bin")

with open("data/gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

print(f"Corpus  : {len(corpus['documents']):,} documents")
print(f"FAISS   : {index.ntotal:,} vectors, dim={index.d}")
print(f"GMM     : {gmm.n_components} components, converged={gmm.converged_}")
print("All artefacts loaded locally ✅")
```

### Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

Then query the live API:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards for gaming?"}'
```

Expected response structure:

```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "...",
  "dominant_cluster": 7
}
```

Send the same query again and you will see `"cache_hit": true` with a `similarity_score` ≥ 0.85.

---

## Files Summary

| File | Size (approx) | Purpose |
|---|---|---|
| `faiss_index.bin` | ~220 MB | Fast vector similarity search index |
| `gmm_model.pkl` | ~2 MB | Fuzzy cluster probability model |
| `corpus.pkl` | ~15 MB | Raw document texts + category labels |
