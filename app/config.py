# app/config.py

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of clusters for fuzzy clustering
NUM_CLUSTERS = 12

# Cache similarity threshold
# ─────────────────────────────────────────────────────────────────────────────
# This value controls how similar a new query must be to a cached query
# (cosine similarity, 0–1) before the cache is considered a "hit".
#
#  0.95+  → near-exact wording match only (very strict, low hit rate)
#  0.85   → same sentence, minor word changes
#  0.75   ← DEFAULT: catches paraphrases with different vocabulary
#            e.g. "graphics cards for gaming" ≈ "GPU for PC gaming"
#  0.65   → broad topic match (lenient, higher hit rate but less precise)
#
# Tune this up if you're getting false positives (unrelated queries matched),
# or down if you're getting too many cache misses for obvious paraphrases.
CACHE_SIMILARITY_THRESHOLD = 0.75

# Number of documents returned from vector search
TOP_K_RESULTS = 3