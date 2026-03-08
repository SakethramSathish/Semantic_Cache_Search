import json
import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import time

class SemanticCache:
    """
    Custom Semantic Cache Layer from First Principles.
    
    This cache stores query responses and retrieves them based on semantic similarity
    rather than exact string matching. It uses the GMM fuzzy clusters to optimize
    the search space.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.85):
        """
        Initializes the Semantic Cache.
        
        Tunable Decision Threshold (threshold):
        - This value reveals the system's tolerance for semantic variance.
        - A very high threshold (e.g., >0.95) acts almost like an exact match cache, prioritizing
          correctness and precision but suffering a low hit rate. It means the system believes
          even minor semantic differences require a fresh computation/lookup.
        - A lower threshold (e.g., 0.70) prioritizes speed and hit rate, mapping loosely related
          queries to the same cached result. It indicates the system assumes broad topics yield
          the same response, which is risky for precision but highly efficient.
        - 0.85 is a balanced default for cosine similarity on all-MiniLM-L6-v2.
        """
        self.encoder = SentenceTransformer(model_name)
        self.gmm = joblib.load('data/gmm_model.joblib')
        self.threshold = threshold
        
        # Cache memory structures
        # Structure:
        # cache_data[cluster_id] = [
        #    {"query": str, "embedding": np.array, "response": dict, "cluster_probs": np.array}
        # ]
        # We index cache entries primarily by their dominant cluster.
        self.cache_data = {i: [] for i in range(self.gmm.n_components)}
        
        # Stats
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0
        
    def _embed(self, text):
        emb = self.encoder.encode([text], show_progress_bar=False)
        emb = np.array(emb).astype('float32')
        faiss.normalize_L2(emb)
        return emb

    def _get_clusters(self, embedding):
        """Returns the cluster probabilities for the embedding."""
        probs = self.gmm.predict_proba(embedding)[0]
        return probs

    def search(self, query):
        """
        Searches the cache for a semantically similar query.
        
        Optimization via Fuzzy Clusters:
        Instead of comparing the incoming query's embedding against ALL cached queries (O(N)),
        we compute the query's fuzzy cluster distribution. We only compare the query against
        cached items that reside in the top `k` most probable clusters (where cumulative probability > 0.8).
        As the cache grows, this heavily restricts the search space, maintaining fast O(1)-like lookup
        efficiency relative to the total cache size.
        """
        emb = self._embed(query)
        probs = self._get_clusters(emb)
        
        # Determine top clusters to search (cumulative probability up to e.g., 0.9)
        # This focuses our search on the most relevant subspaces of the cache.
        sorted_clusters = np.argsort(probs)[::-1]
        
        target_clusters = []
        cumulative_prob = 0.0
        for c in sorted_clusters:
            target_clusters.append(c)
            cumulative_prob += probs[c]
            if cumulative_prob > 0.9:
                break
                
        best_match = None
        best_score = -1.0
        
        # Search only within the target clusters
        for cluster_id in target_clusters:
            for entry in self.cache_data[cluster_id]:
                # Compute inner product (cosine similarity since vectors are normalized)
                score = np.dot(emb[0], entry["embedding"][0])
                if score > best_score:
                    best_score = score
                    best_match = entry
                    
        # Decision threshold evaluation
        if best_score >= self.threshold:
            self.hit_count += 1
            return {
                "cache_hit": True,
                "matched_query": best_match["query"],
                "similarity_score": float(best_score),
                "result": best_match["response"],
                "dominant_cluster": int(sorted_clusters[0])
            }
            
        self.miss_count += 1
        return {
            "cache_hit": False,
            "query_embedding": emb,
            "cluster_probs": probs,
            "dominant_cluster": int(sorted_clusters[0])
        }

    def add(self, query, embedding, probs, dominant_cluster, response):
        """Adds a new entry to the cache under its dominant cluster."""
        entry = {
            "query": query,
            "embedding": embedding,
            "cluster_probs": probs,
            "response": response
        }
        self.cache_data[dominant_cluster].append(entry)
        self.total_entries += 1

    def get_stats(self):
        hit_rate = 0.0
        total_requests = self.hit_count + self.miss_count
        if total_requests > 0:
            hit_rate = self.hit_count / total_requests
            
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def flush(self):
        """Flushes the cache entirely and resets attributes."""
        self.cache_data = {i: [] for i in range(self.gmm.n_components)}
        self.total_entries = 0
        self.hit_count = 0
        self.miss_count = 0
