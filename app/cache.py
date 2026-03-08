import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):

        self.entries = []

        self.threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        if len(self.entries) == 0:
            self.miss_count += 1
            return None

        embeddings = [entry["embedding"] for entry in self.entries]

        sims = cosine_similarity([query_embedding], embeddings)[0]

        best_idx = np.argmax(sims)

        best_score = sims[best_idx]

        if best_score > self.threshold:

            self.hit_count += 1

            entry = self.entries[best_idx]

            return {
                "cache_hit": True,
                "matched_query": entry["query"],
                "similarity_score": float(best_score),
                "result": entry["result"],
                "dominant_cluster": entry["cluster"]
            }

        self.miss_count += 1

        return None

    def store(self, query, embedding, result, cluster):

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = len(self.entries)

        hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) else 0

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0