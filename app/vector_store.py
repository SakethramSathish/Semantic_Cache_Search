import faiss
import numpy as np


class VectorStore:

    def __init__(self, embeddings, documents):

        self.embeddings = np.array(embeddings).astype("float32")
        self.documents = documents

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(self.embeddings)

    def search(self, query_embedding, k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:
            results.append(self.documents[idx])

        return results