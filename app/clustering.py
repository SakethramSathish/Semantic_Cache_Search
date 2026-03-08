from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, embeddings, n_clusters=12):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="tied",
            random_state=42
        )

        self.model.fit(embeddings)

    def get_distribution(self, embedding):

        embedding = np.array([embedding])

        probs = self.model.predict_proba(embedding)[0]

        return probs

    def dominant_cluster(self, embedding):

        probs = self.get_distribution(embedding)

        return int(np.argmax(probs))