import numpy as np
from sklearn.cluster import KMeans

class ClusteringService:
    def __init__(self, doc_embeddings):

        self.doc_embeddings = doc_embeddings
        self.kmeans = None
        self.doc_clusters = None

    def build_clusters(self, n_clusters=30):

        doc_ids = list(self.doc_embeddings.keys())
        X = np.array([self.doc_embeddings[doc_id] for doc_id in doc_ids])

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto"
        )

        labels = self.kmeans.fit_predict(X)

        self.doc_clusters = {
            doc_id: int(label)
            for doc_id, label in zip(doc_ids, labels)
        }

        return self.doc_clusters

    def predict_query_cluster(self, query_embedding):

        return int(self.kmeans.predict([query_embedding])[0])