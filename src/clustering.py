import numpy as np
import umap
import joblib

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


class UMAP_DBSCAN_Clustering:

    def __init__(self, n_components=6, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

        self.umap_models = {}
        self.dbscan_models = {}
        self.embeddings = {}
        self.eps_values = {}
        self.results = {}

    def compute_umap(self, X, neighbors_list=[50]):

        for nn in neighbors_list:

            umap_model = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=nn
            )

            X_u = umap_model.fit_transform(X)
            key = f"UMAP_{nn}"

            self.umap_models[key] = umap_model
            self.embeddings[key] = X_u

            print(f"{key}: shape={X_u.shape}")

    def estimate_eps(self):

        for name, X_emb in self.embeddings.items():

            k = max(2 * X_emb.shape[1], 5)

            nbrs = NearestNeighbors(n_neighbors=k).fit(X_emb)
            dists, _ = nbrs.kneighbors(X_emb)

            k_dist = np.sort(dists[:, k - 1])
            eps_est = round(float(np.percentile(k_dist, 90)), 3)

            self.eps_values[name] = eps_est

    def run_dbscan(self):

        for name, X_emb in self.embeddings.items():

            k = max(2 * X_emb.shape[1], 5)
            eps_v = self.eps_values[name]

            dbscan_model = DBSCAN(eps=eps_v, min_samples=k)
            labels = dbscan_model.fit_predict(X_emb)

            self.dbscan_models[name] = dbscan_model

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

            self.results[name] = {
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": n_noise / len(labels) * 100,
                "eps": eps_v,
                "k": k
            }

            print(f"\n{name}")
            print(f" eps={eps_v}, min_samples={k}")
            print(f" Clusters detectados : {n_clusters}")
            print(f" Outliers            : {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    def save(self, path="models/umap_dbscan_pipeline.pkl"):        
        """Guardar todo el pipeline"""
        joblib.dump(self, path)
        print(f"Modelo guardado en: {path}")

    @staticmethod
    def load(path):
        """Cargar pipeline"""
        model = joblib.load(path)
        print(f"Modelo cargado desde: {path}")
        return model