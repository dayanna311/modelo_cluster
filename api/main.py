from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

from src.preprocessing import FifaPreprocessor
from src.clustering import UMAP_DBSCAN_Clustering


app = FastAPI(title="FIFA Clustering API")


# -------------------------
# cargar modelos al iniciar
# -------------------------
@app.on_event("startup")
def load_models():

    global preprocessor
    global clustering_model

    preprocessor = joblib.load("models/fifa_preprocessor.pkl")
    clustering_model = joblib.load("models/clustering_pipeline.pkl")

    print("Modelos cargados")


# -------------------------
# schema entrada
# -------------------------
class PlayerInput(BaseModel):

    age: float
    overall: float
    potential: float
    value: float
    wage: float
    shooting: float
    passing: float
    dribbling: float
    defending: float
    physicality: float


# -------------------------
# endpoint prueba
# -------------------------
@app.get("/")
def root():
    return {"message": "FIFA clustering API running"}


# -------------------------
# endpoint clustering
# -------------------------
@app.post("/cluster")
def cluster_player(player: PlayerInput):

    data = np.array([[

        player.age,
        player.overall,
        player.potential,
        player.value,
        player.wage,
        player.shooting,
        player.passing,
        player.dribbling,
        player.defending,
        player.physicality

    ]])

    X_scaled = preprocessor.scaler.transform(data)

    umap_model = list(clustering_model.umap_models.values())[0]
    embedding = umap_model.transform(X_scaled)

    dbscan_model = list(clustering_model.dbscan_models.values())[0]
    label = dbscan_model.fit_predict(embedding)

    return {"cluster": int(label[0])}