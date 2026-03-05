from fastapi import FastAPI
import joblib
import numpy as np
import sys
from pydantic import BaseModel
from pathlib import Path


# ROOT_DIR apunta a la raíz del proyecto (modelo_cluster)
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Agregar src al sys.path para poder importar tus scripts
sys.path.insert(0, str(ROOT_DIR / "src"))


import preprocessing
import clustering

app = FastAPI(title="FIFA Clustering API")


# -------------------------
# cargar modelos al iniciar
# -------------------------
@app.on_event("startup")
def load_models():

    global preprocessor
    global clustering_model

    # Cargar modelos

    MODEL_DIR = ROOT_DIR / "models"
    preprocessor = joblib.load(MODEL_DIR / "fifa_preprocessor.pkl")
    clustering_model = joblib.load(MODEL_DIR / "clustering_pipeline.pkl")

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