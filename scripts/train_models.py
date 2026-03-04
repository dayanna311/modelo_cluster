import sys
from pathlib import Path

# Obtener la raíz del proyecto (dos niveles arriba de scripts/)
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Agregar src al sys.path
sys.path.insert(0, str(ROOT_DIR / "src"))

# Ahora podemos importar los archivos directamente sin usar "src."
import preprocessing
import clustering
import pandas as pd
import joblib

preprocessor = preprocessing.FifaPreprocessor()
cluster_pipeline = clustering.UMAP_DBSCAN_Clustering()


# -------------------
# paths
# -------------------

DATA_PATH = "data/FIFA_Complete_DATASET.xlsx" 
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)


# -------------------
# cargar dataset
# -------------------
print("Loading dataset...")

df = pd.read_excel(DATA_PATH)


# -------------------
# preprocessing
# -------------------
print("Running preprocessing...")


X_scaled, y, df_clean = preprocessor.preprocess(df)


# -------------------
# clustering
# -------------------
print("Training clustering models...")



cluster_pipeline.compute_umap(X_scaled, neighbors_list=[50])
cluster_pipeline.estimate_eps()
cluster_pipeline.run_dbscan()


# -------------------
# guardar modelos
# -------------------
print("Saving models...")

joblib.dump(preprocessor, MODEL_DIR / "fifa_preprocessor.pkl")
joblib.dump(cluster_pipeline, MODEL_DIR / "clustering_pipeline.pkl")

print("Models saved successfully!")