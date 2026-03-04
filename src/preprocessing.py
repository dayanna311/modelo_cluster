import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FifaPreprocessor:

    def __init__(self):
        self.rename_map = {
            "Age": "age", "Overall Rating": "overall", "Potential": "potential",
            "Value": "value", "Wage": "wage", "Shooting": "shooting",
            "Passing": "passing", "Dribbling2": "dribbling",
            "Defense": "defending", "Physicality": "physicality",
            "Best Position": "best_position"
        }

        self.feature_cols = [
            "age","overall","potential","value","wage",
            "shooting","passing","dribbling","defending","physicality"
        ]

        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.cat_order = None

    def pos_group(self, p):

        p = str(p).upper().strip()

        if p == "GK": return "GK"
        if p in ["CB","LB","RB","LWB","RWB"]: return "DEF"
        if p in ["CDM","CM","CAM","LM","RM"]: return "MID"
        if p in ["LW","RW","CF","ST"]: return "FWD"

        return "OTHER"

    def preprocess(self, df):

        df = df.rename(columns=self.rename_map)
        df = df.fillna(0)

        df["target_str"] = df["best_position"].apply(self.pos_group)
        df = df[df["target_str"] != "OTHER"].copy().reset_index(drop=True)

        df["target"] = self.le.fit_transform(df["target_str"])
        self.cat_order = list(self.le.classes_)

        X = df[self.feature_cols].values
        y = df["target"].values

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, df

    def transform(self, df):
        """Usar en nuevos datos sin volver a entrenar"""
        df = df.rename(columns=self.rename_map)
        df = df.fillna(0)

        X = df[self.feature_cols].values
        return self.scaler.transform(X)

    def save(self, path="models\fifa_preprocessor.pkl"):
        
        joblib.dump(self, path)
        print(f"Preprocessor guardado en {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)