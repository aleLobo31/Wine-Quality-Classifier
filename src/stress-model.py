import pandas as pd
import joblib
from pathlib import Path

# Definimos variables globales de interés 
BASE_DIR = Path(__file__).resolve().parent.parent
STRESS_DATA_PATH = BASE_DIR / "data" / "processed" / "StressWineQT.csv"
DTREE_PATH = BASE_DIR / "models" / "dtree-wine-clf.pkl"
LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}


def main():
    # Importamos los datos de estrés
    df = pd.read_csv(STRESS_DATA_PATH)

    # Extraemos las variables explicativas y guardamos el valor a predecir
    y_real = df['quality_label']
    X = df.drop(columns=["quality_label"])

    # Cargamos el modelo
    dtree = joblib.load(DTREE_PATH)

    # Predecimos usando el modelo
    y_pred = dtree.predict(X)

    for idx, y in enumerate(y_pred):
        label_y = LABELS[y]
        print(f"Modelo: {label_y}, Realidad: {y_real[idx]}")
        

if __name__ == "__main__":
    main()