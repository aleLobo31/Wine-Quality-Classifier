import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WineQT.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"

def main() -> None:
    # Importamos el dataset crudo
    df = pd.read_csv(RAW_DATA_PATH)

    # Limpiamos las variables redundantes que han exhibido correlaciones elevadas con otras
    redundant_vars = ["pH", "fixed acidity", "free sulfur dioxide", "Id"]
    df.drop(columns=redundant_vars, inplace=True) # Modifico el mismo df
    
    # Aplicamos logaritmos a las variables cuya kurtosis es superior a 3
    leptokurtic_vars = ["chlorides", "residual sugar", "sulphates", "total sulfur dioxide"]
    for var in leptokurtic_vars:
        df[var] = np.log1p(df[var])

    # Agrupamos en tres clases ["Low" (3, 4), "Medium" (5, 6), "High" (7, 8)]
    quality_labels = ["Low", "Medium", "High"]
    df['quality_label'] = pd.cut(df['quality'], bins=[2, 4, 6, 8], labels=quality_labels)
    df.drop(columns=["quality"], inplace=True)

    # Comprobamos el número de muestras que quedan dentro de cada clase objetivo
    conteo = df['quality_label'].value_counts()
    porcentaje = df['quality_label'].value_counts(normalize=True) * 100

    print("\n--- DISTRIBUCIÓN DE CLASES (TARGET) ---")
    print(conteo)
    print("\n--- PORCENTAJES ---")
    print(porcentaje)

    # Guardamos el dataset limpio
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Dataset limpio guardado en {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()