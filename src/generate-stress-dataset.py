import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "StressWineQT.csv"

def main():
    # Importamos el dataset limpio
    clean_df = pd.read_csv(PROCESSED_DATA_PATH)

    # Obtenemos los vinos de calidad alta originales
    high_wines = clean_df[clean_df['quality_label'] == "High"].drop(columns=["quality_label"])
    
    # Creo mi vino benchmark que voy a someter a pruebas (puedo usar mean() porque he eliminado las variables categóricas)
    benchmark_wine = high_wines.mean()
    # print(high_wines.describe())

    # Defino el espacio de pruebas extremas que voy a emplear (máxima sal de los mejores vinos hasta algo por encima de lo conocido en la muestra)
    chlorides_q = np.linspace(0.6, 0.12, 6)

    wine_list = []
    for q in chlorides_q:
        stress_wine = benchmark_wine.copy()
        stress_wine['chlorides'] = q
        stress_wine['quality_label'] = "Low"
        wine_list.append(stress_wine)

    # Casteamos la lista a un pandas dataframe
    df_stress = pd.DataFrame(wine_list)
    # print(df_stress.head())

    # Imprimimos el dataset de prueba sin índice
    df_stress.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()