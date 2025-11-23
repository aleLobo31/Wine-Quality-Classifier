import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"

labels = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

def main():
    # Importamos el Dataset Limpio
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separamos las variables explicativas de la variable objetivo
    y = df['quality_label']
    X = df.drop(columns=["quality_label"])

    # Hacemos el encoding de la variable objetivo
    y_encoded = y.map(labels)
    print(y_encoded)

    # Separamos el conjunto de Test del conjunto de Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded # Uso stratify para que en ambos conjuntos haya un 3% de baja calidad
        ) 

    # Creamos el modelo basado en un Árbol de Decisión
    clf_tree = DecisionTreeClassifier(
        max_depth=4, # Para que las decisiones tengan sentido necesitamos que el árbol no pueda ser muy profundo
        class_weight='balanced', # Necesario para la clase minoritaria
        random_state=42
    )

    # Entrenamos el modelo con el conjunto de entrenamiento
    clf_tree.fit(X_train, y_train)

    # Visualizamos el árbol creado
    plt.figure(figsize=(24, 12))
    plot_tree(
        clf_tree,
        feature_names=X.columns,
        class_names=['Low', 'Medium', 'High'],
        filled=True, # Colorea los nodos según la clase dominante
        rounded=True,   
        fontsize=11
    )
    plt.title("Árbol de Decisión: Reglas de Calidad del Vino", fontsize=20)
    plt.show()

    # Predecimos con el conjunto de Test
    y_pred = clf_tree.predict(X_test)
                                    
    # Comprobamos la matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        display_labels=['Low', 'Medium', 'High'],
        cmap='Blues',
        colorbar=False,
        ax=ax
    )
    plt.title("Matriz de Confusión (Datos de Test)", fontsize=16)
    plt.show()

    # Vemos métricas f1-score y de recall para analizar el comportamiento del modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

    # Comprobamos si hay overfitting en los datos
    print("-" * 30)
    print("Chequeo Rápido de Overfitting:")
    print(f"Accuracy en Train: {clf_tree.score(X_train, y_train):.4f}")
    print(f"Accuracy en Test:  {clf_tree.score(X_test, y_test):.4f}")

    
if __name__ == "__main__":
    main()