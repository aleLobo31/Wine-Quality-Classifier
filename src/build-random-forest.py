import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter

# Definimos variables globales de interés 
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"
MODEL_PATH = BASE_DIR / "models"
LABELS = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

def create_training_testing_data(use_smote: bool=True) -> list:
    # Importamos el Dataset Limpio
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separamos las variables explicativas de la variable objetivo
    y = df['quality_label']
    X = df.drop(columns=["quality_label"])

    # Hacemos el encoding de la variable objetivo
    y_encoded = y.map(LABELS)

    # Separamos el conjunto de Test del conjunto de Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_encoded,
        test_size=0.4,
        random_state=42,
        stratify=y_encoded # Uso stratify para que en ambos conjuntos haya un 3% de baja calidad
        )
        
    # Convertimos de vuelta a DataFrame para mantener nombres de columnas
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    # Aplicamos SMOTE para balancear las muestras de cada clase en el conjunto de entrenamiento
    if(use_smote):
        smote = SMOTE(random_state=42)
        print(f"\nAntes de SMOTE: {Counter(y_train)}")

        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"\nDespués de SMOTE: {Counter(y_train)}")

    return X_train, X_test, y_train, y_test

def create_model(random_state=42) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, n_jobs=-1)

def train_model_gridsearch(model: RandomForestClassifier, X_train: list, y_train: list) -> GridSearchCV:
    # Definimos el espacio de búsqueda para Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150, 200],      # Número de árboles en el bosque
        'max_depth': [4, 8, 16],                 # Profundidad máxima de cada árbol
        'min_samples_split': [10, 15, 20],        # Mínimo de muestras para dividir
        'min_samples_leaf': [5, 10, 15],          # Mínimo de muestras por hoja
        'max_features': ['sqrt', 'log2'],         # Número de features a considerar en cada split
        'class_weight': ['balanced', {0:10, 1:1, 2:5}], # Manejo de clases desbalanceadas
        'bootstrap': [True]                       # Usar bootstrap sampling
    }

    # Instanciamos el algoritmo de búsqueda
    searched_forest = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                # Cross-validation folds
        scoring='f1_macro',  # Valor objetivo
        n_jobs=-1,           # Opción para paralelizar
        verbose=0            # Mostrar progreso
    )

    # Entrenamos el modelo
    print(f"\n🌲 Probando {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinaciones...\n")
    searched_forest.fit(X_train, y_train)

    # Mostramos por consola el mejor bosque
    print("\n--- 🏆 EL BOSQUE GANADOR ---")
    print("Mejores Hiperparámetros encontrados:", searched_forest.best_params_)
    print(f"Mejor F1-Macro durante el entrenamiento: {searched_forest.best_score_:.4f}")

    return searched_forest

def plot_confusion_matrix(y_test: list, y_pred: list) -> None:
    """
    Dibuja la matriz de confusión para el conjunto de test
    """
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    
    # Calculamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Creamos el display de la matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=['Low', 'Medium', 'High'])
    
    # Dibujamos la matriz con un colormap personalizado
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    
    # Configuramos el título y etiquetas
    plt.title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.ylabel('Clase Real', fontsize=12)
    plt.tight_layout()
    
    plt.show()
    
    # Imprimimos información adicional
    print("\nMatriz de confusión:")
    print(cm)
    print("\nInterpretación:")
    print("- Diagonal principal: predicciones correctas")
    print("- Fuera de la diagonal: errores de clasificación")
    
    return None

def evaluate_model(model: RandomForestClassifier, X_train: list, y_train:list, X_test: list, y_test: list) -> None:
    # Predecimos las categorías usando el mejor árbol
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred_test, target_names=['Low', 'Medium', 'High']))
    
    # Imprimimos F1Score overall
    microF1Score = f1_score(y_test, y_pred_test, average='micro')
    print('F1 Micro Score is : ', microF1Score)

    # Ploteamos la matriz de confusión
    plot_confusion_matrix(y_test, y_pred_test)

    # Ploteamos la curva ROC
    plot_roc_curve(model, X_test, y_test)

    return None

def plot_roc_curve(model: RandomForestClassifier, X_test: list, y_test: list) -> None:
    """
    Dibuja la curva ROC para clasificación multiclase (One-vs-Rest)
    """
    print("\n--- CURVA ROC ---")
    
    # Binarizamos las etiquetas para One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Obtenemos las probabilidades de predicción
    y_score = model.predict_proba(X_test)
    
    # Calculamos ROC y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds_dict = dict()
    best_threshold = dict()
    youden_j = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds_dict[i] = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Calculamos el índice de Youden J para cada punto
        j_scores = tpr[i] - fpr[i]
        
        # Encontramos el índice del mejor umbral (máximo Youden J)
        best_idx = j_scores.argmax()
        best_threshold[i] = thresholds_dict[i][best_idx]
        youden_j[i] = j_scores[best_idx]
    
    # Calculamos la curva ROC micro-average (todas las clases juntas)
    fpr["micro"], tpr["micro"], thresholds_micro = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calculamos el mejor umbral para micro-average
    j_scores_micro = tpr["micro"] - fpr["micro"]
    best_idx_micro = j_scores_micro.argmax()
    best_threshold["micro"] = thresholds_micro[best_idx_micro]
    youden_j["micro"] = j_scores_micro[best_idx_micro]
    
    # Imprimimos los mejores umbrales por consola
    print("\n--- MEJORES UMBRALES DE DECISIÓN (Youden J Index) ---")
    class_names = ['Low', 'Medium', 'High']
    for i, name in enumerate(class_names):
        print(f"{name:8} → Umbral óptimo = {best_threshold[i]:.4f}, Youden J = {youden_j[i]:.4f}")
    print(f"{'Micro-avg':8} → Umbral óptimo = {best_threshold['micro']:.4f}, Youden J = {youden_j['micro']:.4f}")
    
    # Dibujamos las curvas
    plt.figure(figsize=(10, 8))
    
    # Colores para cada clase
    colors = cycle(['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Dibujamos la curva para cada clase
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{name} (AUC = {roc_auc[i]:.3f})')
        
        # Marcamos el punto óptimo (mejor umbral según Youden J)
        best_idx = (tpr[i] - fpr[i]).argmax()
        plt.scatter(fpr[i][best_idx], tpr[i][best_idx], color=color, s=150, 
                   marker='o', edgecolors='black', linewidths=2, zorder=5,
                   label=f'{name} Umbral óptimo ({best_threshold[i]:.3f})')
    
    # Dibujamos la curva micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='navy', lw=2, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Marcamos el punto óptimo para micro-average
    best_idx_micro = (tpr["micro"] - fpr["micro"]).argmax()
    plt.scatter(fpr["micro"][best_idx_micro], tpr["micro"][best_idx_micro], 
               color='navy', s=150, marker='D', edgecolors='black', linewidths=2, zorder=5,
               label=f'Micro-avg Umbral óptimo ({best_threshold["micro"]:.3f})')
    
    # Dibujamos la línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Clasificador Aleatorio (AUC = 0.50)')
    
    # Configuramos el gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curva ROC - Clasificación de Calidad de Vino', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
        
    plt.show()
    
    # Imprimimos los resultados
    print("\n--- AUC SCORES ---")
    for i, name in enumerate(class_names):
        print(f"{name:8} → AUC = {roc_auc[i]:.4f}")
    print(f"{'Micro-avg':8} → AUC = {roc_auc['micro']:.4f}")

    return None

def main():
    # Preparamos los datos
    X_train, X_test, y_train, y_test = create_training_testing_data(use_smote=True)

    # Creamos el modelo basado en Random Forest
    clf_forest = create_model()

    # Entrenamos el modelo usando Grid Search
    best_forest = train_model_gridsearch(clf_forest, X_train, y_train)
    
    # Evaluamos el modelo
    evaluate_model(best_forest.best_estimator_, X_train, y_train, X_test, y_test)

    # Guardamos el modelo entrenado
    model_path = MODEL_PATH / "random-forest-wine-clf.pkl"
    joblib.dump(best_forest.best_estimator_, model_path)
    print(f"\n✓ Modelo guardado en: {model_path}")

if __name__ == "__main__":
    main()