import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
    
    # Aplicamos RobustScaler para normalizar las características
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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
        verbose=2            # Mostrar progreso
    )

    # Entrenamos el modelo
    print(f"\n🌲 Probando {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinaciones...\n")
    searched_forest.fit(X_train, y_train)

    # Mostramos por consola el mejor bosque
    print("\n--- 🏆 EL BOSQUE GANADOR ---")
    print("Mejores Hiperparámetros encontrados:", searched_forest.best_params_)
    print(f"Mejor F1-Macro durante el entrenamiento: {searched_forest.best_score_:.4f}")

    return searched_forest

def evaluate_model(model: RandomForestClassifier, X_train: list, y_train:list, X_test: list, y_test: list) -> None:
    # Predecimos las categorías usando el mejor árbol
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred_test, target_names=['Low', 'Medium', 'High']))

    return None

def main():
    # Preparamos los datos
    X_train, X_test, y_train, y_test = create_training_testing_data(use_smote=False)

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