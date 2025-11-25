import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
        test_size=0.3,
        random_state=42,
        stratify=y_encoded # Uso stratify para que en ambos conjuntos haya un 3% de baja calidad
        )
    
    # Aplicamos SMOTE para balancear las muestras de cada clase en el conjunto de entrenamiento
    if(use_smote):
        smote = SMOTE(random_state=42)
        print(f"Antes de SMOTE: {Counter(y_train)}")

        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Después de SMOTE: {Counter(y_train)}")

    return X_train, X_test, y_train, y_test

def create_model(random_state=42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=random_state)

def train_model(model: DecisionTreeClassifier, X_train: list, y_train: list) -> GASearchCV:
    # Definimos el ADN del árbol
    tree_adn = {
        'max_depth': Integer(2, 10), # Profundidad del árbol
        'criterion': Categorical(['gini', 'entropy']), # Como se crean nuevas ramas
        'min_samples_split': Integer(10, 20), # Mínimo de muestras para dividir el árbol (evitar overfitting)
        'min_samples_leaf': Integer(5, 10), # Minimo de muestras por hoja (evitar overfitting)
        'class_weight': Categorical(['balanced', None]) # Usar pesos balanceados o no
    }

    # Creamos el experimento evolutivo
    evolved_tree = GASearchCV(
        estimator=model,
        cv=3,                        # Validación cruzada (3 exámenes por individuo)
        scoring='f1_macro',          # F1 Macro es nuestra fitness function
        population_size=32,          # 15 Árboles compitiendo en cada generación
        generations=10,              # 10 Rondas de evolución
        tournament_size=4,           # 3 Árboles se pelean para reproducirse
        elitism=True,                # El mejor árbol siempre sobrevive intacto
        crossover_probability=0.8,   # 80% de probabilidad de mezclar padres
        mutation_probability=0.1,    # 10% de probabilidad de mutación aleatoria
        param_grid=tree_adn,         # 'Genoma' de nuestro árbol
        criteria='max',              # Queremos MAXIMIZAR el F1-Score
        n_jobs=-1,                   # Usa todos los núcleos de la CPU
        verbose=True
    )

    # Entrenamos el modelo con el conjunto de entrenamiento
    evolved_tree.fit(X_train, y_train)

    # Mostramos por consola los resultados del proceso de evolución
    print("\n--- EL ÁRBOL GANADOR ---")
    print("Mejores Genes encontrados:", evolved_tree.best_params_)
    print(f"Mejor F1-Macro durante el entrenamiento: {evolved_tree.best_score_:.4f}")

    # Dibujamos el árbol
    plot_best_tree(evolved_tree, X_train)

    return evolved_tree

def evaluate_model(model: DecisionTreeClassifier, X_train: list, y_train:list, X_test: list, y_test: list) -> None:
    # Predecimos las categorías usando el mejor árbol
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred_test, target_names=['Low', 'Medium', 'High']))

    return None

def plot_best_tree(best_tree: GASearchCV, X_train: list) -> None:
    print("Generando imagen del árbol... (Puede tardar unos segundos si es muy grande)")
    plt.figure(figsize=(35, 15)) 
    plot_tree(
        best_tree.best_estimator_,
        feature_names=X_train.columns,      # Pone nombres químicos en vez de "X[0]"
        class_names=['Low', 'Medium', 'High'], # Pone nombres de clases en vez de "0, 1, 2"
        filled=True,                        # Colorea las cajas según la clase dominante
        rounded=True,                       # Bordes suaves (estética)
        fontsize=10,                        # Tamaño de letra (ajusta si se ve pequeño)
        proportion=True,                    # El tamaño de la caja indica cuántos vinos caen ahí
        precision=2                         # Decimales en los umbrales
    )
    plt.title(f"El Árbol Evolucionado (F1-Macro: {best_tree.best_score_:.2f})", fontsize=20)
    plt.show()

def main():
    # Preparamos los datos
    X_train, X_test, y_train, y_test = create_training_testing_data(use_smote=True)

    # Creamos el modelo basado en un Árbol de Decisión
    clf_tree = create_model()

    # Entrenamos el modelo
    evolved_tree = train_model(clf_tree, X_train, y_train)
    
    # Evaluamos el modelo
    evaluate_model(evolved_tree.best_estimator_, X_train, y_train, X_test, y_test)

    #TODO: Guardar el modelo
    
if __name__ == "__main__":
    main()