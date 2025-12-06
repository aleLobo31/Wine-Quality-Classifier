import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc, make_scorer
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter
from itertools import cycle

# Definimos variables globales de interés 
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"
MODEL_PATH = BASE_DIR / "models"
LABELS = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
OPTIMIZE_WITH_AG = False

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
    
    # Aplicamos SMOTE para balancear las muestras de cada clase en el conjunto de entrenamiento
    if(use_smote):
        smote = SMOTE(random_state=42)
        print(f"\nAntes de SMOTE: {Counter(y_train)}")

        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"\nDespués de SMOTE: {Counter(y_train)}")

    return X_train, X_test, y_train, y_test

def create_model(random_state=42) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=random_state)

def train_model_GAsearch(model: DecisionTreeClassifier, X_train: list, y_train: list) -> GASearchCV:
    # Definimos el ADN del árbol
    tree_adn = {
        'max_depth': Integer(3, 6), # Profundidad del árbol
        'criterion': Categorical(['gini', 'entropy']), # Como se crean nuevas ramas
        'min_samples_split': Integer(20, 40), # Mínimo de muestras para dividir el árbol (evitar overfitting)
        'min_samples_leaf': Integer(10, 25) # Minimo de muestras por hoja (evitar overfitting)
    }

    # Creamos el experimento evolutivo
    evolved_tree = GASearchCV(
        estimator=model,
        cv=5,                        # Validación cruzada 
        scoring='f1_macro',          # F1 Macro es nuestra fitness function
        population_size=32,          # Árboles compitiendo en cada generación
        generations=16,              # Rondas de evolución
        tournament_size=3,           # Árboles se pelean para reproducirse
        elitism=True,                # El mejor árbol siempre sobrevive intacto
        crossover_probability=0.8,   # Probabilidad de mezclar padres
        mutation_probability=0.15,   # Probabilidad de mutación aleatoria
        param_grid=tree_adn,         # 'Genoma' de nuestro árbol
        criteria='max',              # Queremos MAXIMIZAR el F1-Score
        n_jobs=-1,                   # Paralelizar entrenamiento
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

def train_model_gridsearch(model: DecisionTreeClassifier, X_train: list, y_train: list) -> GridSearchCV:
    # Definimos el espacio de búsqueda
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [10, 15, 20],
        'min_samples_leaf': [5, 10, 15],
        'class_weight': ['balanced', {0:10, 1:1, 2:5}]
    }

    # Instanciamos el algoritmo de búsqueda
    searched_tree = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3, # Cross-validation folds
        scoring='f1_macro', # Valor objetivo f1_macro make_scorer(f1_low_scorer)
        n_jobs=-1, # Opción para parelelizar
        verbose=2 # Mostrar progreso
    )

    # Entrenamos el modelo
    searched_tree.fit(X_train, y_train)

    # Mostramos por consola el mejor árbol
    print("\n--- EL ÁRBOL GANADOR ---")
    print("Mejores Genes encontrados:", searched_tree.best_estimator_)
    print(f"Mejor F1-Macro durante el entrenamiento: {searched_tree.best_score_:.4f}")

    # Dibujamos el árbol
    plot_best_tree(searched_tree, X_train)

    return searched_tree

def evaluate_model(model: DecisionTreeClassifier, X_train: list, y_train:list, X_test: list, y_test: list) -> None:
    # Predecimos las categorías usando el mejor árbol
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred_test, target_names=['Low', 'Medium', 'High']))

    # Dibujamos la curva ROC
    plot_roc_curve(model, X_test, y_test)

    return None

def plot_best_tree(best_tree: GASearchCV, X_train: list) -> None:
    print("\nGenerando imagen del árbol... (Puede tardar unos segundos si es muy grande)")
    
    # Calculamos dinámicamente el tamaño de la imagen en base a la profundidad del árbol
    tree_depth = best_tree.best_estimator_.get_depth()
    fig_width = max(40, tree_depth * 16)
    fig_height = max(20, tree_depth * 16)
    font_size = max(8, 12 - tree_depth)

    # Creamos la imagen con las dimensiones indicadas
    plt.figure(figsize=(fig_width, fig_height)) 

    # Dibujamos el árbol
    plot_tree(
        best_tree.best_estimator_,
        feature_names=X_train.columns,         # Pone nombres químicos en vez de "X[0]"
        class_names=['Low', 'Medium', 'High'], # Pone nombres de clases en vez de "0, 1, 2"
        filled=True,                           # Colorea las cajas según la clase dominante
        rounded=True,                          # Bordes suaves (estética)
        fontsize=font_size,                    # Tamaño de letra (ajusta si se ve pequeño)
        proportion=True,                       # El tamaño de la caja indica cuántos vinos caen ahí
        precision=2                            # Decimales en los umbrales
    )
    plt.title(f"\nEl Árbol Evolucionado (F1-Macro: {best_tree.best_score_:.2f})", fontsize=20)
    plt.show()
    return None

def plot_roc_curve(model: DecisionTreeClassifier, X_test: list, y_test: list) -> None:
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
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculamos la curva ROC micro-average (todas las clases juntas)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Dibujamos las curvas
    plt.figure(figsize=(10, 8))
    
    # Colores para cada clase
    colors = cycle(['#FF6B6B', '#4ECDC4', '#45B7D1'])
    class_names = ['Low', 'Medium', 'High']
    
    # Dibujamos la curva para cada clase
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{name} (AUC = {roc_auc[i]:.3f})')
    
    # Dibujamos la curva micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='navy', lw=2, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
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

def f1_low_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0], average='macro')
def main():
    # Preparamos los datos
    X_train, X_test, y_train, y_test = create_training_testing_data(use_smote=False)

    # Creamos el modelo basado en un Árbol de Decisión
    clf_tree = create_model()

    # Entrenamos el modelo
    if OPTIMIZE_WITH_AG:
        best_tree = train_model_GAsearch(clf_tree, X_train, y_train)
    else:
        best_tree = train_model_gridsearch(clf_tree, X_train, y_train)
    
    # Evaluamos el modelo
    evaluate_model(best_tree.best_estimator_, X_train, y_train, X_test, y_test)

    # Guardamos el modelo entrenado
    model_path = MODEL_PATH / "dtree-wine-clf.pkl"
    joblib.dump(best_tree.best_estimator_, model_path)

if __name__ == "__main__":
    main()