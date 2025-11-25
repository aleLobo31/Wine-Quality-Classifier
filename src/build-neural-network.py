import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from scikeras.wrappers import KerasClassifier

# Defino variables globales de interés
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"
MODEL_PATH = BASE_DIR / "models"
LABELS = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

def create_training_testing_data(use_smote=True) -> list:
    # Importamos los datos limpios
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Separamos las variables explicativas de la variable objetivo
    y = df['quality_label']
    X = df.drop(columns=['quality_label'])

    # Realizamos el encoding de la variable objetivo
    y_encoded = y.map(LABELS)

    # Creamos un conjunto de entrenamiento y otro de validación
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y_encoded, 
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    # Intancio Robust Scaler
    rscaler = RobustScaler()

    # Aplicamos un escalado para hacer más robusto el aprendizaje de la NN
    X_train_scaled = rscaler.fit_transform(X_train)
    X_test_scaled = rscaler.transform(X_test)

    # Aplicamos SMOTE para balancear las muestras de cada clase en el conjunto de entrenamiento
    if(use_smote):
        smote = SMOTE(random_state=42)
        print(f"Antes de SMOTE: {Counter(y_train)}")

        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"Después de SMOTE: {Counter(y_train)}")

    return X_train_scaled, X_test_scaled, y_train, y_test

def create_model(n_neurons: int, dropout_rate: float, learning_rate: float, activation: str) -> tf.keras.models.Sequential:
    # Creamos el modelo
    clf_nn = tf.keras.models.Sequential()

    # Definimos la arquitectura
    clf_nn.add(tf.keras.layers.Input( # Definimos la capa de entrada
        shape=(8,
        )))
    
    clf_nn.add(tf.keras.layers.Dense( # Definimos las capas ocultas
        units=n_neurons, 
        activation=activation
        ))
    clf_nn.add(tf.keras.layers.Dropout(dropout_rate)) # Añadimos un dropout para que en cada iteración de entrenamiento se apaguen algunas neuronas y así evitar overfitting
    clf_nn.add(tf.keras.layers.Dense(
        units=n_neurons // 2, 
        activation=activation
        ))
    clf_nn.add(tf.keras.layers.Dense( # Definimos la capa de salida
        units=3, 
        activation='softmax' # Con softmax transformamos las salidas en probabilidades
        )) 
    
    # Definimos cómo va a aprender esta NN
    clf_nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return clf_nn

def train_model(X_train: list, y_train: list) -> RandomizedSearchCV:
    # Creamos un wrapper de la Red Neuronal para pode rusar el Engine del AG
    clf_keras = KerasClassifier(
        model=create_model,
        epochs=30,
        verbose=0,
    )

    # Creamos un callback para detener la red cuando hay overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Definimos los genes que queremos optimizar
    nn_adn = {
        'model__n_neurons': randint(16, 128),
        'model__activation': ['relu', 'tanh', 'selu'],
        'model__dropout_rate': uniform(0.1, 0.4),
        'model__learning_rate': loguniform(0.0001, 0.01),
        'batch_size': [16, 32, 64]
    }

    # Instanciamos el Algoritmo de Búsqueda Genético
    evolved_nn = RandomizedSearchCV(
        estimator=clf_keras,
        param_distributions=nn_adn,
        cv=3,
        scoring='f1_macro',         
        n_iter=20,                   # Número de individuos a evaluar
        n_jobs=1,                    # Con Keras/Tensorflow, mejor n_jobs=1 para no liar a la GPU/CPU
        verbose=2,
        random_state=42
    )

    # Entrenamos el modelo
    evolved_nn.fit(
        X_train, 
        y_train,
        callbacks=[callback]
    )

    # Imprimimos por consola el mejor individuo con su fitness function
    print("\n--- MEJOR ARQUITECTURA ENCONTRADA ---")
    print(evolved_nn.best_params_)
    print(f"Mejor F1-Macro: {evolved_nn.best_score_:.4f}")

    return evolved_nn

def evaluate_model(model: RandomizedSearchCV, X_train: list, y_train: list, X_test: list, y_test: list) -> None:
    # Predecimos usando el modelo
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

def main():
    # Creamos el conjunto de entrenamiento y de test.
    X_train, X_test, y_train, y_test = create_training_testing_data()

    # Entrenamos el modelo
    clf_nn = train_model(X_train, y_train)

    # Evaluamos el modelo
    evaluate_model(clf_nn, X_train, y_train, X_test, y_test)

    #TODO: Guardar el modelo

if __name__ == "__main__":
    main()