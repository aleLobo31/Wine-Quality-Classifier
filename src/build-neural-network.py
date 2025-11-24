import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Defino variables globales de interés
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "CleanWineQT.csv"
MODEL_PATH = BASE_DIR / "models"
LABELS = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

def create_training_testing_data() -> list:
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
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Intancio Robust Scaler
    rscaler = RobustScaler()

    # Aplicamos un escalado para hacer más robusto el aprendizaje de la NN
    X_train_scaled = rscaler.fit_transform(X_train)
    X_test_scaled = rscaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def create_model(n_inputs: int) -> tf.keras.models.Sequential:
    # Creamos el modelo
    clf_nn = tf.keras.models.Sequential()

    # Definimos la arquitectura
    clf_nn.add(tf.keras.layers.Input( # Definimos la capa de entrada
        shape=(n_inputs,
        )))
    
    clf_nn.add(tf.keras.layers.Dense( # Definimos las capas ocultas
        units=32, 
        activation='relu'
        ))
    clf_nn.add(tf.keras.layers.Dropout(0.2)) # Añadimos un dropout para que en cada iteración de entrenamiento se apaguen algunas neuronas y así evitar overfitting
    clf_nn.add(tf.keras.layers.Dense(
        units=32, 
        activation='relu'
        ))
    clf_nn.add(tf.keras.layers.Dense( # Definimos la capa de salida
        units=3, 
        activation='softmax' # Con softmax transformamos las salidas en probabilidades
        )) 
    
    # Definimos cómo va a aprender esta NN
    clf_nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return clf_nn

def train_model(model: tf.keras.models.Sequential, X_train: list, y_train: list) -> None:
    # Calculamos el valor de cada clase
    labels_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Transformamos los pesos calculados en un diccionario para pasarlo como entrada en fit
    labels_weights_dict = dict(enumerate(labels_weights))

    # Creamos un callback para detener la red cuando hay overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entrenamos el modelo
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        class_weight=labels_weights_dict,
        callbacks=[callback]
    )

    return None

def evaluate_model(model: tf.keras.models.Sequential, X_train: list, y_train: list, X_test: list, y_test: list) -> None:
    # Predecimos usando el modelo
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_train = model.predict(X_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    # Imprimimos las métricas asociadas a la Matriz de Confusión para evaluar el modelo
    print("\n--- REPORTE DE CLASIFICACIÓN (TRAIN) ---")
    print(classification_report(y_train, y_pred_train, target_names=['Low', 'Medium', 'High']))
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST) ---")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

def main():
    # Creamos el conjunto de entrenamiento y de test.
    X_train, X_test, y_train, y_test = create_training_testing_data()

    # Creamos el modelo
    clf_nn = create_model(X_train.shape[1])

    # Entrenamos el modelo
    train_model(clf_nn, X_train, y_train)

    # Evaluamos el modelo
    evaluate_model(clf_nn, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()