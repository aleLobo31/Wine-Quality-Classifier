"""
Script para realizar un t-test sobre los f1-score macro de modelos entrenados con GA Search.

Null Hypothesis (H0): La media de los f1-score macro es <= 0.52
Alternative Hypothesis (H1): La media de los f1-score macro es > 0.52

Este script ejecuta múltiples veces el entrenamiento del modelo de árbol de decisión
con GA Search (sin SMOTE y sin random_state) para obtener una muestra de f1-scores.
Luego realiza un t-test unilateral (one-tailed) para evaluar si la media es 
significativamente mayor que 0.52.
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

# Importamos el módulo de construcción del árbol
sys.path.append(str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("build_decision_tree", Path(__file__).parent / "build-decision-tree.py")
bdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdt)

# Variables de configuración
NUM_EXPERIMENTS = 30  # Número de experimentos a realizar
NULL_HYPOTHESIS_MEAN = 0.52  # Media de la hipótesis nula
ALPHA = 0.05  # Nivel de significancia (95% de confianza)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "reports"


def run_single_experiment(experiment_num: int) -> float:
    """
    Ejecuta un experimento completo: prepara datos, entrena modelo y evalúa.
    
    Args:
        experiment_num: Número del experimento para logging
        
    Returns:
        f1_macro: F1-score macro del modelo en el conjunto de test
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO {experiment_num + 1}/{NUM_EXPERIMENTS}")
    print(f"{'='*60}")
    
    # Preparamos los datos (sin SMOTE, con stratify pero sin random_state)
    X_train, X_test, y_train, y_test = bdt.create_training_testing_data(use_smote=False)
    
    # Creamos el modelo (sin random_state)
    clf_tree = bdt.create_model(random_state=None)
    
    # Entrenamos con GA Search
    best_tree = bdt.train_model_GAsearch(clf_tree, X_train, y_train)
    
    # Evaluamos y obtenemos el f1_macro
    f1_macro = bdt.evaluate_model(best_tree.best_estimator_, X_train, y_train, X_test, y_test)
    
    print(f"\n✓ F1-Macro obtenido: {f1_macro:.4f}")
    
    return f1_macro


def perform_ttest(f1_scores: list) -> dict:
    """
    Realiza un t-test unilateral para evaluar si la media de f1_scores > 0.52
    
    Args:
        f1_scores: Lista de f1-scores macro obtenidos en los experimentos
        
    Returns:
        results: Diccionario con los resultados del t-test
    """
    # Convertimos a numpy array
    scores = np.array(f1_scores)
    
    # Calculamos estadísticas descriptivas
    mean_score = np.mean(scores)
    std_score = np.std(scores, ddof=1)  # Desviación estándar muestral
    se_score = std_score / np.sqrt(len(scores))  # Error estándar
    
    # Realizamos el t-test unilateral (alternative='greater')
    # H0: mean <= 0.52
    # H1: mean > 0.52
    t_statistic, p_value_two_tailed = stats.ttest_1samp(scores, NULL_HYPOTHESIS_MEAN)
    
    # Para un test unilateral (greater), dividimos el p-value por 2
    # Solo si el t_statistic es positivo (media muestral > media nula)
    if t_statistic > 0:
        p_value = p_value_two_tailed / 2
    else:
        p_value = 1 - (p_value_two_tailed / 2)
    
    # Calculamos intervalo de confianza (95%)
    confidence_level = 1 - ALPHA
    df = len(scores) - 1  # Grados de libertad
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
    margin_error = t_critical * se_score
    ci_lower = mean_score - margin_error
    ci_upper = mean_score + margin_error
    
    # Determinamos si rechazamos H0
    reject_null = p_value < ALPHA
    
    results = {
        'n_experiments': len(scores),
        'f1_scores': scores.tolist(),
        'mean': mean_score,
        'std': std_score,
        'se': se_score,
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
        't_statistic': t_statistic,
        'p_value': p_value,
        'alpha': ALPHA,
        'null_hypothesis_mean': NULL_HYPOTHESIS_MEAN,
        'confidence_interval': (ci_lower, ci_upper),
        'reject_null': reject_null,
        'conclusion': 'Rechazamos H0' if reject_null else 'No rechazamos H0'
    }
    
    return results


def print_results(results: dict) -> None:
    """
    Imprime los resultados del t-test de forma clara y estructurada.
    
    Args:
        results: Diccionario con los resultados del análisis
    """
    print("\n" + "="*70)
    print("RESULTADOS DEL T-TEST")
    print("="*70)
    
    print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   • Número de experimentos: {results['n_experiments']}")
    print(f"   • Media de F1-Macro: {results['mean']:.4f}")
    print(f"   • Desviación estándar: {results['std']:.4f}")
    print(f"   • Error estándar: {results['se']:.4f}")
    print(f"   • Mínimo: {results['min']:.4f}")
    print(f"   • Mediana: {results['median']:.4f}")
    print(f"   • Máximo: {results['max']:.4f}")
    print(f"   • IC 95%: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    
    print("\n🧪 HIPÓTESIS:")
    print(f"   • H0 (Nula): μ ≤ {results['null_hypothesis_mean']}")
    print(f"   • H1 (Alternativa): μ > {results['null_hypothesis_mean']}")
    
    print("\n📈 RESULTADOS DEL T-TEST:")
    print(f"   • Estadístico t: {results['t_statistic']:.4f}")
    print(f"   • p-value (unilateral): {results['p_value']:.6f}")
    print(f"   • Nivel de significancia (α): {results['alpha']}")
    
    print("\n✅ CONCLUSIÓN:")
    if results['reject_null']:
        print(f"   → {results['conclusion']} (p-value = {results['p_value']:.6f} < α = {results['alpha']})")
        print(f"   → Evidencia estadística SIGNIFICATIVA de que la media de F1-Macro")
        print(f"      es MAYOR que {results['null_hypothesis_mean']} con {(1-results['alpha'])*100}% de confianza.")
        print(f"   → Media observada: {results['mean']:.4f}")
    else:
        print(f"   → {results['conclusion']} (p-value = {results['p_value']:.6f} ≥ α = {results['alpha']})")
        print(f"   → NO hay evidencia estadística suficiente para afirmar que la media")
        print(f"      de F1-Macro es mayor que {results['null_hypothesis_mean']}.")
        print(f"   → Media observada: {results['mean']:.4f}")
    
    print("\n" + "="*70)


def save_results(results: dict) -> None:
    """
    Guarda los resultados en un archivo JSON.
    
    Args:
        results: Diccionario con los resultados del análisis
    """
    # Creamos el directorio de reportes si no existe
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generamos nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ttest_results_{timestamp}.json"
    filepath = RESULTS_DIR / filename
    
    # Convertimos tupla de IC a lista para JSON
    results_copy = results.copy()
    results_copy['confidence_interval'] = list(results_copy['confidence_interval'])
    results_copy['timestamp'] = timestamp
    
    # Guardamos
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_copy, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {filepath}")


def main():
    """
    Función principal que ejecuta el pipeline completo del t-test.
    """
    print("\n" + "🔬" + "="*68 + "🔬")
    print("   T-TEST: F1-MACRO DE MODELOS CON GA SEARCH vs H0: μ ≤ 0.52")
    print("🔬" + "="*68 + "🔬")
    
    f1_scores = []
    
    # Ejecutamos los experimentos
    print(f"\n🚀 Ejecutando {NUM_EXPERIMENTS} experimentos...")
    
    for i in range(NUM_EXPERIMENTS):
        try:
            f1_macro = run_single_experiment(i)
            f1_scores.append(f1_macro)
        except Exception as e:
            print(f"\n❌ Error en experimento {i + 1}: {e}")
            print("Continuando con el siguiente experimento...")
            continue
    
    # Verificamos que tengamos al menos algunos resultados
    if len(f1_scores) < 5:
        print("\n❌ ERROR: No se obtuvieron suficientes resultados válidos.")
        print(f"   Solo se completaron {len(f1_scores)} experimentos de {NUM_EXPERIMENTS}.")
        print("   Se necesitan al menos 5 para realizar el análisis estadístico.")
        return
    
    print(f"\n✓ Se completaron {len(f1_scores)} experimentos exitosamente.")
    
    # Realizamos el t-test
    print("\n📊 Realizando análisis estadístico...")
    results = perform_ttest(f1_scores)
    
    # Mostramos los resultados
    print_results(results)
    
    # Guardamos los resultados
    save_results(results)
    
    print("\n✅ Análisis completado exitosamente.\n")


if __name__ == "__main__":
    main()
