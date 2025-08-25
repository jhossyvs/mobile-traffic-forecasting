import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import save_model

from src.training import train_mlp_walk_forward


def grid_search_mlp_parallel(series, test_size, param_grid,
                             carpeta_modelos="results/models",
                             nombre_csv="results/metrics/resultados_gridsearch.csv",
                             n_jobs=4):
    """
    Ejecuta una búsqueda en grid de hiperparámetros para MLP en paralelo.
    Guarda métricas y modelos entrenados.

    Parameters
    ----------
    series : np.ndarray
        Serie multivariada (time_steps, features).
    test_size : int
        Cantidad de muestras para test (walk-forward).
    param_grid : dict
        Diccionario de hiperparámetros.
    carpeta_modelos : str
        Carpeta donde guardar modelos.
    nombre_csv : str
        Ruta del CSV con resultados.
    n_jobs : int
        Número de procesos en paralelo.

    Returns
    -------
    mejor_config : dict
        Configuración del mejor modelo.
    mejor_score : float
        RMSE global del mejor modelo.
    """

    os.makedirs(carpeta_modelos, exist_ok=True)
    os.makedirs(os.path.dirname(nombre_csv), exist_ok=True)

    grid = list(ParameterGrid(param_grid))

    def entrenar_y_guardar(i, config):
        try:
            start = time.time()
            score, scores, modelo = train_mlp_walk_forward(series, test_size, config)

            nombre_modelo = f"{carpeta_modelos}/modelo_{i}_RMSE_{score:.4f}.keras"
            save_model(modelo, nombre_modelo)

            resultado = config.copy()
            resultado["modelo"] = nombre_modelo
            resultado["RMSE_global"] = score
            for j, s in enumerate(scores):
                resultado[f"RMSE_t+{j+1}"] = s

            elapsed = time.time() - start
            print(f"✅ Proceso {i+1}/{len(grid)} - Config: {config} => RMSE: {score:.4f} | Tiempo: {elapsed:.2f}s")

            return resultado, score, modelo
        except Exception as e:
            print(f"❌ Error en config {config}: {e}")
            return None

    resultados_completos = Parallel(n_jobs=n_jobs)(
        delayed(entrenar_y_guardar)(i, cfg) for i, cfg in enumerate(grid)
    )

    # Filtrar válidos
    resultados_completos = [r for r in resultados_completos if r is not None]
    if not resultados_completos:
        print("⚠️ No se logró entrenar ningún modelo exitosamente.")
        return None, None

    resultados, scores, modelos = zip(*resultados_completos)

    # Guardar CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(nombre_csv, index=False)

    # Seleccionar mejor
    idx_mejor = np.argmin(scores)
    mejor_config = resultados[idx_mejor]
    mejor_score = scores[idx_mejor]
    mejor_modelo = modelos[idx_mejor]

    save_model(mejor_modelo, f"{carpeta_modelos}/mejor_modelo.keras")
    print("\n🟢 Mejor configuración encontrada:")
    print(mejor_config)
    print(f"✅ Modelo guardado como: {carpeta_modelos}/mejor_modelo.keras")

    return mejor_config, mejor_score