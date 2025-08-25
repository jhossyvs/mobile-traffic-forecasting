import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.models import save_model

from src.training import train_mlp_walk_forward


def grid_search_mlp_parallel(series, test_size, param_grid,
                             models_folder="results/models",
                             csv_path="results/metrics/gridsearch_results.csv",
                             n_jobs=4):
    """
    Execute a parallel grid search for MLP hyperparameters.
    Saves metrics and trained models.

    Parameters
    ----------
    series : np.ndarray
        Multivariate time series (time_steps, features).
    test_size : int
        Number of samples reserved for testing (walk-forward validation).
    param_grid : dict
        Dictionary of hyperparameters to explore.
    models_folder : str
        Folder to save trained models.
    csv_path : str
        Path to save the CSV with results.
    n_jobs : int
        Number of parallel processes.

    Returns
    -------
    best_config : dict
        Configuration of the best model.
    best_score : float
        Global RMSE of the best model.
    """

    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    grid = list(ParameterGrid(param_grid))

    def train_and_save(i, config):
        try:
            start = time.time()
            score, scores, model = train_mlp_walk_forward(series, test_size, config)

            model_name = f"{models_folder}/model_{i}_RMSE_{score:.4f}.keras"
            save_model(model, model_name)

            result = config.copy()
            result["model"] = model_name
            result["RMSE_global"] = score
            for j, s in enumerate(scores):
                result[f"RMSE_t+{j+1}"] = s

            elapsed = time.time() - start
            print(f"✅ Process {i+1}/{len(grid)} - Config: {config} => RMSE: {score:.4f} | Time: {elapsed:.2f}s")

            return result, score, model
        except Exception as e:
            print(f"❌ Error in config {config}: {e}")
            return None

    full_results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_save)(i, cfg) for i, cfg in enumerate(grid)
    )

    # Filter valid results
    full_results = [r for r in full_results if r is not None]
    if not full_results:
        print("⚠️ No model was successfully trained.")
        return None, None

    results, scores, models = zip(*full_results)

    # Save CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)

    # Select best model
    best_idx = np.argmin(scores)
    best_config = results[best_idx]
    best_score = scores[best_idx]
    best_model = models[best_idx]

    save_model(best_model, f"{models_folder}/best_model.keras")
    print("\n🟢 Best configuration found:")
    print(best_config)
    print(f"✅ Model saved as: {models_folder}/best_model.keras")

    return best_config, best_score