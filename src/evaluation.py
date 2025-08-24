import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Tuple, List

def evaluate_forecasts(
    real: np.ndarray, 
    predicted: np.ndarray
) -> Tuple[float, List[float]]:
    """
    Evaluate forecasts using RMSE.
    
    Args:
        real (np.ndarray): Array of true values [samples, steps_out].
        predicted (np.ndarray): Array of predictions [samples, steps_out].
    
    Returns:
        Tuple[float, List[float]]:
            - Global RMSE
            - List of RMSE per forecast horizon
    """
    scores = []
    # RMSE per timestep
    for t in range(real.shape[1]):
        rmse = sqrt(mean_squared_error(real[:, t], predicted[:, t]))
        scores.append(rmse)

    # Global RMSE
    s = 0
    for row in range(real.shape[0]):
        for col in range(real.shape[1]):
            s += (real[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (real.shape[0] * real.shape[1]))

    return score, scores