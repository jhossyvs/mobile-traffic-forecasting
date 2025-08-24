import numpy as np
from typing import Tuple
from .models import fit_mlp, predict_mlp
from .evaluation import evaluate_forecasts

def train_mlp_walk_forward(
    series: np.ndarray,
    test_size: int,
    config: dict
) -> Tuple[float, list, object]:
    """
    Train and evaluate an MLP model with walk-forward validation.
    
    Args:
        series (np.ndarray): Multivariate time series [timesteps, features].
        test_size (int): Number of timesteps to reserve for testing.
        config (dict): Model hyperparameters.
    
    Returns:
        Tuple[float, list, object]: 
            - Global RMSE score
            - List of RMSE per forecast horizon
            - Trained MLP model
    """
    steps_out = config['steps_out']

    # Split train/test
    train_series, test_series = series[0:-test_size, :], series[-test_size:, :]

    # Fit model once on training set
    model = fit_mlp(train_series, config)

    # History starts with training set
    history_series = train_series.tolist()

    predictions, real = list(), list()

    for i in range(len(test_series) - steps_out + 1):
        # Forecast
        yhat_sequence = predict_mlp(model, np.array(history_series), config)
        predictions.append(yhat_sequence)

        # True values (target = region R1 assumed in col 0)
        real.append(test_series[i:i+steps_out, 0])

        # Update history with one step
        history_series.append(test_series[i].tolist())

    predictions = np.array(predictions)
    real = np.array(real)

    # Evaluate forecasts
    score, scores = evaluate_forecasts(real, predictions)
    return score, scores, model