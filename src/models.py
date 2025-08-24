import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# -----------------------------
# Data preparation utilities
# -----------------------------
def split_sequences(sequences, steps_in, steps_out):
    """
    Split a multivariate time series into input/output samples.

    Args:
        sequences (np.ndarray): Array with shape (timesteps, features).
        steps_in (int): Number of past timesteps to use as input.
        steps_out (int): Number of future timesteps to predict.

    Returns:
        X (np.ndarray): Input samples, shape (samples, steps_in, features).
        y (np.ndarray): Output samples, shape (samples, steps_out).
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + steps_in
        out_end_ix = end_ix + steps_out
        if out_end_ix > len(sequences) - 1:
            break
        X.append(sequences[i:end_ix, :])
        y.append(sequences[end_ix:out_end_ix, 0])  # forecast only region R1
    return np.array(X), np.array(y)


# -----------------------------
# Multistep MLP model
# -----------------------------
def multivar_multistep_mlp_fit(train_series, config):
    """
    Fit a multivariate multistep MLP model.

    Args:
        train_series (np.ndarray): Training time series (timesteps, features).
        config (dict): Model configuration dictionary.

    Returns:
        Trained Keras model.
    """
    steps_in = config['steps_in']
    steps_out = config['steps_out']
    n_neurons = config['n_neurons']
    n_epochs = config['n_epochs']
    n_batch = config['n_batch']
    activation = config['activation']
    loss = config['loss']

    # Generate samples
    X, y = split_sequences(train_series, steps_in, steps_out)

    # Flatten input for MLP
    n_input = X.shape[1] * X.shape[2]
    X = X.reshape((X.shape[0], n_input))

    # Define MLP
    model = Sequential()
    model.add(Dense(n_neurons[0], activation=activation, input_dim=n_input))
    for units in n_neurons[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(steps_out))
    model.compile(loss=loss, optimizer='adam')

    # Train
    model.fit(X, y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def multivar_multistep_mlp_predict(model, history_series, config):
    """
    Predict next n steps using trained MLP.

    Args:
        model: Trained Keras model.
        history_series (np.ndarray): Historical series (timesteps, features).
        config (dict): Model configuration.

    Returns:
        np.ndarray: Predicted sequence of length steps_out.
    """
    steps_in = config['steps_in']
    n_features = history_series.shape[1]

    X_input = history_series[-steps_in:, :]
    X_input = X_input.reshape(1, steps_in * n_features)
    yhat = model.predict(X_input, verbose=0)
    return yhat[0]


# -----------------------------
# Evaluation utilities
# -----------------------------
def evaluate_forecasts(real, predicted):
    """
    Compute RMSE per forecast horizon and overall.

    Args:
        real (np.ndarray): True sequences (samples, steps_out).
        predicted (np.ndarray): Predicted sequences (samples, steps_out).

    Returns:
        tuple: (global_rmse, [rmse_per_step])
    """
    scores = []
    for t in range(real.shape[1]):
        rmse = sqrt(mean_squared_error(real[:, t], predicted[:, t]))
        scores.append(rmse)

    # global RMSE
    s = 0
    for row in range(real.shape[0]):
        for col in range(real.shape[1]):
            s += (real[row, col] - predicted[row, col])**2
    score = sqrt(s / (real.shape[0] * real.shape[1]))
    return score, scores


def evaluate_multivar_multistep_mlp(series, test_size, config):
    """
    Evaluate a multistep MLP with walk-forward validation.

    Args:
        series (np.ndarray): Multivariate time series (timesteps, features).
        test_size (int): Number of samples reserved for test.
        config (dict): Model configuration.

    Returns:
        tuple: (global_rmse, rmse_per_step, trained_model)
    """
    steps_out = config['steps_out']
    train_series, test_series = series[0:-test_size, :], series[-test_size:, :]

    model = multivar_multistep_mlp_fit(train_series, config)

    history_series = train_series.tolist()
    predictions, real = [], []

    for i in range(len(test_series) - steps_out + 1):
        yhat_seq = multivar_multistep_mlp_predict(model, np.array(history_series), config)
        predictions.append(yhat_seq)
        real.append(test_series[i:i+steps_out, 0])
        history_series.append(test_series[i].tolist())

    predictions = np.array(predictions)
    real = np.array(real)
    score, scores = evaluate_forecasts(real, predictions)
    return score, scores, model


# -----------------------------
# Placeholders for CNN / LSTM
# -----------------------------
def multivar_multistep_cnn_fit(*args, **kwargs):
    raise NotImplementedError("CNN model not implemented yet.")

def multivar_multistep_lstm_fit(*args, **kwargs):
    raise NotImplementedError("LSTM model not implemented yet.")