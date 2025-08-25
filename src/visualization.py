import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(csv_path: str, save_path: str = None):
    """
    Generate a heatmap of RMSE_global across neurons and steps_in.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with grid search results.
    save_path : str, optional
        If provided, save the plot as an image.
    """
    df = pd.read_csv(csv_path)

    # Crear tabla pivote (neurons vs steps_in)
    pivot_table = df.pivot_table(
        index="n_neurons",
        columns="steps_in",
        values="RMSE_global",
        aggfunc="min"
    )

    # Graficar heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "RMSE"}
    )
    plt.xlabel("Input window (steps_in)")
    plt.ylabel("Hidden neurons")
    plt.title("Grid Search RMSE Heatmap")
    plt.tight_layout()

    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_rmse_per_horizon(csv_path: str, best_model_index: int = None, save_path: str = None):
    """
    Plot RMSE per forecast horizon (t+1, t+2, t+3) from a grid search CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with grid search results.
    best_model_index : int, optional
        Index of the row to plot (default: the row with min RMSE_global).
    save_path : str, optional
        Path to save the figure.
    """
    df = pd.read_csv(csv_path)

    if best_model_index is None:
        best_model_index = df["RMSE_global"].idxmin()

    row = df.iloc[best_model_index]
    steps_out = int(row["steps_out"])
    rmse_values = [row[f"RMSE_t+{i+1}"] for i in range(steps_out)]
    horizons = [f"t+{i+1}" for i in range(steps_out)]

    plt.figure(figsize=(6, 4))
    plt.plot(horizons, rmse_values, marker="o", label="Best model")
    plt.title("RMSE per Forecast Horizon")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()