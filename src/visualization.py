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