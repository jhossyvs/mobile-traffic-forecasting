import pandas as pd
import numpy as np

def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw Orange Telecom dataset.
    
    Args:
        path (str): Path to CSV file (Multivariate-Mobility-Paris.csv).
    
    Returns:
        pd.DataFrame: DataFrame with datetime index and regions as columns.
    """
    df = pd.read_csv(path, sep=";")

    df["Datetime"] =  pd.to_datetime(df['Date'] + ' ' + df['Hour'], dayfirst=True)
    df = df.drop(columns=["Date", "Hour"])
    df = df.set_index("Datetime")

    return df

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 30-minute data to hourly resolution by summation.
    
    Args:
        df (pd.DataFrame): DataFrame with 30-min resolution.
    
    Returns:
        pd.DataFrame: Hourly aggregated series.
    """
    df_hourly = df.resample("1H").sum()
    return df_hourly

def scale_to_thousands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale series values by dividing by 1000 
    (so values represent thousands of users).
    
    Args:
        df (pd.DataFrame): Input DataFrame with time series.
    
    Returns:
        pd.DataFrame: Scaled DataFrame.
    """
    df_scaled = df / 1000.0
    return df_scaled

def clean_series(df, col="R1"):
    """
    Replace zeros with NaN and interpolate missing values linearly.
    Default column: R1
    """
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].interpolate(method="linear")
    return df