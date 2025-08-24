import pandas as pd

def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw Orange Telecom dataset.
    
    Args:
        path (str): Path to CSV file (Multivariate-Mobility-Paris.csv).
    
    Returns:
        pd.DataFrame: DataFrame with datetime index and regions as columns.
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
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