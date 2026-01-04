import pandas as pd

def extract_time_features(df, datetime_col):
    """Extract hour, day, month, weekend from datetime column."""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["hour"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df
