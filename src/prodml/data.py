from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(
        subset=[
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "trip_distance",
            "fare_amount",
        ]
    )
    df = df[
        (df["lpep_pickup_datetime"] >= "2025-01-01")
        & (df["lpep_pickup_datetime"] < "2025-02-01")
    ]
    df = df[df["trip_distance"] > 0]
    df = df[df["fare_amount"] > 0]

    df["trip_duration_min"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] < 180)]
    df = df[df["passenger_count"] > 0]
    df = df[df["total_amount"] < df["total_amount"].quantile(0.99)]
    df = df.dropna(subset=["trip_type"])
    df = df.drop(columns=["ehail_fee"], errors="ignore")
    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, val
