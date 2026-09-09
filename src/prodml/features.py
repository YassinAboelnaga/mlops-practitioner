from typing import Any
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

FEATURES = ["PU_DO", "trip_distance"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


def to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df[FEATURES].to_dict(orient="records")


def fit_vectorizer(dicts: list[dict[str, Any]]) -> DictVectorizer:
    dv = DictVectorizer()
    dv.fit(dicts)
    return dv
