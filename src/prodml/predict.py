import pickle
import time
import functools
from typing import Any

import logging

logger = logging.getLogger("prodml.predict")


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed*1000:.2f} ms")
        return result

    return wrapper


class DurationPredictor:
    def __init__(self, dv, model):
        self.dv = dv
        self.model = model

    @classmethod
    def load(cls, model_path: str) -> "DurationPredictor":
        with open(model_path, "rb") as f_in:
            dv, model = pickle.load(f_in)
        return cls(dv, model)

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        X = self.dv.transform([features])
        pred = self.model.predict(X)
        return float(pred[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        X = self.dv.transform(features_list)
        preds = self.model.predict(X)
        return [float(p) for p in preds]
