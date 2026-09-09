import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from prodml.config import settings
from prodml.data import load_data, clean_data, split_data
from prodml.features import add_features, to_dicts, fit_vectorizer

import logging
from prodml.logging_conf import setup_logging

logger = logging.getLogger("prodml.train")


def main() -> None:
    setup_logging()
    df = load_data(settings.data_path)
    df = clean_data(df)
    df = add_features(df)

    train, val = split_data(df, settings.test_size, settings.random_state)

    train_dicts = to_dicts(train)
    val_dicts = to_dicts(val)

    dv = fit_vectorizer(train_dicts)
    X_train = dv.transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = train["trip_duration_min"].values
    y_val = val["trip_duration_min"].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Validation RMSE: {rmse:.3f}")
    logger.info(f"Validation MAE: {mae:.3f}")

    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    with open(settings.model_path, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    os.makedirs(os.path.dirname(settings.report_path), exist_ok=True)
    report_content = f"""# Module 2 — Refactored Package Model

**Validation RMSE:** {rmse:.3f}
**Validation MAE:** {mae:.3f}
"""
    existing = ""
    if os.path.exists(settings.report_path):
        with open(settings.report_path) as f:
            existing = f.read()
    with open(settings.report_path, "w") as f:
        f.write(report_content + existing)


if __name__ == "__main__":
    main()
