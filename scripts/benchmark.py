import pickle
import time
import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, clean_data, split_data
from prodml.features import add_features, to_dicts


def benchmark(predict_fn, X, n_runs=500):
    latencies = []
    for i in range(n_runs):
        start = time.perf_counter()
        predict_fn(X[i : i + 1])
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    mean = sum(latencies) / len(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    return mean, p95


def main():
    with open(settings.model_path, "rb") as f:
        dv, model = pickle.load(f)

    df = load_data(settings.data_path)
    df = clean_data(df)
    df = add_features(df)
    _, val = split_data(df, settings.test_size, settings.random_state)

    val_sample = val.iloc[:500]
    X_val = dv.transform(to_dicts(val_sample)).toarray().astype(np.float32)

    mean_pkl, p95_pkl = benchmark(lambda x: model.predict(x), X_val)

    session = ort.InferenceSession("models/model.onnx")
    input_name = session.get_inputs()[0].name
    mean_onnx, p95_onnx = benchmark(lambda x: session.run(None, {input_name: x}), X_val)

    print(f"Pickle — mean: {mean_pkl:.3f}ms, p95: {p95_pkl:.3f}ms")
    print(f"ONNX   — mean: {mean_onnx:.3f}ms, p95: {p95_onnx:.3f}ms")


if __name__ == "__main__":
    main()
