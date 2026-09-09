import pickle
import numpy as np
import onnxruntime as ort
from prodml.config import settings
from prodml.data import load_data, clean_data, split_data
from prodml.features import add_features, to_dicts


def test_pickle_onnx_parity():
    with open(settings.model_path, "rb") as f:
        dv, model = pickle.load(f)

    df = load_data(settings.data_path)
    df = clean_data(df)
    df = add_features(df)
    _, val = split_data(df, settings.test_size, settings.random_state)

    val_sample = val.iloc[:500]
    X_val = dv.transform(to_dicts(val_sample)).toarray().astype(np.float32)

    pred_pkl = model.predict(X_val)

    session = ort.InferenceSession("models/model.onnx")
    input_name = session.get_inputs()[0].name
    pred_onnx = session.run(None, {input_name: X_val})[0].flatten()

    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)
