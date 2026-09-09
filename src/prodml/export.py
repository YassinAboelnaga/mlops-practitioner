import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from prodml.config import settings


def export_to_onnx(pkl_path: str, onnx_path: str) -> None:
    with open(pkl_path, "rb") as f:
        dv, model = pickle.load(f)
    n_features = len(dv.get_feature_names_out())
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    export_to_onnx(settings.model_path, "models/model.onnx")
