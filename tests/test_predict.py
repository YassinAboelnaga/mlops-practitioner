def test_predict_one_returns_float(trained_model, sample_features):
    result = trained_model.predict_one(sample_features)
    assert isinstance(result, float)


def test_predict_one_sane_range(trained_model, sample_features):
    result = trained_model.predict_one(sample_features)
    assert 0 < result < 180


def test_predict_deterministic(trained_model, sample_features):
    result1 = trained_model.predict_one(sample_features)
    result2 = trained_model.predict_one(sample_features)
    assert result1 == result2


def test_predict_unseen_pu_do(trained_model):
    unseen = {"PU_DO": "9999_9999", "trip_distance": 3.0}
    result = trained_model.predict_one(unseen)
    assert isinstance(result, float)
