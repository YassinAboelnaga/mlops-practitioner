import pytest
import pandas as pd
from prodml.features import add_features


@pytest.mark.parametrize(
    "pu, do, expected",
    [
        (74, 42, "74_42"),
        (0, 0, "0_0"),
        (999, 1, "999_1"),
    ],
)
def test_add_features_pu_do_format(pu, do, expected):
    df = pd.DataFrame({"PULocationID": [pu], "DOLocationID": [do]})
    result = add_features(df)
    assert result["PU_DO"].iloc[0] == expected


def test_add_features_missing_category_raises():
    df = pd.DataFrame({"PULocationID": [74]})  # missing DOLocationID
    with pytest.raises(KeyError):
        add_features(df)


def test_add_features_zero_distance_allowed():
    df = pd.DataFrame(
        {"PULocationID": [1], "DOLocationID": [2], "trip_distance": [0.0]}
    )
    result = add_features(df)
    assert result["trip_distance"].iloc[0] == 0.0
