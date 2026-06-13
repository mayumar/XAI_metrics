import numpy as np
import pandas as pd


def test_preprocess_phmd_splits_removes_identifiers_and_scales():
    from prueba.data_processing import preprocess_phmd_splits

    data = {
        "train": pd.DataFrame(
            {
                "sensor_a": [10.0, 20.0, 30.0],
                "sensor_b": [1.0, 3.0, 5.0],
                "fault": [0, 0, 1],
                "unit": ["u1", "u1", "u2"],
            }
        ),
        "val": pd.DataFrame(
            {
                "sensor_a": [15.0],
                "sensor_b": [2.0],
                "fault": [0],
                "unit": ["u3"],
            }
        ),
        "test": pd.DataFrame(
            {
                "sensor_a": [25.0],
                "sensor_b": [4.0],
                "fault": [1],
                "unit": ["u4"],
            }
        ),
    }

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_norm,
        X_val_norm,
        X_test_norm,
        anomalias_fraccion,
    ) = preprocess_phmd_splits(
        data,
        target="fault",
        identifier_columns=["unit"],
    )

    assert list(X_train.columns) == ["sensor_a", "sensor_b"]
    assert X_train.iloc[0].tolist() == [10.0, 1.0]
    assert np.allclose(X_train_norm.min().to_numpy(), 0.0)
    assert np.allclose(X_train_norm.max().to_numpy(), 1.0)
    assert y_train.tolist() == [0, 0, 1]
    assert y_val.tolist() == [0]
    assert y_test.tolist() == [1]
    assert anomalias_fraccion == 2 / 5
    assert X_val_norm.iloc[0].tolist() == [0.25, 0.25]
    assert X_test_norm.iloc[0].tolist() == [0.75, 0.75]
