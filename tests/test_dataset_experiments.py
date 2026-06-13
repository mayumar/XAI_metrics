import pandas as pd


def _write_split(dataset_dir, split, X, y):
    X.to_csv(dataset_dir / f"X_{split}.csv", index=False)
    pd.Series(y, name="target").to_csv(
        dataset_dir / f"y_{split}.csv",
        index=False,
    )


def test_load_and_prepare_dataset_splits(tmp_path):
    from prueba.run_dataset_experiments import (
        load_dataset_splits,
        prepare_training_data,
    )

    dataset_dir = tmp_path / "example"
    dataset_dir.mkdir()

    _write_split(
        dataset_dir,
        "train",
        pd.DataFrame(
            {
                "a": [0.0, 1.0],
                "b": [10.0, 20.0],
                "unit": ["u1", "u2"],
                "fault": [0, 1],
            }
        ),
        [0, 1],
    )
    _write_split(
        dataset_dir,
        "val",
        pd.DataFrame(
            {
                "a": [2.0, 3.0],
                "b": [30.0, 40.0],
                "unit": ["u3", "u4"],
                "fault": [0, 0],
            }
        ),
        [0, 0],
    )
    _write_split(
        dataset_dir,
        "test",
        pd.DataFrame(
            {
                "a": [1.5],
                "b": [25.0],
                "unit": ["u5"],
                "fault": [1],
            }
        ),
        [1],
    )

    splits = load_dataset_splits(dataset_dir)
    X_fit, y_fit, X_test, y_test, contamination, scaler = (
        prepare_training_data(splits)
    )

    assert X_fit.shape == (4, 2)
    assert y_fit.tolist() == [0, 1, 0, 0]
    assert X_fit.min().tolist() == [0.0, 0.0]
    assert X_fit.max().tolist() == [1.0, 1.0]
    assert X_test.iloc[0].tolist() == [0.5, 0.5]
    assert y_test.tolist() == [1]
    assert contamination == 0.25
    assert scaler is not None


def test_select_observations_includes_both_classes():
    from prueba.run_dataset_experiments import select_observations

    y_test = pd.Series([0, 0, 0, 1, 1, 1])
    observations = select_observations(y_test, number=4, seed=42)

    assert len(observations) == 4
    assert set(y_test.loc[observations]) == {0, 1}
