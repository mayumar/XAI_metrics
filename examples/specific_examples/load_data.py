# examples/specific_examples/load_data.py
from pathlib import Path
import pickle
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_data_specific_example():
    model_path = PROJECT_ROOT / "examples/models/hydraulic/IForest/hydraulic_IForest_seed_0.pkl"
    X_test_path = PROJECT_ROOT / "examples/data/hydraulic/X_test.csv"
    y_test_path = PROJECT_ROOT / "examples/data/hydraulic/y_test.csv"
    attributions_path = PROJECT_ROOT / "examples/attributions/hydraulic/IForest/LIME/hydraulic_IForest_lime_attributions.csv"

    # direct class without config
    with model_path.open("rb") as file:
        model = pickle.load(file)
    model = model.to("cpu")

    X_test = pd.read_csv(X_test_path, index_col=0)
    y_test = pd.read_csv(y_test_path, index_col=0)

    attributions_df = pd.read_csv(attributions_path, index_col=0)

    observations = attributions_df.index.astype(X_test.index.dtype).tolist()
    attributions = attributions_df.to_numpy(dtype=float)

    return model, X_test, y_test, attributions, observations