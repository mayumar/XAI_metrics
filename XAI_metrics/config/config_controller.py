from pathlib import Path
import torch
import yaml
import torch.nn as nn
import pandas as pd

from typing import Mapping, Any, Callable

def default_model_loader(model_path):
    return torch.load(model_path)

class ConfigController:
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path = "config.yaml",
        model_loader: Callable | None = None
    ):
        self.config = self._load_config(config)
        self.model_loader = model_loader or default_model_loader

    def _load_config(self, config: Mapping[str, Any] | str | Path):
        if config is "config.yaml":
            config_path = Path(__file__).resolve().parent / config
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
            
        if isinstance(config, Mapping):
            return dict(config)
        
        config_path = Path(config)
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
        
        
    def get_metrics_config(self):
        return self.config.get("metrics", [])
    

    def _validate_X_y_indexes(self, X_test, y_test):
        only_in_X = X_test.index.difference(y_test.index).tolist()
        only_in_y = y_test.index.difference(X_test.index).tolist()

        if only_in_X or only_in_y:
            raise ValueError(
                "X_test and y_test must contain the same indexes. "
                f"Indexes only in X_test: {only_in_X}. "
                f"Indexes only in y_test: {only_in_y}."
            )

        y_test = y_test.loc[X_test.index]

        return X_test, y_test

    
    def build_context(self):
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        required = [
            "model_path",
            "X_test_path",
            "y_test_path",
            "attributions_path",
        ]

        missing = [key for key in required if key not in ctx_cfg]
        if missing:
            raise ValueError(f"Missing context config fields: {missing}")
        
        model = self.model_loader(ctx_cfg['model_path'])
        if not isinstance(model, nn.Module):
            raise TypeError("The loaded model must be a torch.nn.Module.")
        
        X_test = pd.read_csv(ctx_cfg["X_test_path"], index_col=0)
        y_test = pd.read_csv(ctx_cfg['y_test_path'], index_col=0).squeeze("columns")

        self._validate_X_y_indexes(X_test, y_test)