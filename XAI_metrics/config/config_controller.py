from pathlib import Path
import torch
import yaml
import torch.nn as nn
import pandas as pd

from XAI_metrics.base import MetricContext

from typing import Mapping, Any, Callable, Dict, Tuple, List
import warnings

def default_model_loader(model_path: str | Path):
    """
    Load a PyTorch model from disk.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the saved PyTorch model.

    Returns
    -------
    Any
        Loaded object returned by ``torch.load``.
    """
    return torch.load(model_path)

class ConfigController:
    """
    Controller for loading configuration files and building metric contexts.

    The controller loads a configuration from a mapping or YAML file, validates
    the required input data, loads the model, and creates a
    :class:`MetricContext` object used by the metric evaluation pipeline.

    Parameters
    ----------
    config : Mapping[str, Any], str or pathlib.Path, default="config.yaml"
        Configuration source. If a mapping is provided, it is copied directly.
        If a path is provided, the YAML file is loaded. The default value loads
        the ``config.yaml`` file located next to this module.
    model_loader : Callable, optional
        Function used to load the model from ``model_path``. If not provided,
        :func:`default_model_loader` is used.
    """
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path = "config.yaml",
        model_loader: Callable | None = None
    ):
        self.config = self._load_config(config)
        self.model_loader = model_loader or default_model_loader


    def _load_config(self, config: Mapping[str, Any] | str | Path) -> Dict:
        """
        Load the configuration from a mapping or YAML file.

        Parameters
        ----------
        config : Mapping[str, Any], str or pathlib.Path
            Configuration mapping or path to a YAML configuration file.

        Returns
        -------
        Dict
            Loaded configuration dictionary.
        """
        if config is "config.yaml":
            config_path = Path(__file__).resolve().parent / config
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
            
        if isinstance(config, Mapping):
            return dict(config)
        
        config_path = Path(config)
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    

    def _validate_X_y_indexes(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Validate that ``X_test`` and ``y_test`` contain the same indexes.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test input data.
        y_test : pandas.Series
            Test labels.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            ``X_test`` and ``y_test`` aligned using the index order of
            ``X_test``.

        Raises
        ------
        ValueError
            If ``X_test`` and ``y_test`` do not contain the same indexes.
        """
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
    

    def _validate_observations(
        self,
        observations: List,
        X_test_index: pd.Index
    ) -> List:
        """
        Validate that attribution observations exist in the test data index.

        If possible, the observation indexes are converted to the same dtype as
        ``X_test_index`` before validation.

        Parameters
        ----------
        observations : list
            Observation identifiers obtained from the attributions file.
        X_test_index : pandas.Index
            Index of the test input data.

        Returns
        -------
        list
            Validated observation identifiers.

        Raises
        ------
        ValueError
            If the observations cannot be converted to the dtype of
            ``X_test_index`` or if any observation is missing from ``X_test``.
        """
        observations_index = pd.Index(observations)

        if observations_index.isin(X_test_index).all():
            return observations
        
        try:
            observations_typed_idx = observations_index.astype(X_test_index.dtype).tolist()
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Observations from attributions_path are not compatible with "
                "the index type of X_test. "
                f"Observations dtype: {observations_index.dtype}. "
                f"X_test index dtype: {X_test_index.dtype}. "
            ) from e
        
        missing = [obs for obs in observations_typed_idx if obs not in X_test_index]

        if missing:
            raise ValueError(
                "All observations must exist in both X_test and y_test. "
                f"Missing in X_test: {missing}. "
            )

        return observations_typed_idx
        
        
    def get_metrics_config(self) -> List:
        """
        Return the metrics configuration section.

        Returns
        -------
        list
            List of metric configurations. If the section is not defined, an
            empty list is returned.
        """
        return self.config.get("metrics", [])

    
    def build_context(self) -> MetricContext:
        """
        Build a metric evaluation context from the loaded configuration.

        The configuration must include a ``context`` section with the paths to
        the model, test data, labels, and attributions. The method loads all
        required objects, validates their indexes, and returns a
        :class:`MetricContext` instance.

        Returns
        -------
        MetricContext
            Context object containing the model, test data, labels,
            observations, and attribution values.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or required fields are not
            provided.
        TypeError
            If the loaded model is not an instance of ``torch.nn.Module``.
        """
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
        y_test = pd.read_csv(ctx_cfg['y_test_path'], index_col=0).iloc[:, 0]

        if y_test.shape[1] > 1:
            warnings.warn(
                "y_test contains more than one column. "
                f"Only the first column will be used: {y_test.columns[0]!r}. "
                f"Ignored columns: {list(y_test.columns[1:])}.",
                UserWarning,
                stacklevel=2,
            )

        X_test, y_test = self._validate_X_y_indexes(X_test, y_test)

        attributions_df = pd.read_csv(ctx_cfg['attributions_path'], index_col=0)

        observations = attributions_df.index.tolist()
        observations = self._validate_observations(observations, X_test.index)

        attributions = attributions_df.to_numpy(dtype=float)

        extras = {}

        return MetricContext(
            model=model,
            X_test=X_test,
            y_test=y_test,
            observations=observations,
            attributions=attributions
        )