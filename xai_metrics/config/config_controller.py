from pathlib import Path
import torch
import yaml
import torch.nn as nn
import pandas as pd
from itertools import product
import pickle
import joblib

from xai_metrics.base import MetricContext

from typing import Mapping, Any, Callable, Dict, Tuple, List
import warnings

def default_model_loader(model_path: str | Path) -> Any:
    """
    Load a model from disk.

    Files with ``.pkl`` or ``.pickle`` extensions are loaded with
    :mod:`pickle`. Files with ``.joblib`` or ``.jl`` extensions are loaded
    with :mod:`joblib`. Any other file extension is loaded with
    :func:`torch.load` using ``weights_only=False``.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the saved model file.

    Returns
    -------
    Any
        Loaded model object.
    """
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as f:
            return pickle.load(f)

    if suffix in {".joblib", ".jl"}:
        return joblib.load(model_path)

    return torch.load(model_path, weights_only=False)



class ConfigController:
    """
    Controller for loading configuration files and building metric contexts.

    The controller can load a configuration from a dictionary-like object or
    from a YAML file. It supports both direct context definitions and automatic
    discovery of contexts from dataset, model and attribution directories.
    """
    def __init__(
        self,
        config: Mapping[str, Any] | str | Path = "config.yaml",
        model_loader: Callable[[str | Path], Any] | None = None
    ):
        """
        Parameters
        ----------
        config : Mapping[str, Any], str or pathlib.Path, default="config.yaml"
            Configuration source. If a mapping is provided, it is copied directly.
            If a path is provided, the YAML file is loaded. The default value loads
            the ``config.yaml`` file located next to this module.
        model_loader : Callable or None, optional
            Function used to load models from disk. The function must accept a
            model path as input and return the loaded model object. If ``None``,
            :func:`default_model_loader` is used.
        """
        self.config = self._load_config(config)
        self.model_loader = model_loader or default_model_loader


    def _load_config(self, config: Mapping[str, Any] | str | Path) -> Dict:
        """
        Load a configuration from a mapping or YAML file.

        Parameters
        ----------
        config : Mapping[str, Any], str or pathlib.Path
            Configuration mapping or path to a YAML configuration file. If the
            value is ``"config.yaml"``, the file is loaded from the directory that
            contains this module.

        Returns
        -------
        Dict
            Loaded configuration dictionary. If the YAML file is empty, an empty
            dictionary is returned.
        """
        if not isinstance(config, Mapping) and config == "config.yaml":
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
        Validate and align the indexes of ``X_test`` and ``y_test``.

        Both objects must contain exactly the same indexes. The returned
        ``y_test`` is reordered to match the index order of ``X_test``.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test input data.
        y_test : pandas.Series
            Test labels.

        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series]
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
        Validate that attribution observations exist in ``X_test``.

        If the observation identifiers do not initially match the index values
        of ``X_test``, the method tries to convert them to the same dtype as
        ``X_test_index`` before validating them.

        Parameters
        ----------
        observations : List
            Observation identifiers obtained from the attribution file index.
        X_test_index : pandas.Index
            Index of the test input data.

        Returns
        -------
        List
            Validated observation identifiers, possibly converted to the dtype
            of ``X_test_index``.

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
    

    def _validate_context_metadata(self, ctx_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Validate and return the metadata required for a metric context.

        The required metadata fields are ``dataset_name``, ``model_name`` and
        ``xai_method_name``. These values are used to identify the context when
        running metrics over one or more datasets, models and explanation
        methods.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any]
            Context configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the validated metadata fields.

        Raises
        ------
        ValueError
            If any required metadata field is missing or empty.
        """
        required = ["dataset_name", "model_name", "xai_method_name"]
        missing = [key for key in required if not ctx_cfg.get(key)]

        if missing:
            raise ValueError(
                "Context metadata is required when passing direct paths. "
                f"Missing fields: {missing}. "
                f"Required fields: {required}."
            )

        return {
            "dataset_name": ctx_cfg["dataset_name"],
            "model_name": ctx_cfg["model_name"],
            "xai_method_name": ctx_cfg["xai_method_name"],
        }
    

    def _iter_context_configs(self) -> List[Dict[str, Any]]:
        """
        Build the list of context configurations to evaluate.

        If the ``context`` section contains direct paths, a single context
        configuration is returned. If it contains ``datasets_dir``, ``models_dir``
        and ``attributions_dir``, the method searches those directories and creates
        one configuration for each valid combination of dataset, model, test data
        and attribution file.

        When automatic discovery is used, the optional ``device`` field from the
        root ``context`` section is copied into each generated context
        configuration.

        Returns
        -------
        List[Dict[str, Any]]
            List of context configuration dictionaries. Each dictionary contains
            metadata and paths required by :meth:`build_context`, namely
            ``dataset_name``, ``model_name``, ``xai_method_name``, ``model_path``,
            ``X_test_path``, ``y_test_path`` and ``attributions_path``. It may also
            contain ``device``.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or if no valid context
            configurations can be found.
        """
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        if not all(key in ctx_cfg for key in ("datasets_dir", "models_dir", "attributions_dir")):
            self._validate_context_metadata(ctx_cfg)
            return [ctx_cfg]
        
        datasets_root = Path(ctx_cfg["datasets_dir"]).expanduser().resolve()
        models_root = Path(ctx_cfg["models_dir"]).expanduser().resolve()
        attributions_root = Path(ctx_cfg["attributions_dir"]).expanduser().resolve()

        context_configs = []

        for dataset_path in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
            dataset_name = dataset_path.name

            models_dataset_root = models_root / dataset_name
            attribution_dataset_root = attributions_root / dataset_name

            if not models_dataset_root.exists() or not attribution_dataset_root.exists():
                continue

            X_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("x_test")
            )

            if not X_files:
                continue

            y_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("y_test")
            )

            if not y_files:
                continue

            for model_dir in sorted(path for path in models_dataset_root.iterdir() if path.is_dir()):
                model_name = model_dir.name
                attribution_dataset_model_root = attribution_dataset_root / model_name

                if not attribution_dataset_model_root.exists():
                    continue

                model_files = sorted(
                    path for path in model_dir.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".joblib"}
                )
                
                for xai_method_dir in sorted(
                    path for path in attribution_dataset_model_root.iterdir()
                    if path.is_dir()
                ):
                    xai_method_name = xai_method_dir.name
                    attribution_dataset_model_xai_dir = attribution_dataset_model_root / xai_method_name

                    attribution_files = sorted(
                        path for path in attribution_dataset_model_xai_dir.rglob("*.csv")
                        if path.is_file()
                    )

                    for model_path, X_path, y_path, attributions_path in product(
                        model_files,
                        X_files,
                        y_files,
                        attribution_files
                    ):
                        context_configs.append({
                            "dataset_name": dataset_name,
                            "model_name": model_name,
                            "xai_method_name": xai_method_name,
                            "model_path": str(model_path),
                            "X_test_path": str(X_path),
                            "y_test_path": str(y_path),
                            "attributions_path": str(attributions_path),
                            "device": ctx_cfg.get("device")
                        })

        if not context_configs:
            raise ValueError(
                "No MetricContext configs found from datasets_dir, models_dir and attributions_dir."
            )

        return context_configs
        
        
    def get_metrics_config(self) -> List:
        """
        Return the metrics configuration section.

        Returns
        -------
        List
            List of metric configurations. If the section is not defined, an
            empty list is returned.
        """
        return self.config.get("metrics", [])

    
    def build_context(self, ctx_cfg: Mapping[str, Any] | None = None) -> Tuple[MetricContext, Dict[str, Any]]:
        """
        Build a metric evaluation context.

        The context configuration must contain paths to the model, test data, test
        labels and attribution file. The method loads these objects, validates
        their indexes, converts attribution values to a NumPy array, optionally
        moves the model to the configured device, and returns both the
        :class:`~xai_metrics.base.MetricContext` and its identifying metadata.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any] or None, optional
            Context configuration to use. If ``None``, the ``context`` section of
            the loaded configuration is used. The configuration must contain
            ``model_path``, ``X_test_path``, ``y_test_path`` and
            ``attributions_path``. It must also contain the metadata fields
            ``dataset_name``, ``model_name`` and ``xai_method_name``. It may
            optionally contain ``device``.

        Returns
        -------
        Tuple[MetricContext, Dict[str, Any]]
            Metric context and metadata dictionary. The metadata contains
            ``dataset_name``, ``model_name`` and ``xai_method_name``.

        Raises
        ------
        ValueError
            If the ``context`` section is missing, required paths are missing,
            metadata is incomplete, or indexes are inconsistent.
        TypeError
            If the loaded model is not an instance of ``torch.nn.Module``.
        RuntimeError
            If a CUDA device is requested but CUDA is not available.

        Warns
        -----
        UserWarning
            If ``y_test`` contains more than one column. In that case, only the
            first column is used.
        """
        ctx_cfg = dict(ctx_cfg or self.config.get("context") or {})

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

        device = ctx_cfg.get("device")

        if device is not None:
            device = str(device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Device {device!r} requested but CUDA is not available.")
        
        model = self.model_loader(ctx_cfg["model_path"])

        if not isinstance(model, nn.Module):
            raise TypeError("The loaded model must be a torch.nn.Module.")

        if device is not None:
            model = model.to(device)
        
        X_test = pd.read_csv(ctx_cfg["X_test_path"], index_col=0)

        y_test_df = pd.read_csv(ctx_cfg['y_test_path'], index_col=0)
        if y_test_df.shape[1] > 1:
            warnings.warn(
                "y_test contains more than one column. "
                f"Only the first column will be used: {y_test_df.columns[0]!r}. "
                f"Ignored columns: {list(y_test_df.columns[1:])}.",
                UserWarning,
                stacklevel=2,
            )
        y_test = y_test_df.iloc[:, 0]

        X_test, y_test = self._validate_X_y_indexes(X_test, y_test)

        attributions_df = pd.read_csv(ctx_cfg['attributions_path'], index_col=0)

        observations = attributions_df.index.tolist()
        observations = self._validate_observations(observations, X_test.index)

        attributions = attributions_df.to_numpy(dtype=float)

        metric_context = MetricContext(
            model=model,
            X_test=X_test,
            y_test=y_test,
            observations=observations,
            attributions=attributions,
            device=device
        )

        metadata = self._validate_context_metadata(ctx_cfg)

        return metric_context, metadata
    
    
    def build_contexts(self) -> List[Tuple[MetricContext, Dict[str, Any]]]:
        """
        Build all metric evaluation contexts defined by the configuration.

        The method first obtains all context configurations through
        :meth:`_iter_context_configs` and then builds each corresponding
        :class:`~xai_metrics.base.MetricContext`.

        Returns
        -------
        List[Tuple[MetricContext, Dict[str, Any]]]
            List of metric contexts together with their metadata.
        """
        return [
            self.build_context(ctx_cfg)
            for ctx_cfg in self._iter_context_configs()
        ]
