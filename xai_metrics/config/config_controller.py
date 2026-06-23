# xai_metrics/config/config_controller.py
from pathlib import Path
import torch
import yaml
import torch.nn as nn
import pandas as pd
from itertools import product
import pickle
import joblib

from xai_metrics.base import MetricContext, ExplainerContext

from typing import Mapping, Any, Callable, Dict, Tuple, List
import warnings

def default_model_loader(model_path: str | Path) -> Any:
    """
    Load a model from disk.

    Files with ``.pkl`` or ``.pickle`` extensions are loaded with
    :mod:`pickle`. Files with ``.joblib`` or ``.jl`` extensions are loaded with
    :mod:`joblib`. Any other file extension is loaded with :func:`torch.load`
    using ``weights_only=False``.

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
    Controller for loading configuration files and building contexts.

    The controller can load configuration data from a mapping or from a YAML
    file. It supports direct context definitions and automatic discovery from
    dataset, model and attribution directories.

    Metric contexts include models, test data, labels and precomputed
    attributions. Explainer contexts include models, background data and,
    optionally, batches to explain.
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
            Configuration source. If a mapping is provided, it is copied
            directly. If a path is provided, the YAML file is loaded. The
            default value loads the ``config.yaml`` file located next to this
            module.
        model_loader : Callable[[str or pathlib.Path], Any] or None, optional
            Function used to load models from disk. The function must receive a
            model path and return the loaded model object. If ``None``,
            :func:`default_model_loader` is used.
        """
        self.config = self._load_config(config)
        self.model_loader = model_loader or default_model_loader


    def _load_config(self, config: Mapping[str, Any] | str | Path) -> Dict:
        """
        Load configuration data from a mapping or YAML file.

        Parameters
        ----------
        config : Mapping[str, Any], str or pathlib.Path
            Configuration mapping or path to a YAML configuration file. If the
            value is ``"config.yaml"``, the file is loaded from the directory
            containing this module.

        Returns
        -------
        Dict
            Loaded configuration dictionary. If the YAML file is empty, an
            empty dictionary is returned.
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
        Validate and align feature and target indexes.

        Both objects must contain exactly the same indexes. The returned target
        series is reordered to match the index order of the feature dataframe.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Input data.
        y_test : pandas.Series
            Labels or targets associated with ``X_test``.

        Returns
        -------
        Tuple[pandas.DataFrame, pandas.Series]
            Input data and target values aligned using the index order of
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


    def _normalise_class_labels(self, labels: pd.Series) -> pd.Series:
        """Return integer-valued class labels with an integer dtype.

        CSV files containing labels such as ``0.0`` and ``1.0`` are commonly
        inferred as floating point by pandas. Quantus uses these values as
        class-column indexes, so integer-valued numeric labels must be exposed
        as integers. Non-numeric or genuinely continuous targets are left
        unchanged.
        """
        numeric_labels = pd.to_numeric(labels, errors="coerce")

        if numeric_labels.notna().all() and (numeric_labels % 1 == 0).all():
            return numeric_labels.astype("int64")

        return labels
    

    def _validate_observations(
        self,
        observations: List,
        X_test_index: pd.Index
    ) -> List:
        """
        Validate that attribution observations exist in the input data.

        If the observation identifiers do not initially match the input index,
        the method tries to convert them to the same dtype as ``X_test_index``
        before validating them.

        Parameters
        ----------
        observations : List
            Observation identifiers obtained from the attribution file index.
        X_test_index : pandas.Index
            Index of the input data.

        Returns
        -------
        List
            Validated observation identifiers, possibly converted to the dtype
            of ``X_test_index``.

        Raises
        ------
        ValueError
            If the observations cannot be converted to the dtype of
            ``X_test_index`` or if any observation is missing.
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
    

    def _validate_metric_context_metadata(self, ctx_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Validate and return metadata required for a metric context.

        Metric contexts must be identified by dataset, model and explanation
        method names. These fields are used to organise metric results.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any]
            Context configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing ``dataset_name``, ``model_name`` and
            ``xai_method_name``.

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
    

    def _validate_explainer_context_metadata(self, ctx_cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Validate and return metadata required for an explainer context.

        Explainer contexts must be identified by dataset and model names. These
        fields are used to associate generated explanations with the
        corresponding data and model.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any]
            Context configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing ``dataset_name`` and ``model_name``.

        Raises
        ------
        ValueError
            If any required metadata field is missing or empty.
        """
        required = ["dataset_name", "model_name"]
        missing = [key for key in required if not ctx_cfg.get(key)]

        if missing:
            raise ValueError(
                "Explainer context metadata is required when passing direct paths. "
                f"Missing fields: {missing}. "
                f"Required fields: {required}."
            )
        
        return {
            "dataset_name": ctx_cfg['dataset_name'],
            "model_name": ctx_cfg['model_name']
        }
    

    def _iter_metric_context_configs(self) -> List[Dict[str, Any]]:
        """
        Build metric context configurations from the loaded configuration.

        If the ``context`` section contains direct paths, a single context
        configuration is returned. If it contains ``datasets_dir``,
        ``models_dir`` and ``attributions_dir``, the method searches those
        directories and creates one configuration for each valid combination of
        dataset, model, input data, labels and attribution file.

        The generated configurations always use the keys expected by
        :meth:`build_metric_context`, even when the discovered files are named
        as batches rather than test sets.

        Returns
        -------
        List[Dict[str, Any]]
            List of metric context configuration dictionaries.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or no valid metric context
            configurations can be found.
        """
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        if not all(key in ctx_cfg for key in ("datasets_dir", "models_dir", "attributions_dir")):
            self._validate_metric_context_metadata(ctx_cfg)
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
                if path.name.lower().startswith("x_batch")
            )
            test = False

            if not X_files:
                test = True
                X_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("x_test")
                )

            if not X_files:
                continue

            if not test:
                y_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_batch")
                )
            else:
                y_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_test")
                )

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
    

    def _iter_explainer_context_configs(self) -> List[Dict[str, Any]]:
        """
        Build explainer context configurations from the loaded configuration.

        If the ``context`` section contains direct paths, a single context
        configuration is returned. If it contains ``datasets_dir`` and
        ``models_dir``, the method searches those directories and creates one
        configuration for each valid combination of dataset, model, background
        data and batch data.

        Background data are discovered from files starting with
        ``x_background`` or, if none are found, ``x_train``. Batch data are
        discovered from files starting with ``x_batch`` or, if none are found,
        ``x_test``. Target files are added when matching ``y_background``,
        ``y_train``, ``y_batch`` or ``y_test`` files are available.

        Returns
        -------
        List[Dict[str, Any]]
            List of explainer context configuration dictionaries.

        Raises
        ------
        ValueError
            If the ``context`` section is missing or no valid explainer context
            configurations can be found.
        """
        ctx_cfg = self.config.get("context")

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")
        
        if not all(key in ctx_cfg for key in ("datasets_dir", "models_dir")):
            self._validate_metric_context_metadata(ctx_cfg)
            return [ctx_cfg]
        
        datasets_root = Path(ctx_cfg['datasets_dir']).expanduser().resolve()
        models_root = Path(ctx_cfg['models_dir']).expanduser().resolve()

        context_configs = []

        for dataset_path in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
            dataset_name = dataset_path.name
            models_dataset_root = models_root / dataset_name

            if not models_dataset_root.exists():
                continue

            X_background_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("x_background")
            )
            train = False

            if not X_background_files:
                train = True
                X_background_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("x_train")
                )

            if not X_background_files:
                continue

            if not train:
                y_background_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_background")
                )
            else:
                y_background_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_train")
                )

            X_batch_files = sorted(
                path for path in dataset_path.glob("*.csv")
                if path.name.lower().startswith("x_batch")
            )
            test = False

            if not X_batch_files:
                test = True
                X_batch_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("x_test")
                )

            if not X_batch_files:
                continue

            if not test:
                y_batch_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_batch")
                )
            else:
                y_batch_files = sorted(
                    path for path in dataset_path.glob("*.csv")
                    if path.name.lower().startswith("y_test")
                )

            for model_dir in sorted(path for path in models_dataset_root.iterdir() if path.is_dir()):
                model_name = model_dir.name

                model_files = sorted(
                    path for path in model_dir.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".joblib"}
                )

                for model_path, X_background_path, X_batch_path in product(
                    model_files,
                    X_background_files,
                    X_batch_files
                ):
                    base_config = {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "X_background_path": str(X_background_path),
                        "X_batch_path": str(X_batch_path),
                        "device": ctx_cfg.get("device"),
                    }

                    if y_background_files:
                        base_config["y_background_path"] = str(y_background_files[0])

                    if y_batch_files:
                        base_config["y_batch_path"] = str(y_batch_files[0])

                    context_configs.append(base_config)

        if not context_configs:
            raise ValueError(
                "No ExplainerContext configs found from datasets_dir and models_dir."
            )
        
        return context_configs
        
        
    def get_metrics_config(self) -> List:
        """
        Return the metrics configuration section.

        Returns
        -------
        List
            List of metric configuration dictionaries. If the section is not
            defined, an empty list is returned.
        """
        return self.config.get("metrics", [])
    

    def get_explainers_config(self) -> List:
        """
        Return the explainers configuration section.

        Returns
        -------
        List
            List of explainer configuration dictionaries. If the section is not
            defined, an empty list is returned.
        """
        return self.config.get("explainers", [])

    
    def build_metric_context(self, ctx_cfg: Mapping[str, Any] | None = None) -> Tuple[MetricContext, Dict[str, Any]]:
        """
        Build a metric evaluation context.

        The context configuration must contain paths to the model, input data,
        labels and attribution file. The method loads these objects, validates
        indexes, converts attribution values to a NumPy array, optionally moves
        the model to the configured device and returns both the
        :class:`MetricContext` and its metadata.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any] or None, optional
            Context configuration to use. If ``None``, the ``context`` section
            of the loaded configuration is used. The configuration must contain
            ``model_path``, ``X_test_path``, ``y_test_path`` and
            ``attributions_path``. It must also contain ``dataset_name``,
            ``model_name`` and ``xai_method_name``. It may optionally contain
            ``device``.

        Returns
        -------
        Tuple[MetricContext, Dict[str, Any]]
            Metric context and metadata dictionary.

        Raises
        ------
        ValueError
            If the ``context`` section is missing, required paths are missing,
            metadata are incomplete or indexes are inconsistent.
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
        y_test = self._normalise_class_labels(y_test)

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

        metadata = self._validate_metric_context_metadata(ctx_cfg)

        return metric_context, metadata


    def build_explainers_context(
        self,
        ctx_cfg: Mapping[str, Any] | None = None
    ) -> Tuple[ExplainerContext, Dict[str, Any]]:
        """
        Build an explainer context.

        The context configuration must contain a model path and background
        input data. It may also contain background labels, a batch of inputs to
        explain, batch labels and a device. The method loads these objects,
        validates batch indexes when labels are available and returns both the
        :class:`ExplainerContext` and its metadata.

        Parameters
        ----------
        ctx_cfg : Mapping[str, Any] or None, optional
            Context configuration to use. If ``None``, the ``context`` section
            of the loaded configuration is used. The configuration must contain
            ``model_path`` and ``X_background_path``. It must also contain
            ``dataset_name`` and ``model_name``. It may optionally contain
            ``y_background_path``, ``X_batch_path``, ``X_test_path``,
            ``y_batch_path``, ``y_test_path`` and ``device``.

        Returns
        -------
        Tuple[ExplainerContext, Dict[str, Any]]
            Explainer context and metadata dictionary.

        Raises
        ------
        ValueError
            If the ``context`` section is missing, required fields are missing,
            metadata are incomplete or batch indexes are inconsistent.
        RuntimeError
            If a CUDA device is requested but CUDA is not available.

        Warns
        -----
        UserWarning
            If a target file contains more than one column. In that case, only
            the first column is used.
        """
        ctx_cfg = dict(ctx_cfg or self.config.get("context") or {})

        if not ctx_cfg:
            raise ValueError("Config must include a 'context' section.")

        required = ["X_background_path"]
        missing = [key for key in required if key not in ctx_cfg]

        if missing:
            raise ValueError(f"Missing explainer context config fields: {missing}")
            
        X_background = pd.read_csv(ctx_cfg["X_background_path"], index_col=0)

        y_background = None
        if "y_background_path" in ctx_cfg:
            y_background_df = pd.read_csv(ctx_cfg["y_background_path"], index_col=0)

            if y_background_df.shape[1] > 1:
                warnings.warn(
                    "y_background contains more than one column. "
                    f"Only the first column will be used: {y_background_df.columns[0]!r}. "
                    f"Ignored columns: {list(y_background_df.columns[1:])}.",
                    UserWarning,
                    stacklevel=2,
                )

            y_background = y_background_df.iloc[:, 0]
            y_background = self._normalise_class_labels(y_background)
        
        model = None
        if "model_path" in ctx_cfg:
            model = self.model_loader(ctx_cfg["model_path"])
        
        X_batch_path = ctx_cfg.get("X_batch_path", ctx_cfg.get("X_test_path"))
        
        X_batch = None
        if X_batch_path is not None:
            X_batch = pd.read_csv(X_batch_path, index_col=0)
        
        y_batch_path = ctx_cfg.get("y_batch_path", ctx_cfg.get("y_test_path"))

        y_batch = None
        if y_batch_path is not None:
            y_batch_df = pd.read_csv(y_batch_path, index_col=0)

            if y_batch_df.shape[1] > 1:
                warnings.warn(
                    "y_batch contains more than one column. "
                    f"Only the first column will be used: {y_batch_df.columns[0]!r}. "
                    f"Ignored columns: {list(y_batch_df.columns[1:])}.",
                    UserWarning,
                    stacklevel=2,
                )

            y_batch = y_batch_df.iloc[:, 0]
            y_batch = self._normalise_class_labels(y_batch)

            if X_batch is not None:
                X_batch, y_batch = self._validate_X_y_indexes(X_batch, y_batch)
        
        device = ctx_cfg.get("device")
        if device is not None:
            device = str(device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Device {device!r} requested but CUDA is not available.")

        explainer_context = ExplainerContext(
            X_background=X_background,
            y_background=y_background,
            model=model,
            X_batch=X_batch,
            y_batch=y_batch,
            device=device
        )

        metadata = self._validate_explainer_context_metadata(ctx_cfg)

        return explainer_context, metadata
    
    
    def build_metric_contexts(self) -> List[Tuple[MetricContext, Dict[str, Any]]]:
        """
        Build all metric evaluation contexts defined by the configuration.

        The method first obtains all metric context configurations through
        :meth:`_iter_metric_context_configs` and then builds each corresponding
        :class:`MetricContext`.

        Returns
        -------
        List[Tuple[MetricContext, Dict[str, Any]]]
            List of metric contexts together with their metadata.
        """
        return [
            self.build_metric_context(ctx_cfg)
            for ctx_cfg in self._iter_metric_context_configs()
        ]
    

    def build_explainers_contexts(self) -> List[Tuple[ExplainerContext, Dict[str, Any]]]:
        """
        Build all explainer contexts defined by the configuration.

        The method first obtains all explainer context configurations through
        :meth:`_iter_explainer_context_configs` and then builds each
        corresponding :class:`ExplainerContext`.

        Returns
        -------
        List[Tuple[ExplainerContext, Dict[str, Any]]]
            List of explainer contexts together with their metadata.
        """
        return [
            self.build_explainers_context(ctx_cfg)
            for ctx_cfg in self._iter_explainer_context_configs()
        ]