# xai_metrics/explainers/maple.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch import Tensor

from xai_metrics.base import register_explainer, BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping, Tuple, Callable


class _MAPLEModel:
    """
    Internal MAPLE surrogate model.

    This class builds the components required to compute MAPLE-style local
    explanations. A tree ensemble is first fitted on the background training
    data. For each explained observation, training points are weighted according
    to how often they share tree leaves with that observation. A weighted Ridge
    model is then fitted locally, and its coefficients are returned as feature
    attributions.

    The implementation follows the MAPLE idea introduced by Plumb, Molitor and
    Talwalkar (2018), using scikit-learn estimators and a local weighted linear
    model.
    """
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fe_type: str = "rf",
        n_estimators: int = 200,
        max_features: float | int | str | None = 0.5,
        min_samples_leaf: int = 10,
        regularization: float = 0.001,
        random_state: int | None = 42
    ):
        """
        Parameters
        ----------
        X_train : numpy.ndarray
            Background training data used to fit the tree ensemble and local
            linear models.
        y_train : numpy.ndarray
            Model responses associated with ``X_train``.
        X_val : numpy.ndarray
            Validation data used to select the subset of features retained by
            the local linear models.
        y_val : numpy.ndarray
            Model responses associated with ``X_val``.
        fe_type : str, optional
            Tree ensemble used to define local weights. Supported values are
            ``"rf"`` for random forests and ``"gbrt"`` for gradient boosting.
            The default value is ``"rf"``.
        n_estimators : int, optional
            Number of trees in the ensemble. The default value is ``200``.
        max_features : float, int, str or None, optional
            Maximum number of features considered by the ensemble. The default
            value is ``0.5``.
        min_samples_leaf : int, optional
            Minimum number of samples required in each leaf. The default value
            is ``10``.
        regularization : float, optional
            Ridge regularisation strength used by the local linear models. The
            default value is ``0.001``.
        random_state : int or None, optional
            Random state used by the ensemble and validation split. The default
            value is ``42``.

        Raises
        ------
        ValueError
            If ``fe_type`` is not ``"rf"`` or ``"gbrt"``.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_train = X_train.shape[0]
        self.num_features = X_train.shape[1]
        self.num_val = X_val.shape[0]
        self.regularization = regularization

        if fe_type == "rf":
            self.ensemble = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, # type: ignore
                random_state=random_state
            )
        elif fe_type == "gbrt":
            self.ensemble = GradientBoostingRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features, # type: ignore
                max_depth=None,
                random_state=random_state
            )
        else:
            raise ValueError(
                f"Unknown MAPLE forest ensemble type: {fe_type!r}. "
                "Expected 'rf' or 'gbrt'."
            )
        
        self.ensemble.fit(X_train, y_train)

        self.train_leaf_ids = np.asarray(self.ensemble.apply(X_train))

        if self.train_leaf_ids.ndim == 3:
            self.train_leaf_ids = self.train_leaf_ids[:, :, 0]

        if self.train_leaf_ids.ndim == 1:
            self.train_leaf_ids = self.train_leaf_ids.reshape(-1, 1)

        val_leaf_ids = np.asarray(self.ensemble.apply(X_val))

        if val_leaf_ids.ndim == 3:
            val_leaf_ids = val_leaf_ids[:, :, 0]

        if val_leaf_ids.ndim == 1:
            val_leaf_ids = val_leaf_ids.reshape(-1, 1)

        self.feature_scores = np.zeros(self.num_features, dtype=float)
        if fe_type == "rf":
            estimators = self.ensemble.estimators_
        else:
            estimators = self.ensemble.estimators_[:, 0] # type: ignore
        
        for estimator in estimators:
            root_feature = estimator.tree_.feature[0] # type: ignore

            if root_feature >= 0:
                self.feature_scores[root_feature] += estimator.tree_.impurity[0] # type: ignore
        feature_order = np.argsort(-self.feature_scores)

        best_rmse = np.inf
        best_features = np.arange(self.num_features)

        for retain in range(1, self.num_features + 1):
            selected = np.sort(feature_order[:retain])
            predictions = np.empty(X_val.shape[0], dtype=float)

            for index, row in enumerate(X_val):
                weights = self.training_point_weights(val_leaf_ids[index])

                local_model = Ridge(alpha=regularization)
                local_model.fit(X_train[:, selected], y_train, sample_weight=weights)

                predictions[index] = local_model.predict(row[selected].reshape(1, -1))[0]

            rmse = np.sqrt(np.mean((predictions - y_val) ** 2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_features = selected
        
        self.selected_features = best_features
        self.selected_X_train = X_train[:, best_features]

        
    def training_point_weights(self, instance_leaf_ids: np.ndarray) -> np.ndarray:
        """
        Compute MAPLE local weights for the training observations.

        Training observations receive higher weights when they share leaves
        with the explained instance across more trees.

        Parameters
        ----------
        instance_leaf_ids : numpy.ndarray
            Leaf identifiers of the explained instance for each tree in the
            ensemble.

        Returns
        -------
        numpy.ndarray
            One-dimensional array containing one local weight per training
            observation.
        """
        matches = self.train_leaf_ids == instance_leaf_ids
        leaf_sizes = matches.sum(axis=0)

        valid_trees = leaf_sizes > 0

        return (matches[:, valid_trees] / leaf_sizes[valid_trees]).sum(axis=1)
    

    def explain(self, x: np.ndarray) -> np.ndarray:
        """
        Explain one observation with a local weighted linear model.

        Parameters
        ----------
        x : numpy.ndarray
            Observation to explain.

        Returns
        -------
        numpy.ndarray
            Coefficient vector of the local Ridge model. Coefficients for
            features not selected during validation are set to zero.
        """
        x = np.asarray(x).reshape(1, -1)
        leaf_ids = np.asarray(self.ensemble.apply(x))

        if leaf_ids.ndim == 3:
            leaf_ids = leaf_ids[:, :, 0]

        if leaf_ids.ndim == 1:
            leaf_ids = leaf_ids.reshape(-1, 1)

        leaf_ids = leaf_ids[0]

        weights = self.training_point_weights(leaf_ids)

        local_model = Ridge(alpha=self.regularization)
        local_model.fit(self.selected_X_train, self.y_train, sample_weight=weights)

        coefficients = np.zeros(self.num_features, dtype=float)
        coefficients[self.selected_features] = np.asarray(local_model.coef_).reshape(-1)

        return coefficients



@register_explainer
class MAPLEExplainer(BaseExplainer):
    """
    MAPLE explainer.

    This explainer computes local feature attributions using a MAPLE-style
    surrogate model. Background data are split into training and validation
    subsets. The training subset is used to fit the tree ensemble and local
    linear models, while the validation subset is used to select the retained
    feature subset.

    For each observation, the explainer obtains the model response on the
    background data, builds or reuses a cached internal MAPLE model, and returns
    the coefficients of a local weighted Ridge model as attributions.

    The explainer is based on MAPLE by Plumb, Molitor and Talwalkar (2018),
    "Model Agnostic Supervised Local Explanations", and is registered under the
    name ``"MAPLE"``.
    """
    NAME = "MAPLE"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : ExplainerContext
            Shared explainer context. It must contain ``X_background``. The
            background data columns define the order of the returned attribution
            vectors.
        params : Mapping[str, Any] or None, optional
            Explainer-specific parameters. Supported keys are:

            - ``mode`` : str, optional
              Model task type. If ``"classification"``, the explainer uses
              ``predict_proba``. Otherwise, it uses ``predict``. The default
              value is ``"classification"``.

            - ``output_index`` : int, optional
              Output column selected from two-dimensional model predictions.
              The default value is ``1``.

            - ``validation_size`` : float, optional
              Fraction of background observations used for validation. The
              default value is ``0.2``.

            - ``data_type`` : str, optional
              Type of background data. Supported values are ``"tabular"`` and
              ``"time_series"``. For time series, the validation set is taken
              from the end of the background data. The default value is
              ``"tabular"``.

            - ``fe_type`` : str, optional
              Tree ensemble used by MAPLE. Supported values are ``"rf"`` and
              ``"gbrt"``. The default value is ``"rf"``.

            - ``n_estimators`` : int, optional
              Number of trees in the ensemble. The default value is ``200``.

            - ``max_features`` : float, int, str or None, optional
              Maximum number of features considered by the ensemble. The
              default value is ``0.5``.

            - ``min_samples_leaf`` : int, optional
              Minimum number of samples required in each leaf. The default
              value is ``10``.

            - ``regularization`` : float, optional
              Ridge regularisation strength used by the local linear models.
              The default value is ``0.001``.

            - ``random_state`` : int or None, optional
              Random state used by the ensemble and background split. The
              default value is ``42``.

            - ``abs`` : bool, optional
              Whether to return absolute attribution values. The default value
              is ``False``.

            If ``None``, an empty dictionary is used.

        Raises
        ------
        ValueError
            If fewer than three background observations are available.
        """
        super().__init__(context, params)
        
        self.cols = list(context.X_background.columns)
        background = self._to_numpy(context.X_background)

        if background.shape[0] < 3:
            raise ValueError(
                "MAPLEExplainer requires at least three background observations."
            )
        
        self.X_train, self.X_val = self._split_background(background)

        self._explainers = {}


    def _to_numpy(self, inputs: Any) -> np.ndarray:
        """
        Convert input data to a NumPy array.

        Parameters
        ----------
        inputs : Any
            Input observation or batch of observations. Supported inputs include
            pandas dataframes, torch tensors and array-like objects.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array with one row per observation and columns in
            the same order as ``context.X_background``.
        """
        if isinstance(inputs, pd.DataFrame):
            values = inputs.loc[:, self.cols].to_numpy()
        elif isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return values
    
    
    def _get_predict_fn(self, model: Any, predict_fn: Callable | None = None) -> Callable:
        """
        Resolve the prediction function used by MAPLE.

        If ``predict_fn`` is provided, it is returned directly. Otherwise, the
        method searches for ``predict_proba`` in classification mode and
        ``predict`` in any other mode. The method is searched first on the model
        itself and then on ``model.model``.

        Parameters
        ----------
        model : Any
            Model to explain.
        predict_fn : Callable or None, optional
            Custom prediction function. If provided, it is used directly.

        Returns
        -------
        Callable
            Prediction function used to obtain model responses.

        Raises
        ------
        AttributeError
            If the required prediction method is not available on ``model`` or
            ``model.model``.
        """
        if predict_fn is not None:
            return predict_fn

        mode = self.params.get("mode", "classification")

        if mode == "classification":
            method_name = "predict_proba"
        else:
            method_name = "predict"

        if hasattr(model, method_name):
            return getattr(model, method_name)

        if hasattr(model, "model") and hasattr(model.model, method_name):
            return getattr(model.model, method_name)

        raise AttributeError(
            f"No se encontro {method_name} en {type(model)} ni en model.model"
        )
    

    def _split_background(self, background: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split background data into training and validation subsets.

        Parameters
        ----------
        background : numpy.ndarray
            Background data used by MAPLE.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Training and validation subsets.

        Raises
        ------
        ValueError
            If ``validation_size`` is not between 0 and 1, if ``data_type`` is
            not supported, or if the split leaves no training observations.
        """
        validation_size = float(self.params.get("validation_size", 0.2))
        data_type = self.params.get("data_type", "tabular")

        if not 0 < validation_size < 1:
            raise ValueError(
                "validation_size must be between 0 and 1."
            )

        if data_type not in {"tabular", "time_series"}:
            raise ValueError(
                "data_type must be 'tabular' or 'time_series'."
            )

        if len(background) < 2:
            raise ValueError(
                "MAPLE requires at least two background observations."
            )
        
        if data_type == "time_series":
            validation_samples = max(1, int(np.ceil(len(background) * validation_size)))
            split_index = len(background) - validation_samples

            if split_index < 1:
                raise ValueError(
                    "validation_size leaves no observations for training."
                )
            
            X_train = background[:split_index]
            X_val = background[split_index:]

            return X_train, X_val
        
        X_train, X_val = train_test_split(
            background,
            test_size=validation_size,
            shuffle=True,
            random_state=self.params.get("random_state", 42)
        )
        
        return X_train, X_val
    

    def _model_response(
        self,
        model: Any,
        inputs: np.ndarray
    ) -> np.ndarray:
        """
        Compute the scalar model response used by MAPLE.

        For two-dimensional predictions, the column selected by
        ``output_index`` is used. The returned response is always flattened and
        converted to floating point values.

        Parameters
        ----------
        model : Any
            Model whose response is evaluated.
        inputs : numpy.ndarray
            Input data passed to the model prediction function.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of scalar model responses.

        Raises
        ------
        ValueError
            If ``output_index`` is incompatible with the model output shape.
        """
        prediction = np.asarray(self._get_predict_fn(model)(inputs))

        if prediction.ndim == 2:
            output_index = int(self.params.get("output_index", 1))

            if output_index >= prediction.shape[1]:
                raise ValueError(
                    f"MAPLE output_index={output_index} is invalid for "
                    f"model output shape {prediction.shape}."
                )
            
            prediction = prediction[:, output_index]

        return prediction.reshape(-1).astype(float)
    

    def _build_explainer(self, model: Any) -> _MAPLEModel:
        """
        Build an internal MAPLE model for the given model.

        Parameters
        ----------
        model : Any
            Model to explain.

        Returns
        -------
        _MAPLEModel
            Internal MAPLE model fitted on the model responses obtained from
            the background training and validation subsets.
        """
        y_train = self._model_response(model, self.X_train)
        y_val = self._model_response(model, self.X_val)

        return  _MAPLEModel(
            X_train=self.X_train,
            y_train=y_train,
            X_val=self.X_val,
            y_val=y_val,
            fe_type=self.params.get("fe_type", "rf"),
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_features=self.params.get("max_features", 0.5),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
            regularization=float(self.params.get("regularization", 0.001)),
            random_state=self.params.get("random_state", 42)
        )
    
    def _get_explainer(self, model: Any) -> _MAPLEModel:
        """
        Return a cached internal MAPLE model for the given model.

        A separate internal explainer is cached for each model object using the
        model identity. If no valid cached explainer exists, a new one is built.

        Parameters
        ----------
        model : Any
            Model to explain.

        Returns
        -------
        _MAPLEModel
            Cached or newly created internal MAPLE model.
        """
        key = id(model)
        cached = self._explainers.get(key)

        if cached is not None and cached[0] is model:
            return cached[1]

        explainer = self._build_explainer(model)
        self._explainers[key] = (model, explainer)

        return explainer
    

    def explain(
        self,
        model: Any,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Generate MAPLE attributions for the provided inputs.

        The method converts the inputs to a NumPy array, obtains a cached
        internal MAPLE model and computes one local coefficient vector per
        observation. These coefficient vectors are returned as feature
        attributions.

        Parameters
        ----------
        model : Any
            Model whose predictions are to be explained.
        inputs : Any
            Input observation or batch of observations to explain.
        targets : Any or None, optional
            Unused by this explainer. The explained output is controlled by the
            model prediction function and, for two-dimensional outputs, by
            ``output_index``.
        **kwargs : Any
            Additional keyword arguments. They are accepted for compatibility
            with the common explainer interface.

        Returns
        -------
        numpy.ndarray
            Attribution matrix with one row per explained observation and one
            column per feature. If ``abs=True``, absolute attribution values are
            returned.
        """
        X = self._to_numpy(inputs)

        explainer = self._get_explainer(model)

        attributions = np.asarray([
            explainer.explain(row)
            for row in X
        ])

        if self.params.get("abs", False):
            attributions = np.abs(attributions)

        return attributions