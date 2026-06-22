# xai_metrics/base/explainer_registry.py
from xai_metrics.base import BaseExplainer, ExplainerContext

from typing import Type, List, Mapping, Any

EXPLAINER_REGISTRY = {}

def register_explainer(cls: Type[BaseExplainer]) -> Type[BaseExplainer]:
    """
    Register an explainer class in the global explainer registry.

    The explainer is registered using its ``NAME`` attribute. If the class does
    not define ``NAME``, the class name is used instead.

    Parameters
    ----------
    cls : Type[BaseExplainer]
        Explainer class to register.

    Returns
    -------
    Type[BaseExplainer]
        The same explainer class passed as input. This allows the function to
        be used as a decorator.

    Raises
    ------
    ValueError
        If another explainer with the same name is already registered.

    Examples
    --------
    >>> @register_explainer
    ... class MyExplainer(BaseExplainer):
    ...     NAME = "my_explainer"
    ...     def explain(self, model, inputs, targets=None, **kwargs):
    ...         return inputs
    """
    name = getattr(cls, "NAME", cls.__name__)

    if name in EXPLAINER_REGISTRY:
        raise ValueError(f"Duplicate explainer name registered: {name}")
    
    EXPLAINER_REGISTRY[name] = cls
    return cls


def list_explainers() -> List[str]:
    """
    Return the names of the registered explainers.

    Returns
    -------
    List[str]
        Sorted list containing the names of all explainers currently registered
        in ``EXPLAINER_REGISTRY``.
    """
    return sorted(EXPLAINER_REGISTRY.keys())


def build_explainers_from_config(
    explainers_cfg: List[Mapping[str, Any]],
    context: ExplainerContext
) -> List[BaseExplainer]:
    """
    Build explainer instances from a configuration list.

    Each explainer configuration must contain a ``name`` field matching a
    registered explainer. It may also contain a ``params`` field with
    explainer-specific parameters.

    Parameters
    ----------
    explainers_cfg : List[Mapping[str, Any]]
        List of explainer configuration dictionaries. Each dictionary must
        contain the explainer ``name`` and may contain ``params``.
    context : ExplainerContext
        Shared explainer context passed to every explainer instance.

    Returns
    -------
    List[BaseExplainer]
        List of instantiated explainer objects.

    Raises
    ------
    ValueError
        If an explainer name from ``explainers_cfg`` is not registered.
    KeyError
        If an explainer configuration does not contain the required ``name``
        field.
    """
    explainers = []
        
    for explainer_cfg in explainers_cfg:
        name = explainer_cfg['name']
        params = explainer_cfg.get("params")

        if name not in EXPLAINER_REGISTRY:
            raise ValueError(
                f"Explainer '{name}' not registered. "
                f"Available: {list(EXPLAINER_REGISTRY.keys())}"
            )
        
        explainer_cls = EXPLAINER_REGISTRY[name]

        explainer = explainer_cls(
            context=context,
            params=params
        )
        explainers.append(explainer)

    return explainers