# xai_metrics/base/explainer_registry.py
from xai_metrics.base import BaseExplainer, ExplainerContext

from typing import Type, List, Mapping, Any

EXPLAINER_REGISTRY = {}

def register_explainer(cls: Type[BaseExplainer]) -> Type[BaseExplainer]:
    """
    Register an explainer class.

    The class is registered using its ``NAME`` attribute. If the class does not
    define ``NAME``, its class name is used instead.

    Parameters
    ----------
    cls : Type[BaseExplainer]
        Explainer class to register.

    Returns
    -------
    Type[BaseExplainer]
        The registered class. This allows the function to be used as a
        decorator.

    Raises
    ------
    ValueError
        If another explainer has already been registered with the same name.
    """
    name = getattr(cls, "NAME", cls.__name__)

    if name in EXPLAINER_REGISTRY:
        raise ValueError(f"Duplicate explainer name registered: {name}")
    
    EXPLAINER_REGISTRY[name] = cls
    return cls


def list_explainers() -> List[str]:
    """
    Return the names of all registered explainers.

    Returns
    -------
    List[str]
        Sorted list with the names of the registered explainers.
    """
    return sorted(EXPLAINER_REGISTRY.keys())


def build_explainers_from_config(
    explainers_cfg: List[Mapping[str, Any]],
    context: ExplainerContext
) -> List[BaseExplainer]:
    """
    Build explainer instances from configuration entries.

    Each configuration entry must contain the explainer ``name`` and may
    optionally contain a ``params`` mapping. The name is used to retrieve the
    corresponding explainer class from the registry.

    Parameters
    ----------
    explainers_cfg : List[Mapping[str, Any]]
        List of explainer configuration entries.
    context : ExplainerContext
        Shared context passed to each explainer instance.

    Returns
    -------
    List[BaseExplainer]
        Instantiated explainers.

    Raises
    ------
    KeyError
        If an explainer configuration entry does not contain ``name``.
    ValueError
        If the requested explainer name is not registered.
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