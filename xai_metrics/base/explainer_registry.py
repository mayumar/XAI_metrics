# xai_metrics/base/explainer_registry.py
import pandas as pd

from xai_metrics.base import BaseExplainer, ExplainerContext

from typing import Type, List, Mapping, Any

EXPLAINER_REGISTRY = {}

def register_explainer(cls: Type[BaseExplainer]) -> Type[BaseExplainer]:
    name = getattr(cls, "NAME", cls.__name__)

    if name in EXPLAINER_REGISTRY:
        raise ValueError(f"Duplicate explainer name registered: {name}")
    
    EXPLAINER_REGISTRY[name] = cls
    return cls


def list_explainers() -> List[str]:
    return sorted(EXPLAINER_REGISTRY.keys())


def build_explainers_from_config(
    explainers_cfg: List[Mapping[str, Any]],
    context: ExplainerContext
) -> List[BaseExplainer]:
    
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