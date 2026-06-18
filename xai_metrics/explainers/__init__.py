# xai_metrics/explainers/__init__.py
from .lime import LIMEExplainer
from .autodiscover import autodiscover_explainers

__all__ = ['LIMEExplainer', 'autodiscover_explainers']