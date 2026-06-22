# xai_metrics/explainers/autodiscover.py
import pkgutil
import importlib
from types import ModuleType

def autodiscover_explainers(package: ModuleType):
    """
    Import all modules inside an explainer package.

    The function walks recursively through the package and imports every module
    it finds. This triggers module-level registration decorators, such as
    ``@register_explainer``, making the discovered explainers available through
    the explainer registry.

    Parameters
    ----------
    package : ModuleType
        Package module to inspect. It must expose ``__name__`` and ``__path__``.
    """
    prefix = package.__name__ + "."
    for m in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(m.name)