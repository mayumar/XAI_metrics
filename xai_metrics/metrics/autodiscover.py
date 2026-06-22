# xai_metrics/metrics/autodiscover.py
import pkgutil
import importlib
from types import ModuleType

def autodiscover_metrics(package: ModuleType):
    """
    Import all submodules of a metrics package.

    This function recursively walks through ``package`` and imports every
    discovered module. Importing the modules triggers any registration logic
    executed at import time, such as ``@register_metric``.

    Parameters
    ----------
    package : ModuleType
        Imported package whose submodules should be discovered. The package
        must expose a ``__path__`` attribute.

    Raises
    ------
    AttributeError
        If ``package`` is not a package with a ``__path__`` attribute.
    ImportError
        If a discovered submodule cannot be imported.
    """
    prefix = package.__name__ + "."
    for m in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(m.name)