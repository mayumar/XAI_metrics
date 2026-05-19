# XAI_metrics/metrics/autodiscover.py
import pkgutil
import importlib
from types import ModuleType

def autodiscover_metrics(package: ModuleType) -> None:
    """
    Importa todos los submódulos de XAI_metrics.metrics para disparar @register_metric.
    """
    prefix = package.__name__ + "."
    for m in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(m.name)