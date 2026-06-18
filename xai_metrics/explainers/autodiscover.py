# XAI_metrics/explainers/autodiscover.py
import pkgutil
import importlib
from types import ModuleType

def autodiscover_explainers(package: ModuleType) -> None:
    prefix = package.__name__ + "."
    for m in pkgutil.walk_packages(package.__path__, prefix):
        importlib.import_module(m.name)