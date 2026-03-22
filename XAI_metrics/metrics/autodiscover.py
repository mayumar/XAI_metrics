# XAI_metrics/metrics/autodiscover.py
import pkgutil
import importlib
from types import ModuleType
from typing import Iterable

def autodiscover_metrics(
    package: ModuleType,
    only_modules: Iterable[str] | None = None,
    exclude_modules: Iterable[str] | None = None
) -> None:
    """
    Importa todos los submódulos de XAI_metrics.metrics para disparar @register_metric.
    """

    only_set = set(only_modules or [])
    exclude_set = set(exclude_modules or [])

    prefix = package.__name__ + "."
    for m in pkgutil.iter_modules(package.__path__, prefix):
        short_name = m.name[len(prefix):]

        if only_set and short_name not in only_set and m.name not in only_set:
            continue

        if short_name in exclude_set or m.name in exclude_set:
            continue

        importlib.import_module(m.name)