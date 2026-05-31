# tests/test_autodiscover.py
import types

from xai_metrics.metrics import autodiscover_metrics

def test_autodiscover_metrics_imports_walked_modules(monkeypatch):
    def make_fake_package():
        package = types.ModuleType("fakepkg")
        package.__path__ = ["/fake"]
        return package

    package = make_fake_package()

    walked = [
        types.SimpleNamespace(name="fakepkg.alpha"),
        types.SimpleNamespace(name="fakepkg.beta")
    ]
    imported = []

    monkeypatch.setattr(
        "XAI_metrics.metrics.autodiscover.pkgutil.walk_packages",
        lambda path, prefix: walked
    )
    monkeypatch.setattr(
        "XAI_metrics.metrics.autodiscover.importlib.import_module",
        lambda name: imported.append(name),
    )

    autodiscover_metrics(package)

    assert imported == ["fakepkg.alpha", "fakepkg.beta"]