import importlib
import json
import sys

from glasses.runtime_alias import install_module_aliases, install_module_path_aliases, write_runtime_alias_bundle


def test_install_module_aliases_redirects_old_name(tmp_path, monkeypatch):
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "newmod.py").write_text("VALUE = 7\n")
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("oldmod", None)
    sys.modules.pop("newmod", None)

    install_module_aliases({"oldmod": "newmod"})

    imported = importlib.import_module("oldmod")

    assert imported.VALUE == 7
    assert sys.modules["oldmod"] is sys.modules["newmod"]


def test_write_runtime_alias_bundle_creates_sitecustomize_and_json(tmp_path):
    out_dir = tmp_path / "shim"

    write_runtime_alias_bundle(out_dir, {"django.core.validators": "django.apps.validators"})

    alias_json = json.loads((out_dir / "module_aliases.json").read_text())
    sitecustomize = (out_dir / "sitecustomize.py").read_text()

    assert alias_json == {"django.core.validators": "django.apps.validators"}
    assert "module_aliases.json" in sitecustomize
    assert "glasses.runtime_alias" not in sitecustomize


def test_install_module_path_aliases_loads_old_name_from_new_path(tmp_path, monkeypatch):
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    legacy_pkg = package_root / "legacy"
    legacy_pkg.mkdir()
    (legacy_pkg / "__init__.py").write_text("")
    real_path = package_root / "realmod.py"
    real_path.write_text("VALUE = 11\n")
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("legacy.mod", None)

    install_module_path_aliases({"legacy.mod": str(real_path)})

    imported = importlib.import_module("legacy.mod")

    assert imported.VALUE == 11
    assert imported.__name__ == "legacy.mod"
