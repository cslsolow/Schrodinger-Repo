import importlib
import json
import importlib.abc
import importlib.util
import sys
from pathlib import Path


class ModuleAliasLoader(importlib.abc.Loader):
    def __init__(self, alias: str, real_name: str):
        self.alias = alias
        self.real_name = real_name

    def create_module(self, spec):
        module = importlib.import_module(self.real_name)
        sys.modules[self.alias] = module
        return module

    def exec_module(self, module):
        return None


class ModuleAliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, aliases: dict[str, str]):
        self.aliases = aliases

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.aliases:
            return None
        real_name = self.aliases[fullname]
        loader = ModuleAliasLoader(fullname, real_name)
        return importlib.util.spec_from_loader(fullname, loader)


class ModulePathAliasLoader(importlib.abc.Loader):
    def __init__(self, alias: str, real_path: str):
        self.alias = alias
        self.real_path = real_path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        with open(self.real_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        module.__file__ = self.real_path
        module.__package__ = self.alias.rpartition(".")[0]
        exec(compile(source, self.real_path, "exec"), module.__dict__)


class ModulePathAliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, path_aliases: dict[str, str]):
        self.path_aliases = path_aliases

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.path_aliases:
            return None
        loader = ModulePathAliasLoader(fullname, self.path_aliases[fullname])
        return importlib.util.spec_from_loader(fullname, loader, origin=self.path_aliases[fullname])


def install_module_aliases(aliases: dict[str, str]) -> ModuleAliasFinder:
    for finder in list(sys.meta_path):
        if isinstance(finder, ModuleAliasFinder):
            sys.meta_path.remove(finder)
    finder = ModuleAliasFinder(aliases)
    sys.meta_path.insert(0, finder)
    return finder


def install_module_path_aliases(path_aliases: dict[str, str]) -> ModulePathAliasFinder:
    for finder in list(sys.meta_path):
        if isinstance(finder, ModulePathAliasFinder):
            sys.meta_path.remove(finder)
    finder = ModulePathAliasFinder(path_aliases)
    sys.meta_path.insert(0, finder)
    return finder


def write_runtime_alias_bundle(out_dir: Path, aliases: dict[str, str], *, path_aliases: dict[str, str] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "module_aliases.json").write_text(json.dumps(aliases, indent=2, sort_keys=True))
    (out_dir / "module_path_aliases.json").write_text(json.dumps(path_aliases or {}, indent=2, sort_keys=True))
    (out_dir / "sitecustomize.py").write_text(
        "import json\n"
        "import importlib\n"
        "import importlib.abc\n"
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "class _AliasLoader(importlib.abc.Loader):\n"
        "    def __init__(self, alias, real_name):\n"
        "        self.alias = alias\n"
        "        self.real_name = real_name\n"
        "    def create_module(self, spec):\n"
        "        module = importlib.import_module(self.real_name)\n"
        "        sys.modules[self.alias] = module\n"
        "        return module\n"
        "    def exec_module(self, module):\n"
        "        return None\n"
        "\n"
        "class _AliasFinder(importlib.abc.MetaPathFinder):\n"
        "    def __init__(self, aliases):\n"
        "        self.aliases = aliases\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname not in self.aliases:\n"
        "            return None\n"
        "        return importlib.util.spec_from_loader(fullname, _AliasLoader(fullname, self.aliases[fullname]))\n"
        "\n"
        "class _PathAliasLoader(importlib.abc.Loader):\n"
        "    def __init__(self, alias, real_path):\n"
        "        self.alias = alias\n"
        "        self.real_path = real_path\n"
        "    def create_module(self, spec):\n"
        "        return None\n"
        "    def exec_module(self, module):\n"
        "        with open(self.real_path, 'r', encoding='utf-8') as handle:\n"
        "            source = handle.read()\n"
        "        module.__file__ = self.real_path\n"
        "        module.__package__ = self.alias.rpartition('.')[0]\n"
        "        exec(compile(source, self.real_path, 'exec'), module.__dict__)\n"
        "\n"
        "class _PathAliasFinder(importlib.abc.MetaPathFinder):\n"
        "    def __init__(self, aliases):\n"
        "        self.aliases = aliases\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname not in self.aliases:\n"
        "            return None\n"
        "        return importlib.util.spec_from_loader(fullname, _PathAliasLoader(fullname, self.aliases[fullname]), origin=self.aliases[fullname])\n"
        "\n"
        "def _install_aliases(aliases, path_aliases):\n"
        "    sys.meta_path[:] = [f for f in sys.meta_path if f.__class__.__name__ not in ('_AliasFinder', '_PathAliasFinder')]\n"
        "    if aliases:\n"
        "        sys.meta_path.insert(0, _AliasFinder(aliases))\n"
        "    if path_aliases:\n"
        "        sys.meta_path.insert(0, _PathAliasFinder(path_aliases))\n"
        "\n"
        "alias_path = Path(__file__).with_name('module_aliases.json')\n"
        "path_alias_path = Path(__file__).with_name('module_path_aliases.json')\n"
        "_install_aliases(json.loads(alias_path.read_text()), json.loads(path_alias_path.read_text()))\n"
    )
    return out_dir
