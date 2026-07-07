import ast
import builtins
import keyword
import re
from pathlib import Path
from typing import Set, Dict, List
import concurrent.futures
import multiprocessing
from rich.progress import Progress


IGNORED_DIRS = {".git", "__pycache__", "migrations", "tests", "testing"}


def _is_ignored_python_path(py_file: Path, repo_path: Path, ignored_dirs: Set[str] = IGNORED_DIRS) -> bool:
    parts = py_file.relative_to(repo_path).parts
    return (
        any(part.startswith(".") or part.startswith("_") for part in parts[:-1])
        or any(part in ignored_dirs for part in parts[:-1])
        or py_file.stem == "__main__"
        or (py_file.stem.startswith("_") and py_file.stem != "__init__")
    )


def _assignment_target_names(target) -> set[str]:
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_assignment_target_names(elt))
    return names


def _is_valid_identifier_name(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def _extract_identifiers_from_single_file(file_path: Path):
    ids = {
        "classes": set(),
        "functions": set(),
        "module_variables": set(),
        "class_attributes": set(),
        "import_references": set(),
        "names": set(),
    }
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    ids["classes"].add(node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith('_'):
                    ids["functions"].add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    ids["import_references"].add(alias.name)
                    if alias.asname:
                        ids["import_references"].add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    ids["import_references"].add(node.module)
                for alias in node.names:
                    if alias.name != "*":
                        ids["import_references"].add(alias.name)
                    if alias.asname:
                        ids["import_references"].add(alias.asname)
            elif isinstance(node, ast.Attribute):
                ids["names"].add(node.attr)
            elif isinstance(node, ast.Name):
                ids["names"].add(node.id)
            elif isinstance(node, ast.arg):
                ids["names"].add(node.arg)

        for stmt in tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    ids["module_variables"].update(_assignment_target_names(target))
            elif isinstance(stmt, ast.ClassDef):
                for class_stmt in stmt.body:
                    if isinstance(class_stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = class_stmt.targets if isinstance(class_stmt, ast.Assign) else [class_stmt.target]
                        for target in targets:
                            ids["class_attributes"].update(_assignment_target_names(target))
    except Exception:
        pass
    return ids

class RepoIdentifierExtractor:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()
        self.internal_identifiers: Dict[str, Set[str]] = {
            "files": set(),
            "directories": set(),
            "libraries": set(),
            "modules": set(),
            "classes": set(),
            "functions": set(),
            "module_variables": set(),
            "class_attributes": set(),
            "import_references": set(),
            "vocabulary": set(),
        }
        self._python_builtin_words = {name.lower() for name in dir(builtins)}
        self._python_keywords = {name.lower() for name in keyword.kwlist}
        self._external_import_roots: Set[str] = set()
        self._external_import_names: Set[str] = set()
        self.STOP_WORDS = {
            'with', 'except', 'finally', 'yield', 'return', 'import', 'from', 'as', 'if', 'else', 'elif',
            'for', 'in', 'while', 'break', 'continue', 'class', 'def', 'try', 'raise', 'is', 'not', 'and', 'or',
            'None', 'True', 'False', 'async', 'await', 'pass', 'global', 'nonlocal', 'assert', 'lambda', 'del',
            'open', 'print', 'len', 'range', 'enumerate', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float',
            'bool', 'type', 'isinstance', 'issubclass', 'getattr', 'setattr', 'hasattr', 'dir', 'vars', 'super',
            'object', 'iter', 'next', 'map', 'filter', 'sorted', 'any', 'all', 'sum', 'min', 'max', 'abs', 'round',
            'zip', 'reversed', 'format', 'input', 'hash', 'id', 'property', 'staticmethod', 'classmethod',
            'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'hashlib', 'pathlib', 'logging', 'argparse',
            'threading', 'multiprocessing', 'concurrent', 'subprocess', 'shutil', 'tempfile', 'io', 'base64',
            'collections', 'itertools', 'functools', 'operator', 'unittest', 'pytest', 'abc', 'typing',
            'total', 'root', 'user', 'group', 'chmod', 'chown', 'bin', 'usr', 'etc', 'var', 'tmp', 'home',
            'drwxr', 'rwxr', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'command', 'found', 'error', 'warning', 'bash', 'shell', 'line', 'stdin', 'stdout', 'stderr',
            'grep', 'sed', 'awk', 'ls', 'cd', 'cat', 'rm', 'mkdir', 'cp', 'mv', 'git', 'diff', 'patch', 'index',
            '__init__.py', '__main__.py', '__init__', '__main__', 'setup.py', 'conftest.py', 'manage.py', 'tox.ini',
            '__str__', '__repr__', '__dict__', '__class__', '__module__', '__name__', '__file__', '__path__',
            '__getitem__', '__setitem__', '__iter__', '__next__', '__len__', '__call__', '__enter__', '__exit__',
            '---', '+++', '@@', 'diff', 'git', 'index', 'a', 'b',
            'args', 'kwargs', 'self', 'cls', 'name', 'path', 'file', 'dir', 'data', 'text', 'content', 'value',
            'input', 'output', 'params', 'config', 'settings', 'options', 'results', 'exception',
            'true', 'false', 'none', 'null', 'status', 'code', 'type', 'mode', 'encoding', 'version',
            'contains', 'exists', 'has', 'is', 'get', 'set', 'update', 'delete', 'create', 'remove'
        }

    def tokenize_identifier(self, identifier: str) -> List[str]:
        if identifier.startswith('__') and identifier.endswith('__'):
            return [identifier]
            
        name = identifier
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        
        tokens = []
        for segment in re.split(r"[_\-.]+", name):
            tokens.extend(re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', segment))
        
        return [t for t in tokens if len(t) > 0]

    def extract(self, max_workers: int = None):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        py_files = []

        self.internal_identifiers["libraries"].update(self._infer_importable_repo_roots())

        for py_file in self.repo_path.rglob("*.py"):
            if _is_ignored_python_path(py_file, self.repo_path):
                continue

            parts = py_file.relative_to(self.repo_path).parts
            module_parts = list(parts[:-1])
            if py_file.stem != "__init__":
                if py_file.stem.startswith('_'):
                    continue
                module_parts.append(py_file.stem)

            if module_parts:
                self.internal_identifiers["modules"].add(".".join(module_parts))

        self._external_import_roots = self._scan_external_import_roots(IGNORED_DIRS)
        self._external_import_names = self._scan_external_import_names(IGNORED_DIRS)

        for path in self.repo_path.rglob("*"):
            relative_parts = path.relative_to(self.repo_path).parts
            checked_parts = relative_parts if path.is_dir() else relative_parts[:-1]
            if any(part.startswith('.') or part.startswith('_') for part in checked_parts) or \
               any(part in IGNORED_DIRS for part in checked_parts):
                continue
            
            if path.is_dir():
                if path != self.repo_path:
                    self.internal_identifiers["directories"].add(path.name)
            elif path.suffix == ".py":
                if _is_ignored_python_path(path, self.repo_path):
                    continue
                self.internal_identifiers["files"].add(path.name)
                py_files.append(path)

        results = []
        with Progress() as progress:
            task = progress.add_task("[yellow]Extracting identifiers...", total=len(py_files))
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_extract_identifiers_from_single_file, f) for f in py_files]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    progress.advance(task)

        reserved_tokens = self.get_reserved_tokens()
        for res in results:
            self.internal_identifiers["classes"].update(c for c in res["classes"] if c.lower() not in reserved_tokens)
            self.internal_identifiers["functions"].update(f for f in res["functions"] if f.lower() not in reserved_tokens)
            self.internal_identifiers["module_variables"].update(
                n for n in res["module_variables"] if n.lower() not in reserved_tokens
            )
            self.internal_identifiers["class_attributes"].update(
                n for n in res["class_attributes"] if n.lower() not in reserved_tokens
            )
            self.internal_identifiers["import_references"].update(
                n for n in res["import_references"] if n.lower() not in reserved_tokens
            )
            self.internal_identifiers["vocabulary"].update(n for n in res["names"] if n.lower() not in reserved_tokens)

    def get_identifiers(self) -> Dict[str, List[str]]:
        return {k: sorted(list(v)) for k, v in self.internal_identifiers.items()}

    def get_reserved_tokens(self) -> Set[str]:
        return (
            self.STOP_WORDS
            | self._python_builtin_words
            | self._python_keywords
            | self._external_import_roots
            | self._external_import_names
        )

    def get_level2_namespace_targets(self) -> Dict[str, List[str]]:
        ids = self.get_identifiers()
        reserved_tokens = self.get_reserved_tokens()
        names = self._filter_level2_names(
            set(ids["module_variables"]) | set(ids["class_attributes"]) | self._repo_internal_import_reference_names()
        )
        return {
            "classes": [name for name in ids["classes"] if name.lower() not in reserved_tokens],
            "functions": [name for name in ids["functions"] if name.lower() not in reserved_tokens],
            "modules": [name for name in ids["modules"] if name.lower() not in reserved_tokens],
            "files": sorted(name for name in self.internal_identifiers["files"] if name.lower() not in reserved_tokens),
            "directories": sorted(
                name for name in self.internal_identifiers["directories"] if name.lower() not in reserved_tokens
            ),
            "names": sorted(names),
        }

    def _scan_external_import_roots(self, ignored_dirs: Set[str]) -> Set[str]:
        local_roots = {name.lower() for name in self.internal_identifiers["libraries"]}
        external_roots = set()

        for py_file in self.repo_path.rglob("*.py"):
            if _is_ignored_python_path(py_file, self.repo_path, ignored_dirs):
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", 1)[0].lower()
                        if root and root not in local_roots:
                            external_roots.add(root)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    module = (node.module or "").strip()
                    if not module:
                        continue
                    root = module.split(".", 1)[0].lower()
                    if root and root not in local_roots:
                        external_roots.add(root)

        return external_roots

    def _scan_external_import_names(self, ignored_dirs: Set[str]) -> Set[str]:
        local_roots = {name.lower() for name in self.internal_identifiers["libraries"]}
        external_names = set()

        for py_file in self.repo_path.rglob("*.py"):
            if _is_ignored_python_path(py_file, self.repo_path, ignored_dirs):
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", 1)[0].lower()
                        if root and root not in local_roots:
                            external_names.add(root)
                            if alias.asname:
                                external_names.add(alias.asname)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue
                    module = (node.module or "").strip()
                    if not module:
                        continue
                    root = module.split(".", 1)[0].lower()
                    if root and root not in local_roots:
                        for alias in node.names:
                            if alias.name != "*":
                                external_names.add(alias.name)
                            if alias.asname:
                                external_names.add(alias.asname)

        return {name.lower() for name in external_names}

    def _scan_cross_file_imports(self) -> Set[str]:
        """AST-scan all .py files within self.repo_path for import targets.
        Returns set of identifiers that are imported by at least one other file."""
        imported = set()
        for py_file in self.repo_path.rglob("*.py"):
            if _is_ignored_python_path(py_file, self.repo_path):
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in (node.names or []):
                        name = alias.name
                        if name != '*':
                            imported.add(name)
                elif isinstance(node, ast.Import):
                    for alias in (node.names or []):
                        # 'import foo.bar' — extract 'foo' and 'bar'
                        for part in alias.name.split('.'):
                            imported.add(part)
        return imported

    def _scan_all_exports(self) -> Set[str]:
        """Find identifiers listed in __all__ across the repo."""
        exported = set()
        for py_file in self.repo_path.rglob("*.py"):
            if _is_ignored_python_path(py_file, self.repo_path):
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        exported.add(elt.value)
        return exported

    def _get_file_stem_identifiers(self) -> Set[str]:
        """Collect all .py file stems and directory names as identifiers to exclude."""
        stems = set()
        for py_file in self.repo_path.rglob("*.py"):
            stem = py_file.stem
            if stem not in ('__init__', '__main__'):
                stems.add(stem)
        for d in self.repo_path.rglob("*"):
            if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_'):
                stems.add(d.name)
        return stems

    def _infer_importable_repo_roots(self) -> Set[str]:
        roots = set()
        for search_root in [self.repo_path, self.repo_path / "src"]:
            if not search_root.is_dir():
                continue
            for child in search_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name.startswith(".") or child.name in IGNORED_DIRS:
                    continue
                if (child / "__init__.py").exists():
                    roots.add(child.name)
        return roots

    def _repo_internal_import_reference_names(self) -> Set[str]:
        local_roots = {name.lower() for name in self.internal_identifiers["libraries"]}
        names = set()

        for reference in self.internal_identifiers["import_references"]:
            if not reference:
                continue
            parts = reference.split(".")
            root = parts[0].lower()
            if root in self._external_import_roots:
                continue
            if root in local_roots:
                names.update(part for part in parts if part)
                continue
            if len(parts) == 1:
                names.add(reference)
        return names

    def _filter_level2_names(self, names: Set[str]) -> Set[str]:
        reserved_tokens = self.get_reserved_tokens()
        filtered = set()
        for name in names:
            if not _is_valid_identifier_name(name):
                continue
            if name.startswith("__") and name.endswith("__"):
                continue
            if name.lower() in reserved_tokens:
                continue
            if len(name) < 4 and not name.isupper():
                continue
            filtered.add(name)
        return filtered

    def get_safe_identifiers(self) -> Dict[str, List[str]]:
        """Return only identifiers that are safe to map.

        Safe means:
        - Length >= 4 characters
        - Not in STOP_WORDS
        - Not imported by other files in the repo
        - Not listed in __all__
        - Not a file/directory name
        """
        all_ids = self.get_identifiers()
        imported_ids = self._scan_cross_file_imports()
        exported_ids = self._scan_all_exports()
        file_stem_ids = self._get_file_stem_identifiers()

        unsafe = imported_ids | exported_ids | file_stem_ids
        reserved_tokens = self.get_reserved_tokens()

        safe = {}
        for category, ids in all_ids.items():
            safe[category] = [
                oid for oid in ids
                if len(oid) >= 4
                and oid.lower() not in reserved_tokens
                and oid not in unsafe
            ]
        return safe
