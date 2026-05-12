import ast
import builtins
import re
from pathlib import Path
from typing import Set, Dict, List
import concurrent.futures
import multiprocessing
from rich.progress import Progress

def _extract_identifiers_from_single_file(file_path: Path):
    # 提取仓库内所有的“语义符号”：类名、函数名、以及所有出现的名称（变量/属性等）
    ids = {"classes": set(), "functions": set(), "names": set()}
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
            # 新增：提取所有属性访问和变量名，建立全量词典
            elif isinstance(node, ast.Attribute):
                ids["names"].add(node.attr)
            elif isinstance(node, ast.Name):
                ids["names"].add(node.id)
            elif isinstance(node, ast.arg):
                ids["names"].add(node.arg)
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
            "vocabulary": set(), # 存储全量词汇
        }
        self._python_builtin_words = {name.lower() for name in dir(builtins)}
        self._external_import_roots: Set[str] = set()
        # 语言与系统基础设施白名单（绝不映射这些词）
        self.STOP_WORDS = {
            # Python 关键字
            'with', 'except', 'finally', 'yield', 'return', 'import', 'from', 'as', 'if', 'else', 'elif',
            'for', 'in', 'while', 'break', 'continue', 'class', 'def', 'try', 'raise', 'is', 'not', 'and', 'or',
            'None', 'True', 'False', 'async', 'await', 'pass', 'global', 'nonlocal', 'assert', 'lambda', 'del',
            # 内置函数
            'open', 'print', 'len', 'range', 'enumerate', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float',
            'bool', 'type', 'isinstance', 'issubclass', 'getattr', 'setattr', 'hasattr', 'dir', 'vars', 'super',
            'object', 'iter', 'next', 'map', 'filter', 'sorted', 'any', 'all', 'sum', 'min', 'max', 'abs', 'round',
            'zip', 'reversed', 'format', 'input', 'hash', 'id', 'property', 'staticmethod', 'classmethod',
            # 常见库/模块名
            'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'hashlib', 'pathlib', 'logging', 'argparse',
            'threading', 'multiprocessing', 'concurrent', 'subprocess', 'shutil', 'tempfile', 'io', 'base64',
            'collections', 'itertools', 'functools', 'operator', 'unittest', 'pytest', 'abc', 'typing',
            # Shell/System 关键指令与状态词
            'total', 'root', 'user', 'group', 'chmod', 'chown', 'bin', 'usr', 'etc', 'var', 'tmp', 'home',
            'drwxr', 'rwxr', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'command', 'found', 'error', 'warning', 'bash', 'shell', 'line', 'stdin', 'stdout', 'stderr',
            'grep', 'sed', 'awk', 'ls', 'cd', 'cat', 'rm', 'mkdir', 'cp', 'mv', 'git', 'diff', 'patch', 'index',
            # Python 基础设施与魔术文件
            '__init__.py', '__main__.py', '__init__', '__main__', 'setup.py', 'conftest.py', 'manage.py', 'tox.ini',
            # 常见的双下划线魔术方法
            '__str__', '__repr__', '__dict__', '__class__', '__module__', '__name__', '__file__', '__path__',
            '__getitem__', '__setitem__', '__iter__', '__next__', '__len__', '__call__', '__enter__', '__exit__',
            # Git Patch 关键字
            '---', '+++', '@@', 'diff', 'git', 'index', 'a', 'b',
            # 通用参数/属性/元数据
            'args', 'kwargs', 'self', 'cls', 'name', 'path', 'file', 'dir', 'data', 'text', 'content', 'value',
            'input', 'output', 'params', 'config', 'settings', 'options', 'results', 'exception',
            'true', 'false', 'none', 'null', 'status', 'code', 'type', 'mode', 'encoding', 'version',
            # 常见动词/逻辑词（防止长标识符被过度拆解映射）
            'contains', 'exists', 'has', 'is', 'get', 'set', 'update', 'delete', 'create', 'remove'
        }

    def tokenize_identifier(self, identifier: str) -> List[str]:
        """
        将标识符拆分为词根列表。
        改进：如果是 Python 魔术方法，不拆分，直接返回。
        """
        if identifier.startswith('__') and identifier.endswith('__'):
            return [identifier]
            
        name = identifier
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        
        # 使用正则表达式拆分标识符
        tokens = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', name)
        
        # 返回原始大小写的词根序列
        return [t for t in tokens if len(t) > 0]

    def extract(self, max_workers: int = None):
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        ignored_dirs = {".git", "__pycache__", "migrations", "tests", "testing"}
        py_files = []

        # 先提取仓库内“库名”（可 import 的顶层 package）：repo_root/<pkg>/__init__.py
        # 这与 directories 不同：directories 是全量目录名；libraries 只记录顶层包名，便于更稳定地做语义映射。
        for child in self.repo_path.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name in ignored_dirs:
                continue
            if (child / "__init__.py").exists():
                self.internal_identifiers["libraries"].add(child.name)

        for py_file in self.repo_path.rglob("*.py"):
            parts = py_file.relative_to(self.repo_path).parts
            if any(part.startswith('.') or part.startswith('_') for part in parts[:-1]) or \
               any(part in ignored_dirs for part in parts[:-1]):
                continue
            if py_file.stem == "__main__":
                continue

            module_parts = list(parts[:-1])
            if py_file.stem != "__init__":
                if py_file.stem.startswith('_'):
                    continue
                module_parts.append(py_file.stem)

            if module_parts:
                self.internal_identifiers["modules"].add(".".join(module_parts))

        self._external_import_roots = self._scan_external_import_roots(ignored_dirs)

        for path in self.repo_path.rglob("*"):
            if any(part.startswith('.') or part.startswith('_') for part in path.parts) or \
               any(part in ignored_dirs for part in path.parts):
                continue
            
            if path.is_dir():
                if path != self.repo_path:
                    self.internal_identifiers["directories"].add(path.name)
            elif path.suffix == ".py":
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
            self.internal_identifiers["vocabulary"].update(n for n in res["names"] if n.lower() not in reserved_tokens)

    def get_identifiers(self) -> Dict[str, List[str]]:
        return {k: sorted(list(v)) for k, v in self.internal_identifiers.items()}

    def get_reserved_tokens(self) -> Set[str]:
        return self.STOP_WORDS | self._python_builtin_words | self._external_import_roots

    def get_level2_namespace_targets(self) -> Dict[str, List[str]]:
        ids = self.get_identifiers()
        reserved_tokens = self.get_reserved_tokens()
        return {
            "classes": [name for name in ids["classes"] if name.lower() not in reserved_tokens],
            "functions": [name for name in ids["functions"] if name.lower() not in reserved_tokens],
            "modules": [name for name in ids["modules"] if name.lower() not in reserved_tokens],
            "files": sorted(name for name in self.internal_identifiers["files"] if name.lower() not in reserved_tokens),
            "directories": sorted(
                name for name in self.internal_identifiers["directories"] if name.lower() not in reserved_tokens
            ),
            "names": [],
        }

    def _scan_external_import_roots(self, ignored_dirs: Set[str]) -> Set[str]:
        local_roots = {name.lower() for name in self.internal_identifiers["libraries"]}
        external_roots = set()

        for py_file in self.repo_path.rglob("*.py"):
            parts = py_file.relative_to(self.repo_path).parts
            if any(part.startswith(".") or part.startswith("_") for part in parts[:-1]) or any(
                part in ignored_dirs for part in parts[:-1]
            ):
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

    def _scan_cross_file_imports(self) -> Set[str]:
        """AST-scan all .py files within self.repo_path for import targets.
        Returns set of identifiers that are imported by at least one other file."""
        imported = set()
        for py_file in self.repo_path.rglob("*.py"):
            parts = py_file.relative_to(self.repo_path).parts
            if any(p.startswith('.') or p.startswith('_') or p in {"migrations", "tests", "testing"} for p in parts[:-1]):
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
            parts = py_file.relative_to(self.repo_path).parts
            if any(p.startswith('.') or p.startswith('_') or p in {"migrations", "tests", "testing"} for p in parts[:-1]):
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
