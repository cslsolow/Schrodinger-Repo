"""Tests for safe identifier filtering in extractor."""
import ast
import tempfile
from pathlib import Path

import pytest

from glasses.extractor import RepoIdentifierExtractor


@pytest.fixture
def sample_repo(tmp_path):
    """Create a minimal Python repo for testing."""
    # Main package
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from mypackage.utils import helper_function\n")

    # utils.py: defines helper_function (imported by __init__.py) and internal_calc (not imported)
    (pkg / "utils.py").write_text(
        "import requests\n"
        "from numpy import ndarray\n"
        "\n"
        "def helper_function(data):\n"
        "    return internal_calc(data)\n"
        "\n"
        "def internal_calc(data):\n"
        "    return data * 2\n"
        "\n"
        "__all__ = ['helper_function']\n"
        "\n"
        "class PublicModel:\n"
        "    db_table = 'my_table'\n"
        "\n"
        "class InternalProcessor:\n"
        "    pass\n"
        "\n"
        "pk = 1\n"
        "idx = 2\n"
    )

    # models.py: standalone file, nothing imports from it
    (pkg / "models.py").write_text(
        "class StandaloneWidget:\n"
        "    def compute_result(self):\n"
        "        return 42\n"
    )

    nested_pkg = pkg / "subpkg"
    nested_pkg.mkdir()
    (nested_pkg / "__init__.py").write_text("")
    (nested_pkg / "worker.py").write_text(
        "def nested_job():\n"
        "    return 1\n"
    )

    return tmp_path


def test_safe_identifiers_exclude_cross_file_imports(sample_repo):
    """Identifiers imported by other files should be excluded."""
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    safe = extractor.get_safe_identifiers()

    all_safe_ids = set()
    for ids in safe.values():
        all_safe_ids.update(ids)

    # helper_function is imported by __init__.py, should NOT be safe
    assert "helper_function" not in all_safe_ids


def test_safe_identifiers_exclude_exported(sample_repo):
    """Identifiers in __all__ should be excluded."""
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    safe = extractor.get_safe_identifiers()

    all_safe_ids = set()
    for ids in safe.values():
        all_safe_ids.update(ids)

    assert "helper_function" not in all_safe_ids


def test_safe_identifiers_exclude_short(sample_repo):
    """Identifiers shorter than 4 characters should be excluded."""
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    safe = extractor.get_safe_identifiers()

    all_safe_ids = set()
    for ids in safe.values():
        all_safe_ids.update(ids)

    assert "pk" not in all_safe_ids
    assert "idx" not in all_safe_ids


def test_safe_identifiers_exclude_file_stems(sample_repo):
    """File name stems should be excluded."""
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    safe = extractor.get_safe_identifiers()

    all_safe_ids = set()
    for ids in safe.values():
        all_safe_ids.update(ids)

    # 'utils' and 'models' are .py file stems
    assert "utils" not in all_safe_ids
    assert "models" not in all_safe_ids


def test_safe_identifiers_include_internal(sample_repo):
    """Purely internal identifiers should be included."""
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    safe = extractor.get_safe_identifiers()

    all_safe_ids = set()
    for ids in safe.values():
        all_safe_ids.update(ids)

    # internal_calc: not imported, not in __all__, len >= 4
    assert "internal_calc" in all_safe_ids
    # InternalProcessor: not imported, not in __all__, len >= 4
    assert "InternalProcessor" in all_safe_ids
    # compute_result: not imported, not in __all__, len >= 4
    assert "compute_result" in all_safe_ids


def test_level2_namespace_targets_limit_scope(sample_repo):
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)

    targets = extractor.get_level2_namespace_targets()

    assert "InternalProcessor" in targets["classes"]
    assert "helper_function" in targets["functions"]
    assert "internal_calc" in targets["functions"]
    assert "mypackage.utils" in targets["modules"]
    assert "mypackage.models" in targets["modules"]
    assert "mypackage.subpkg" in targets["modules"]
    assert "mypackage.subpkg.worker" in targets["modules"]
    assert "utils.py" in targets["files"]
    assert "mypackage" in targets["directories"]
    assert "data" not in targets["functions"]
    assert "db_table" not in targets["names"]


def test_reserved_tokens_include_python_builtins_and_external_roots(sample_repo):
    extractor = RepoIdentifierExtractor(sample_repo)
    extractor.extract(max_workers=1)
    reserved = extractor.get_reserved_tokens()

    assert "list" in reserved
    assert "len" in reserved
    assert "requests" in reserved
    assert "numpy" in reserved
