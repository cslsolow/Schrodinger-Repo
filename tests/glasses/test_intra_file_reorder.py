import ast
import json

from glasses.intra_file_reorder import (
    apply_intra_file_reorder,
    build_intra_file_reorder_plan,
    restore_original_file_order,
    select_intra_file_variant,
)


def _top_level_order(source: str) -> list[str]:
    tree = ast.parse(source)
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _class_method_order(source: str, class_name: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return []


def test_build_intra_file_reorder_plan_generates_five_unique_variants(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "def first():\n    return 1\n\n"
        "def helper():\n    return 0\n\n"
        "class Sample:\n"
        "    def alpha(self):\n        return 'a'\n\n"
        "    def beta(self):\n        return 'b'\n\n"
        "    def gamma(self):\n        return 'c'\n\n"
        "def second():\n    return 2\n\n"
        "def third():\n    return 3\n"
    )
    path = repo_root / "django/core/validators.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/validators.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)

    assert plan["status"] == "ok"
    assert len(plan["variants"]) == 5
    assert len({json.dumps(item["file_orders"], sort_keys=True) for item in plan["variants"]}) == 5


def test_apply_intra_file_reorder_changes_order_by_seed(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n\n"
        "class Sample:\n"
        "    def alpha(self):\n        return 'a'\n\n"
        "    def beta(self):\n        return 'b'\n\n"
        "    def gamma(self):\n        return 'c'\n\n"
        "def third():\n    return 3\n"
    )
    path = repo_root / "django/core/validators.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/validators.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    seed_a = seed_b = None
    seen = {}
    for candidate_seed in range(20):
        variant = select_intra_file_variant(plan, candidate_seed)
        seen.setdefault(variant["index"], candidate_seed)
        if len(seen) >= 2:
            seed_a, seed_b = list(seen.values())[:2]
            break
    assert seed_a is not None and seed_b is not None

    first_repo = tmp_path / "seed1"
    second_repo = tmp_path / "seed2"
    first_repo.mkdir()
    second_repo.mkdir()
    first_path = first_repo / "django/core/validators.py"
    second_path = second_repo / "django/core/validators.py"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text(source)
    second_path.write_text(source)

    first_result = apply_intra_file_reorder(first_repo, plan, runtime_seed=seed_a)
    second_result = apply_intra_file_reorder(second_repo, plan, runtime_seed=seed_b)

    assert first_result["status"] == "ok"
    assert second_result["status"] == "ok"
    assert first_result["variant_index"] != second_result["variant_index"]
    assert first_path.read_text() != second_path.read_text()


def test_apply_intra_file_reorder_keeps_non_reorderable_anchor_statements(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "import math\n\n"
        "def first():\n    return 1\n\n"
        "SENTINEL = 3\n\n"
        "def second():\n    return 2\n\n"
        "class Sample:\n"
        "    label = 'x'\n\n"
        "    def alpha(self):\n        return 'a'\n\n"
        "    flag = True\n\n"
        "    def beta(self):\n        return 'b'\n"
    )
    path = repo_root / "django/core/validators.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/validators.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    apply_intra_file_reorder(repo_root, plan, runtime_seed=3)
    rewritten = path.read_text()

    assert "import math" in rewritten
    assert "SENTINEL = 3" in rewritten
    assert "label = 'x'" in rewritten
    assert "flag = True" in rewritten


def test_build_intra_file_reorder_plan_skips_test_files(tmp_path):
    repo_root = tmp_path / "repo"
    for rel, content in {
        "sympy/core.py": "def first():\n    return 1\n\ndef second():\n    return 2\n",
        "sympy/tests/test_core.py": "def test_x():\n    assert True\n\ndef helper():\n    return 1\n",
    }.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    traj_path = tmp_path / "sympy__sympy-12481.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' sympy/core.py"},
                                {"command": "cd /testbed && sed -n '1,200p' sympy/tests/test_core.py"},
                            ]
                        },
                    },
                    {
                        "role": "exit",
                        "content": "+++ b/sympy/core.py\n+++ b/sympy/tests/test_core.py\n",
                    },
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)

    assert [item["path"] for item in plan["files"]] == ["sympy/core.py"]
    assert {"path": "sympy/tests/test_core.py", "reason": "test_file"} in plan["skipped_files"]


def test_intra_file_reorder_preserves_decorator_dependency_order(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "def stringfilter(fn):\n    return fn\n\n"
        "@stringfilter\n"
        "def decorated(value):\n    return value\n\n"
        "def helper():\n    return 1\n"
    )
    path = repo_root / "django/template/defaultfilters.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-15863.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/template/defaultfilters.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/template/defaultfilters.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    apply_intra_file_reorder(repo_root, plan, runtime_seed=7)
    rewritten = path.read_text()

    assert rewritten.index("def stringfilter") < rewritten.index("def decorated")


def test_intra_file_reorder_preserves_class_base_dependency_order(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "class Base:\n    pass\n\n"
        "class Child(Base):\n    pass\n\n"
        "def helper():\n    return 1\n"
    )
    path = repo_root / "django/core/example.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/example.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/example.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    apply_intra_file_reorder(repo_root, plan, runtime_seed=9)
    rewritten = path.read_text()

    assert rewritten.index("class Base") < rewritten.index("class Child")


def test_restore_original_file_order_roundtrips_reordered_source(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "def first():\n    return 1\n\n"
        "def second():\n    return 2\n\n"
        "class Sample:\n"
        "    def alpha(self):\n        return 'a'\n\n"
        "    def beta(self):\n        return 'b'\n\n"
        "def third():\n    return 3\n"
    )
    path = repo_root / "django/core/validators.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/validators.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    result = apply_intra_file_reorder(repo_root, plan, runtime_seed=17)
    reordered = path.read_text()
    restored = restore_original_file_order(reordered, result["file_orders"]["django/core/validators.py"])

    assert _top_level_order(restored) == _top_level_order(source)
    assert _class_method_order(restored, "Sample") == _class_method_order(source, "Sample")
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))


def test_intra_file_reorder_preserves_property_setter_order(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "class QuerySet:\n"
        "    @property\n"
        "    def query(self):\n"
        "        return self._query\n\n"
        "    @query.setter\n"
        "    def query(self, value):\n"
        "        self._query = value\n\n"
        "    def other(self):\n"
        "        return 1\n"
    )
    path = repo_root / "django/db/models/query.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-12050.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/db/models/query.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/db/models/query.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    apply_intra_file_reorder(repo_root, plan, runtime_seed=11)
    rewritten = path.read_text()

    assert rewritten.index("def query(self):") < rewritten.index("def query(self, value):")


def test_intra_file_reorder_preserves_named_setter_dependency(tmp_path):
    repo_root = tmp_path / "repo"
    source = (
        "class EmailValidator:\n"
        "    @property\n"
        "    def domain_whitelist(self):\n"
        "        return []\n\n"
        "    @domain_whitelist.setter\n"
        "    def domain_whitelist(self, value):\n"
        "        self._domain_whitelist = value\n\n"
        "    def compare(self):\n"
        "        return True\n"
    )
    path = repo_root / "django/core/validators.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)

    traj_path = tmp_path / "django__django-14349.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/validators.py\n"},
                ]
            }
        )
    )

    plan = build_intra_file_reorder_plan(traj_path, repo_root)
    apply_intra_file_reorder(repo_root, plan, runtime_seed=13)
    rewritten = path.read_text()

    assert rewritten.index("def domain_whitelist(self):") < rewritten.index("def domain_whitelist(self, value):")
