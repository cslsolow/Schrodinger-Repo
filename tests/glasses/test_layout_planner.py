import json

from glasses.layout_planner import build_layout_plan, materialize_layout_map, resolve_instance_repo_root
from glasses.layout_policy import get_repo_policy, infer_repo_name, is_protected_path, prefers_source_path


def test_layout_policy_infers_repo_and_protects_django_command_dirs():
    assert infer_repo_name("django__django-16116") == "django"
    policy = get_repo_policy("django")
    assert is_protected_path("django/core/management/commands/makemigrations.py", policy)
    assert prefers_source_path("django/forms/fields.py", policy)
    assert not prefers_source_path("tests/forms_tests/tests.py", policy)


def test_layout_policy_protects_init_and_conftest_files_globally():
    policy = get_repo_policy("matplotlib")
    assert is_protected_path("lib/matplotlib/__init__.py", policy)
    assert is_protected_path("conftest.py", policy)


def test_layout_policy_handles_owner_scoped_repos():
    pydata = get_repo_policy("pydata")
    assert prefers_source_path("xarray/core/dataarray.py", pydata)
    assert not prefers_source_path("doc/conf.py", pydata)

    pallets = get_repo_policy("pallets")
    assert prefers_source_path("src/flask/app.py", pallets)


def test_build_layout_plan_prefers_non_protected_primary_dir(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in [
        "django/forms/fields.py",
        "django/forms/widgets.py",
        "django/core/validators.py",
        "django/template/defaultfilters.py",
        "django/core/management/commands/migrate.py",
        "tests/forms_tests/tests.py",
    ]:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# demo\n")

    traj_path = tmp_path / "django__django-14034.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/fields.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/widgets.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/core/management/commands/migrate.py"},
                            ]
                        },
                    },
                    {
                        "role": "tool",
                        "content": "django/forms/fields.py:10:Field\n"
                        "django/core/management/commands/migrate.py:20:Command\n",
                    },
                    {"role": "exit", "content": "+++ b/django/forms/fields.py\n"},
                ]
            }
        )
    )

    plan = build_layout_plan(traj_path, repo_root)

    assert plan["status"] == "ok"
    assert plan["unit_kind"] == "module_group"
    assert plan["primary_dir"] == "django/forms"
    assert plan["primary_file"]["path"] == "django/forms/fields.py"
    assert [item["path"] for item in plan["support_files"]] == [
        "django/forms/fields.py",
        "django/forms/widgets.py",
    ]
    assert "django/core/management/commands/migrate.py" in {item["path"] for item in plan["skipped_files"]}
    assert plan["target_candidates"]
    assert all(not item["path"].startswith("django/core/management/commands") for item in plan["target_candidates"])


def test_build_layout_plan_expands_module_group_with_same_package_relative_imports(tmp_path):
    repo_root = tmp_path / "repo"
    for rel, content in {
        "django/forms/boundfield.py": "from .widgets import TextInput\n",
        "django/forms/widgets.py": "from .renderers import get_default_renderer\n",
        "django/forms/renderers.py": "def get_default_renderer():\n    return None\n",
        "django/forms/fields.py": "from .boundfield import BoundField\n",
    }.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    traj_path = tmp_path / "django__django-14534.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/boundfield.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/widgets.py"},
                            ]
                        },
                    },
                    {
                        "role": "tool",
                        "content": "django/forms/boundfield.py:10:BoundField\n"
                        "django/forms/widgets.py:20:TextInput\n",
                    },
                    {"role": "exit", "content": "+++ b/django/forms/boundfield.py\n"},
                ]
            }
        )
    )

    plan = build_layout_plan(traj_path, repo_root)

    assert plan["unit_kind"] == "module_group"
    assert [item["path"] for item in plan["support_files"]] == [
        "django/forms/boundfield.py",
        "django/forms/renderers.py",
        "django/forms/widgets.py",
    ]


def test_build_layout_plan_excludes_anchor_modules_from_module_group(tmp_path):
    repo_root = tmp_path / "repo"
    for rel, content in {
        "django/forms/boundfield.py": "from .widgets import TextInput\n",
        "django/forms/widgets.py": "from .renderers import get_default_renderer\n",
        "django/forms/renderers.py": "from pathlib import Path\nDIRS = [Path(__file__).parent / 'templates']\n",
    }.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    traj_path = tmp_path / "django__django-14534.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/boundfield.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/widgets.py"},
                            ]
                        },
                    },
                    {"role": "tool", "content": "django/forms/boundfield.py:10:BoundField\n"},
                    {"role": "exit", "content": "+++ b/django/forms/boundfield.py\n"},
                ]
            }
        )
    )

    plan = build_layout_plan(traj_path, repo_root)

    assert [item["path"] for item in plan["support_files"]] == [
        "django/forms/boundfield.py",
        "django/forms/widgets.py",
    ]
    assert {"path": "django/forms/renderers.py", "reason": "anchor_module"} in plan["skipped_files"]


def test_materialize_layout_map_adds_template_resource_map_for_module_group(tmp_path):
    repo_root = tmp_path / "repo"
    for rel, content in {
        "django/forms/widgets.py": "template_name = 'django/forms/widgets/email.html'\n",
        "django/forms/renderers.py": "pass\n",
        "django/forms/templates/django/forms/widgets/email.html": "<input type='email'>\n",
    }.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    plan = {
        "instance_id": "django__django-14534",
        "repo_name": "django",
        "policy": "django",
        "status": "ok",
        "primary_dir": "django/forms",
        "support_files": [
            {"path": "django/forms/renderers.py", "score": 0.0, "evidence": {"closure": 1}},
            {"path": "django/forms/widgets.py", "score": 6.5, "evidence": {"read": 8}},
        ],
        "target_candidates": [
            {"path": "django/assembly", "score": 10},
        ],
        "skipped_files": [],
    }

    layout_map = materialize_layout_map(plan, repo_root)

    assert layout_map["resource_map"] == {
        "django/forms/templates/django/forms/widgets/email.html": "django/assembly/templates/django/forms/widgets/email.html"
    }


def test_build_layout_plan_reports_when_only_protected_files_exist(tmp_path):
    repo_root = tmp_path / "repo"
    path = repo_root / "django/core/management/commands/migrate.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# demo\n")

    traj_path = tmp_path / "django__django-16116.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/management/commands/migrate.py"},
                            ]
                        },
                    },
                    {"role": "exit", "content": "+++ b/django/core/management/commands/migrate.py\n"},
                ]
            }
        )
    )

    plan = build_layout_plan(traj_path, repo_root)

    assert plan["status"] == "no_candidate_files"
    assert plan["skipped_files"] == [
        {"path": "django/core/management/commands/migrate.py", "reason": "protected"}
    ]


def test_resolve_instance_repo_root_prefers_swebench_prefix(tmp_path):
    repo_root = tmp_path / "swe-bench_django__django-10097"
    repo_root.mkdir()

    resolved = resolve_instance_repo_root(tmp_path, "django__django-10097")

    assert resolved == repo_root


def test_materialize_layout_map_uses_first_viable_target(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in [
        "django/forms/fields.py",
        "django/forms/widgets.py",
        "django/core/__init__.py",
        "django/db/__init__.py",
    ]:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# demo\n")

    traj_path = tmp_path / "django__django-14034.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/fields.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/forms/widgets.py"},
                            ]
                        },
                    },
                    {"role": "tool", "content": "django/forms/fields.py:10:Field\n"},
                    {"role": "exit", "content": "+++ b/django/forms/fields.py\n"},
                ]
            }
        )
    )

    plan = build_layout_plan(traj_path, repo_root)
    layout_map = materialize_layout_map(plan, repo_root)

    assert layout_map["status"] == "ok"
    assert layout_map["path_based_mode"] is True
    assert layout_map["target_dir"].startswith("django/")
    assert layout_map["target_root"] == layout_map["target_dir"]
    assert layout_map["path_map"] == {
        "django/forms/fields.py": f"{layout_map['target_dir']}/fields.py",
        "django/forms/widgets.py": f"{layout_map['target_dir']}/widgets.py",
    }
    assert layout_map["module_map"] == {
        "django.forms.fields": layout_map["path_map"]["django/forms/fields.py"][:-3].replace("/", "."),
        "django.forms.widgets": layout_map["path_map"]["django/forms/widgets.py"][:-3].replace("/", "."),
    }
    assert layout_map["reverse_map"][layout_map["path_map"]["django/forms/fields.py"]] == "django/forms/fields.py"


def test_materialize_layout_map_skips_conflicting_target(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in [
        "django/forms/fields.py",
        "django/forms/widgets.py",
        "django/core/fields.py",
        "django/db/__init__.py",
    ]:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# demo\n")

    plan = {
        "instance_id": "django__django-14034",
        "repo_name": "django",
        "policy": "django",
        "status": "ok",
        "primary_dir": "django/forms",
        "support_files": [
            {"path": "django/forms/fields.py", "score": 10.0, "evidence": {"edited": 1}},
            {"path": "django/forms/widgets.py", "score": 4.0, "evidence": {"read": 2}},
        ],
        "target_candidates": [
            {"path": "django/core", "score": 9},
            {"path": "django/db", "score": 9},
        ],
        "skipped_files": [],
    }

    layout_map = materialize_layout_map(plan, repo_root)

    assert layout_map["status"] == "ok"
    assert layout_map["target_dir"] == "django/db"
    assert layout_map["target_root"] == "django/db"
    assert layout_map["path_map"]["django/forms/fields.py"] == "django/db/fields.py"


def test_materialize_layout_map_reports_when_no_viable_target_exists(tmp_path):
    repo_root = tmp_path / "repo"
    for rel in [
        "django/forms/fields.py",
        "django/core/fields.py",
        "django/db/fields.py",
    ]:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# demo\n")

    plan = {
        "instance_id": "django__django-14034",
        "repo_name": "django",
        "policy": "django",
        "status": "ok",
        "primary_dir": "django/forms",
        "support_files": [
            {"path": "django/forms/fields.py", "score": 10.0, "evidence": {"edited": 1}},
        ],
        "target_candidates": [
            {"path": "django/core", "score": 9},
            {"path": "django/db", "score": 9},
        ],
        "skipped_files": [],
    }

    layout_map = materialize_layout_map(plan, repo_root)

    assert layout_map["status"] == "no_viable_target"
    assert layout_map["path_map"] == {}
