from pathlib import Path

from glasses.layout_planner import materialize_layout_map
from glasses.layout_transformer import apply_layout_map


def test_apply_layout_map_moves_files_and_rewrites_absolute_imports(tmp_path):
    repo_root = tmp_path / "repo"
    files = {
        "django/forms/fields.py": "from django.forms.boundfield import BoundField\n",
        "django/forms/boundfield.py": "class BoundField:\n    pass\n",
        "django/forms/widgets.py": "from django.forms.boundfield import BoundField\n",
        "django/apps/__init__.py": "",
    }
    for rel, content in files.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    layout_map = {
        "instance_id": "django__django-14034",
        "repo_name": "django",
        "status": "ok",
        "source_dir": "django/forms",
        "target_dir": "django/apps",
        "target_root": "django/apps",
        "path_map": {
            "django/forms/boundfield.py": "django/apps/boundfield.py",
            "django/forms/widgets.py": "django/apps/widgets.py",
        },
        "reverse_map": {
            "django/apps/boundfield.py": "django/forms/boundfield.py",
            "django/apps/widgets.py": "django/forms/widgets.py",
        },
        "module_map": {
            "django.forms.boundfield": "django.apps.boundfield",
            "django.forms.widgets": "django.apps.widgets",
        },
    }

    result = apply_layout_map(repo_root, layout_map)

    assert result["status"] == "ok"
    assert not (repo_root / "django/forms/boundfield.py").exists()
    assert (repo_root / "django/apps/boundfield.py").exists()
    assert (repo_root / "django/forms/fields.py").read_text() == "from django.apps.boundfield import BoundField\n"
    assert (repo_root / "django/apps/widgets.py").read_text() == "from django.apps.boundfield import BoundField\n"


def test_apply_layout_map_noops_when_status_not_ok(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "demo.py").parent.mkdir(parents=True, exist_ok=True)
    (repo_root / "demo.py").write_text("print('x')\n")

    result = apply_layout_map(repo_root, {"status": "no_viable_target", "path_map": {}})

    assert result["status"] == "skipped"
    assert (repo_root / "demo.py").exists()


def test_apply_layout_map_rewrites_import_lines_but_not_plain_strings(tmp_path):
    repo_root = tmp_path / "repo"
    files = {
        "django/forms/fields.py": (
            '"""django.forms.boundfield should stay in docstring."""\n'
            "# django.forms.boundfield should stay in comment\n"
            "VALUE = 'django.forms.boundfield should stay in string'\n"
            "import django.forms.boundfield as bf\n"
            "from django.forms import boundfield\n"
        ),
        "django/forms/boundfield.py": "class BoundField:\n    pass\n",
        "django/apps/__init__.py": "",
    }
    for rel, content in files.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    layout_map = {
        "status": "ok",
        "path_map": {
            "django/forms/boundfield.py": "django/apps/forms/boundfield.py",
        },
    }

    apply_layout_map(repo_root, layout_map)

    rewritten = (repo_root / "django/forms/fields.py").read_text()
    assert '"""django.forms.boundfield should stay in docstring."""' in rewritten
    assert "# django.forms.boundfield should stay in comment" in rewritten
    assert "VALUE = 'django.forms.boundfield should stay in string'" in rewritten
    assert "import django.apps.forms.boundfield as bf" in rewritten
    assert "from django.apps.forms import boundfield" in rewritten


def test_apply_layout_map_splits_grouped_from_import_for_single_moved_member(tmp_path):
    repo_root = tmp_path / "repo"
    files = {
        "django/db/models/fields/__init__.py": "from django.core import checks, exceptions, validators\n",
        "django/core/validators.py": "class URLValidator:\n    pass\n",
        "django/records/__init__.py": "",
    }
    for rel, content in files.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    layout_map = {
        "status": "ok",
        "target_root": "django/records",
        "path_map": {
            "django/core/validators.py": "django/records/validators.py",
        },
        "module_map": {
            "django.core.validators": "django.records.validators",
        },
    }

    apply_layout_map(repo_root, layout_map)

    rewritten = (repo_root / "django/db/models/fields/__init__.py").read_text()
    assert rewritten == (
        "from django.core import checks, exceptions\n"
        "from django.records import validators\n"
    )


def test_apply_layout_map_in_path_based_mode_keeps_logical_import_names(tmp_path):
    repo_root = tmp_path / "repo"
    files = {
        "django/forms/fields.py": (
            "from django.forms.boundfield import BoundField\n"
            "from django.forms.widgets import TextInput\n"
        ),
        "django/forms/boundfield.py": (
            "from .widgets import TextInput\n"
            "from django.forms.widgets import TextInput as AbsoluteTextInput\n"
        ),
        "django/forms/widgets.py": "from .renderers import get_default_renderer\n",
        "django/forms/renderers.py": "def get_default_renderer():\n    return None\n",
    }
    for rel, content in files.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    layout_map = {
        "status": "ok",
        "path_based_mode": True,
        "target_root": "django/assembly",
        "path_map": {
            "django/forms/boundfield.py": "django/assembly/boundfield.py",
            "django/forms/widgets.py": "django/assembly/widgets.py",
        },
        "module_map": {
            "django.forms.boundfield": "django.assembly.boundfield",
            "django.forms.widgets": "django.assembly.widgets",
        },
    }

    apply_layout_map(repo_root, layout_map)

    assert (repo_root / "django/assembly/widgets.py").read_text() == "from .renderers import get_default_renderer\n"
    assert (repo_root / "django/assembly/boundfield.py").read_text() == (
        "from .widgets import TextInput\n"
        "from django.forms.widgets import TextInput as AbsoluteTextInput\n"
    )
    assert (repo_root / "django/forms/fields.py").read_text() == (
        "from django.forms.boundfield import BoundField\n"
        "from django.forms.widgets import TextInput\n"
    )


def test_apply_layout_map_keeps_template_resources_in_place_for_code_only_mode(tmp_path):
    repo_root = tmp_path / "repo"
    files = {
        "django/forms/widgets.py": "template_name = 'django/forms/widgets/email.html'\n",
        "django/forms/renderers.py": "pass\n",
        "django/forms/templates/django/forms/widgets/email.html": "<input type='email'>\n",
    }
    for rel, content in files.items():
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    layout_map = {
        "status": "ok",
        "target_root": "django/assembly",
        "path_map": {
            "django/forms/renderers.py": "django/assembly/renderers.py",
            "django/forms/widgets.py": "django/assembly/widgets.py",
        },
        "module_map": {
            "django.forms.renderers": "django.assembly.renderers",
            "django.forms.widgets": "django.assembly.widgets",
        },
        "resource_map": {
            "django/forms/templates/django/forms/widgets/email.html": "django/assembly/templates/django/forms/widgets/email.html"
        },
    }

    apply_layout_map(repo_root, layout_map)

    assert not (repo_root / "django/assembly/templates/django/forms/widgets/email.html").exists()
    assert (repo_root / "django/forms/templates/django/forms/widgets/email.html").exists()
    assert (repo_root / "django/assembly/widgets.py").read_text() == (
        "template_name = 'django/forms/widgets/email.html'\n"
    )
