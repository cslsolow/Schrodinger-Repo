"""Tests for SemanticMappingManager with session notebook."""
import json
import tempfile
from pathlib import Path

import pytest

from glasses.manager import SemanticMappingManager


@pytest.fixture
def mapping_dir(tmp_path):
    """Create a temp dir with mapping files."""
    mapping = {
        "get_queryset": "fetch_dataset",
        "resolve_field": "determine_column",
        "MyModel": "MyEntity",
    }
    token_mapping = {
        "queryset": "dataset",
        "resolve": "determine",
        "field": "column",
    }
    (tmp_path / "mapping_seed_42.json").write_text(json.dumps(mapping))
    (tmp_path / "token_mapping.json").write_text(json.dumps(token_mapping))
    return tmp_path


@pytest.fixture
def manager(mapping_dir):
    return SemanticMappingManager(mapping_dir, seed=42)


def test_to_virtual_records_notebook(manager):
    """to_virtual should record actual replacements in session_notebook."""
    text = "def get_queryset(self): pass"
    result = manager.to_virtual(text)
    assert "fetch_dataset" in result
    assert "fetch_dataset" in manager.session_notebook
    assert manager.session_notebook["fetch_dataset"] == "get_queryset"


def test_to_real_uses_notebook(manager):
    """to_real should reverse replacements recorded in notebook."""
    # First: to_virtual records the replacement
    manager.to_virtual("call get_queryset here")
    # Then: to_real should reverse it
    result = manager.to_real("call fetch_dataset here")
    assert "get_queryset" in result


def test_to_real_fallback_to_backward_map(manager):
    """to_real should fall back to backward_map for unseen virtual names."""
    # Don't call to_virtual first — simulate LLM using a virtual name
    # it learned from task description
    # to_real translates virtual->real, so "determine_column" -> "resolve_field"
    result = manager.to_real("the determine_column method")
    assert "resolve_field" in result


def test_notebook_accumulates_across_steps(manager):
    """Notebook should accumulate across multiple to_virtual calls."""
    manager.to_virtual("get_queryset is defined")
    manager.to_virtual("resolve_field is also here")

    assert len(manager.session_notebook) == 2
    assert "fetch_dataset" in manager.session_notebook
    assert "determine_column" in manager.session_notebook


def test_unmapped_text_unchanged(manager):
    """Text without mapped identifiers should pass through unchanged."""
    text = "just some plain text with no identifiers"
    assert manager.to_virtual(text) == text
    assert manager.to_real(text) == text


def test_get_translator_stats(manager):
    """Stats should report zero LLM cost."""
    manager.to_virtual("get_queryset")
    stats = manager.get_translator_stats()
    assert stats["translator_extra_cost"] == 0.0
    assert stats["session_notebook_size"] == 1


def test_legacy_mapping_dir_works_with_explicit_enabled_layers(tmp_path):
    (tmp_path / "mapping_seed_42.json").write_text(json.dumps({"get_queryset": "fetch_dataset"}))

    manager = SemanticMappingManager(tmp_path, seed=42, enabled_layers=[])

    assert manager.to_virtual("call get_queryset here") == "call fetch_dataset here"


def test_manager_composes_namespace_without_identity(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py", "db": "data"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"], project_name="django")
    generic = manager.to_virtual("django/db/models/aggregates.py Count")
    result = manager.to_virtual_output("django/db/models/aggregates.py Count")

    assert generic == "django/db/models/aggregates.py Tally"
    assert result == "working_repository/data/models/totals.py Tally"


def test_manager_l2_only_still_virtualizes_repo_root(tmp_path):
    bundle_dir = tmp_path / "xarray"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"xarray": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"DataArray": "DataContainer"}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"core": "nucleus", "dataarray.py": "array_data.py"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"], project_name="xarray")

    translated = manager.to_virtual_command("sed -n '1,20p' /testbed/xarray/core/dataarray.py && grep DataArray /testbed/xarray/core/dataarray.py")

    assert "/testbed/working_repository/nucleus/array_data.py" in translated
    assert "DataContainer" in translated


def test_manager_composes_identity_and_namespace(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )
    generic = manager.to_virtual("/testbed/django/db/models/aggregates.py Count")
    result = manager.to_virtual_output("/testbed/django/db/models/aggregates.py Count")

    assert generic == "/testbed/working_repository/db/models/aggregates.py Tally"
    assert result == "/testbed/working_repository/db/models/totals.py Tally"


def test_manager_respects_explicit_empty_enabled_layers(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=[], project_name="django")
    result = manager.to_virtual_output("/testbed/django/db/models/aggregates.py Count")

    assert result == "/testbed/django/db/models/aggregates.py Count"


def test_manager_round_trips_namespace_paths_in_to_real(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    assert manager.to_virtual_output("/testbed/django/db/models/aggregates.py Count") == (
        "/testbed/working_repository/db/models/totals.py Tally"
    )
    assert manager.to_real("cat /testbed/working_repository/db/models/totals.py") == (
        "cat /testbed/django/db/models/aggregates.py"
    )


def test_manager_round_trips_bare_filenames_in_to_real(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"])

    assert manager.to_virtual_output("cat aggregates.py Count") == "cat totals.py Tally"
    assert manager.to_real("cat totals.py") == "cat aggregates.py"


def test_manager_to_virtual_command_keeps_path_map_separate_from_symbol_map(tmp_path):
    bundle_dir = tmp_path / "xarray"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"xarray": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"DataArray": "DataContainer"}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"core": "nucleus", "dataarray.py": "array_data.py"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="xarray",
    )

    command = "sed -n '1,20p' /testbed/xarray/core/dataarray.py && grep DataArray /testbed/xarray/core/dataarray.py"
    translated = manager.to_virtual_command(command)

    assert "/testbed/working_repository/nucleus/array_data.py" in translated
    assert "DataContainer" in translated
    assert "DataContainer.py" not in translated


def test_manager_to_real_patch_round_trips_diff_text(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    patch = (
        "diff --git a/working_repository/db/models/totals.py b/working_repository/db/models/totals.py\n"
        "--- a/working_repository/db/models/totals.py\n"
        "+++ b/working_repository/db/models/totals.py\n"
        "@@ -1 +1 @@\n"
        "-class Tally:\n"
        "+class BetterTally:\n"
    )

    result = manager.to_real_patch(patch)

    assert "working_repository" not in result
    assert "totals.py" not in result
    assert "django/db/models/aggregates.py" in result
    assert "-class Count:" in result


def test_manager_to_real_command_restores_cased_symbol_names(tmp_path):
    bundle_dir = tmp_path / "matplotlib"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"matplotlib": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Axis": "Scale_line"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="matplotlib",
    )

    translated = manager.to_real_command('python -c "print(Scale_line)"')

    assert "Scale_line" not in translated
    assert 'python -c "print(Axis)"' == translated


def test_manager_to_real_command_round_trips_virtual_paths_and_cased_symbols(tmp_path):
    bundle_dir = tmp_path / "xarray"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"xarray": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"DataArray": "DataContainer"}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"core": "nucleus", "dataarray.py": "array_data.py"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="xarray",
    )

    command = "sed -n '1,20p' /testbed/working_repository/nucleus/array_data.py && grep DataContainer /testbed/working_repository/nucleus/array_data.py"
    translated = manager.to_real_command(command)

    assert "/testbed/xarray/core/dataarray.py" in translated
    assert "DataArray" in translated
    assert "DataContainer.py" not in translated


def test_manager_does_not_apply_namespace_path_map_to_plain_text(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"db": "data"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"])

    assert manager.to_virtual("db error Count") == "db error Tally"
    assert manager.to_virtual_output("db error Count") == "db error Tally"


def test_manager_to_real_command_rewrites_python_c_dotted_imports(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"db": "storage_engine", "models": "object_models", "utils": "toolkit"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    command = 'python -c "from working_repository.storage_engine.object_models import Count; import working_repository.toolkit"'
    result = manager.to_real_command(command)

    assert "working_repository" not in result
    assert "django.db.models" in result
    assert "django.utils" in result


def test_manager_mapping_table_is_case_insensitive_for_symbols_and_paths(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"db": "storage_engine", "models": "object_models", "aggregates.py": "totals.py"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    text = "/testbed/DJANGO/DB/MODELS/AGGREGATES.PY count django.DB.Models"
    result = manager.to_virtual_text(text)

    assert "/testbed/working_repository/storage_engine/object_models/totals.py" in result
    assert "Tally" in result
    assert "working_repository.storage_engine.object_models" in result


def test_manager_text_channel_sanitizes_repo_name_in_identifiers_and_domains(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    text = "Django issue mentions django_content_types and code.djangoproject.com"
    result = manager.to_virtual_text(text)

    assert "Django" not in result
    assert "django_content_types" not in result
    assert "code.djangoproject.com" not in result
    assert "working_repository issue" in result
    assert "working_repository_content_types" in result
    assert "code.working_repository.com" in result


def test_manager_text_channel_sanitizes_repo_name_case_insensitively(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    text = "Django, DJANGO, and django should all become the same repo alias."
    result = manager.to_virtual_text(text)

    assert "Django" not in result
    assert "DJANGO" not in result
    assert "django" not in result
    assert result.count("working_repository") == 3


def test_manager_text_channel_sanitizes_repo_prefixed_identifiers(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    text = "Examples mention django30-venv and django_error2.models"
    result = manager.to_virtual_text(text)

    assert "django30-venv" not in result
    assert "django_error2.models" not in result
    assert "working_repository30-venv" in result
    assert "working_repository_error2.models" in result


def test_manager_preserves_shell_command_head_in_virtual_command(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally", "sort": "arrange"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"])

    cmd = "sort /testbed/django/db/models/aggregates.py | grep Count"
    translated = manager.to_virtual_command(cmd)

    assert translated.startswith("sort ")
    assert "arrange " not in translated
    assert "totals.py" in translated
    assert "Tally" in translated


def test_manager_preserves_shell_command_head_in_real_command(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally", "sort": "arrange"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["namespace_l2"]}))

    manager = SemanticMappingManager(bundle_dir, seed=42, enabled_layers=["namespace_l2"])

    cmd = "sort /testbed/django/db/models/totals.py | grep Tally"
    translated = manager.to_real_command(cmd)

    assert translated.startswith("sort ")
    assert "arrange " not in translated
    assert "aggregates.py" in translated
    assert "Count" in translated


def test_manager_rewrites_dotted_module_paths_in_real_command(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"db": "storage_engine", "models": "object_models", "utils": "toolkit"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    cmd = "python -c \"from working_repository.storage_engine.object_models import Sum; import working_repository.toolkit\""
    translated = manager.to_real_command(cmd)

    assert "from django.db.models import Sum" in translated
    assert "import django.utils" in translated


def test_manager_error_output_maps_repo_owned_modules_but_keeps_third_party_real(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(
        json.dumps({"db": "storage_engine", "models": "object_models", "utils": "toolkit"})
    )
    (bundle_dir / "mapping_meta.json").write_text(json.dumps({"enabled_layers": ["identity_l1", "namespace_l2"]}))

    manager = SemanticMappingManager(
        bundle_dir,
        seed=42,
        enabled_layers=["identity_l1", "namespace_l2"],
        project_name="django",
    )

    text = (
        'Traceback (most recent call last):\n'
        '  File "/testbed/django/core/management/__init__.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'django.utils'\n"
        "ModuleNotFoundError: No module named 'asgiref.sync'\n"
    )

    result = manager.to_virtual_error_output(text)

    assert '/testbed/working_repository/core/management/__init__.py' in result
    assert "working_repository.toolkit" in result
    assert "asgiref.sync" in result
    assert "working_repository.sync" not in result
