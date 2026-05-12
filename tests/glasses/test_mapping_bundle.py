import json

import pytest

from glasses.mapping_bundle import compose_forward_map, load_mapping_bundle


def test_compose_forward_map_respects_enabled_layers(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({"django": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"Count": "Tally"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"aggregates.py": "totals.py"}))
    (bundle_dir / "mapping_meta.json").write_text(
        json.dumps(
            {
                "mapping_version": "l2-v1",
                "repo_name": "django",
                "enabled_layers": ["identity_l1", "namespace_l2"],
            }
        )
    )

    bundle = load_mapping_bundle(bundle_dir)

    assert compose_forward_map(bundle, ["namespace_l2"]) == {
        "Count": "Tally",
        "aggregates.py": "totals.py",
    }
    assert compose_forward_map(bundle, ["identity_l1", "namespace_l2"]) == {
        "django": "working_repository",
        "Count": "Tally",
        "aggregates.py": "totals.py",
    }


def test_load_mapping_bundle_includes_mapping_meta(tmp_path):
    bundle_dir = tmp_path / "django"
    bundle_dir.mkdir()
    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (bundle_dir / "mapping_meta.json").write_text(
        json.dumps(
            {
                "mapping_version": "l2-v1",
                "repo_name": "django",
                "enabled_layers": ["identity_l1"],
            }
        )
    )

    bundle = load_mapping_bundle(bundle_dir)

    assert bundle["mapping_meta"] == {
        "mapping_version": "l2-v1",
        "repo_name": "django",
        "enabled_layers": ["identity_l1"],
    }


def test_compose_forward_map_rejects_duplicate_keys_across_layers():
    bundle = {
        "identity_map": {"shared": "working_repository"},
        "namespace_symbol_map": {"shared": "Tally"},
        "namespace_path_map": {},
        "mapping_meta": {},
    }

    with pytest.raises(ValueError, match="shared"):
        compose_forward_map(bundle, ["identity_l1", "namespace_l2"])
