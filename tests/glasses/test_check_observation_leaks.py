import json

import pytest

from glasses.check_observation_leaks import summarize_bundle_quality, summarize_leaks, validate_bundle_quality


def test_summarize_leaks_counts_real_namespace_mentions(tmp_path):
    run_dir = tmp_path / "run" / "demo__demo-1"
    run_dir.mkdir(parents=True)
    (run_dir / "demo__demo-1.traj.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Inspect demo/query.py"},
                    {"role": "assistant", "content": "I found DemoThing"},
                    {
                        "role": "assistant",
                        "content": "ok",
                        "extra": {
                            "actions": [
                                {"command": "grep DemoThing /testbed/demo/query.py"}
                            ]
                        },
                    },
                ]
            }
        )
    )

    summary = summarize_leaks(tmp_path / "run", repo_terms=["demo", "DemoThing"])

    assert summary["traj_count"] == 1
    assert summary["user_message_leak_files"] == 1
    assert summary["assistant_message_leak_files"] == 1
    assert summary["command_leak_files"] == 1


def test_summarize_leaks_reads_list_content_and_counts_one_file_once(tmp_path):
    run_dir = tmp_path / "run" / "demo__demo-2"
    run_dir.mkdir(parents=True)
    (run_dir / "demo__demo-2.traj.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "demo appears twice: demo"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "DemoThing is still visible. DemoThing again.",
                            }
                        ],
                    },
                ]
            }
        )
    )

    summary = summarize_leaks(tmp_path / "run", repo_terms=["demo", "DemoThing"])

    assert summary["traj_count"] == 1
    assert summary["user_message_leak_files"] == 1
    assert summary["assistant_message_leak_files"] == 1
    assert summary["command_leak_files"] == 0


def test_summarize_bundle_quality_reports_forbidden_token_mappings(tmp_path):
    repo_maps_dir = tmp_path / "repo_maps"
    bundle_dir = repo_maps_dir / "demo"
    instance_dir = repo_maps_dir / "demo__demo-1"
    bundle_dir.mkdir(parents=True)
    instance_dir.mkdir()

    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"QuerySet": "SearchSet"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({"query.py": "search.py"}))
    (instance_dir / "token_mapping.json").write_text(
        json.dumps(
            {
                "query": "search",
                "sort": "arrange",
                "requests": "api_calls",
                "list": "arrayish",
            }
        )
    )

    summary = summarize_bundle_quality(repo_maps_dir, "demo")

    assert summary["symbol_changed"] == 1
    assert summary["path_changed"] == 1
    assert summary["token_changed"] == 4
    assert summary["forbidden_shell_changed"] == [{"real": "sort", "virtual": "arrange"}]
    assert summary["forbidden_third_party_changed"] == [{"real": "requests", "virtual": "api_calls"}]
    assert summary["forbidden_builtin_changed"] == [{"real": "list", "virtual": "arrayish"}]


def test_validate_bundle_quality_raises_on_forbidden_token_mappings(tmp_path):
    repo_maps_dir = tmp_path / "repo_maps"
    bundle_dir = repo_maps_dir / "demo"
    instance_dir = repo_maps_dir / "demo__demo-1"
    bundle_dir.mkdir(parents=True)
    instance_dir.mkdir()

    (bundle_dir / "identity_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (instance_dir / "token_mapping.json").write_text(json.dumps({"sort": "arrange"}))

    with pytest.raises(ValueError, match="shell tokens mapped"):
        validate_bundle_quality(repo_maps_dir, "demo")


def test_validate_bundle_quality_raises_on_identity_namespace_overlap(tmp_path):
    repo_maps_dir = tmp_path / "repo_maps"
    bundle_dir = repo_maps_dir / "demo"
    instance_dir = repo_maps_dir / "demo__demo-1"
    bundle_dir.mkdir(parents=True)
    instance_dir.mkdir()

    (bundle_dir / "identity_map.json").write_text(json.dumps({"demo": "working_repository"}))
    (bundle_dir / "namespace_symbol_map.json").write_text(json.dumps({"demo": "demo", "demo.query": "demo.lookup"}))
    (bundle_dir / "namespace_path_map.json").write_text(json.dumps({}))
    (instance_dir / "token_mapping.json").write_text(json.dumps({}))

    with pytest.raises(ValueError, match="identity/namespace overlap"):
        validate_bundle_quality(repo_maps_dir, "demo")
