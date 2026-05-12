"""Tests for Level 4B function body rewrite target extraction and validation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import litellm
import pytest

from glasses.build_verified_level4b_store import _build_candidate, next_free_variant_index, verify_instance
from glasses.level4_llm_function_body_rewrite import (
    RewriteTarget,
    build_level4b_prompt,
    build_level4b_runtime_record,
    extract_pass_to_pass_test_sources,
    extract_rewrite_targets_from_golden_patch,
    generate_level4b_rewrite,
    parse_level4b_response,
    replace_function_in_source,
    rewrite_strength_gate,
    serialize_target_group,
    target_group_item_key,
    validate_rewritten_function_body,
)


def test_extracts_smallest_enclosing_function(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "def outer(x):\n"
        "    def inner(y):\n"
        "        return y + 1\n"
        "    return inner(x)\n"
    )
    patch = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -2,2 +2,2 @@\n"
        "     def inner(y):\n"
        "-        return y + 1\n"
        "+        return y + 2\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert skip_reason is None
    assert len(targets) == 1
    assert targets[0].path == Path("module.py")
    assert targets[0].function_name == "inner"


def test_extracts_from_multi_file_patch_and_skips_module_level_only_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module_level.py").write_text("VALUE = 1\n")
    (repo / "has_function.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    patch = (
        "diff --git a/module_level.py b/module_level.py\n"
        "--- a/module_level.py\n"
        "+++ b/module_level.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
        "diff --git a/has_function.py b/has_function.py\n"
        "--- a/has_function.py\n"
        "+++ b/has_function.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def target(value):\n"
        "-    return value + 1\n"
        "+    return value + 2\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert skip_reason is None
    assert [target.path for target in targets] == [Path("has_function.py")]
    assert [target.function_name for target in targets] == ["target"]


def test_dedupes_multiple_hunks_to_same_function(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "def target(value):\n"
        "    first = value + 1\n"
        "    second = value + 2\n"
        "    return first + second\n"
    )
    patch = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,4 +1,4 @@\n"
        " def target(value):\n"
        "-    first = value + 1\n"
        "+    first = value + 3\n"
        "@@ -2,3 +2,3 @@\n"
        "     second = value + 2\n"
        "-    return first + second\n"
        "+    return second + first\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert skip_reason is None
    assert len(targets) == 1
    assert targets[0].path == Path("module.py")
    assert targets[0].function_name == "target"


def test_hunk_spanning_two_functions_produces_two_targets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "def first(value):\n"
        "    return value + 1\n"
        "\n"
        "def second(value):\n"
        "    return value + 2\n"
    )
    patch = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def first(value):\n"
        "-    return value + 1\n"
        " \n"
        " def second(value):\n"
        "-    return value + 2\n"
        "+    return value + 3\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert skip_reason is None
    assert {(target.path, target.function_name) for target in targets} == {
        (Path("module.py"), "first"),
        (Path("module.py"), "second"),
    }


def test_extract_rewrite_targets_returns_stable_group_order(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "b.py").write_text(
        "def beta(value):\n"
        "    return value + 2\n"
    )
    (repo / "a.py").write_text(
        "def alpha(value):\n"
        "    return value + 1\n"
    )
    patch = (
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def beta(value):\n"
        "-    return value + 2\n"
        "+    return value + 3\n"
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def alpha(value):\n"
        "-    return value + 1\n"
        "+    return value + 4\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert skip_reason is None
    assert [(target.path, target.start_line, target.function_name) for target in targets] == [
        (Path("a.py"), 1, "alpha"),
        (Path("b.py"), 1, "beta"),
    ]


def test_skips_when_no_function_body_exists(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 1\n")
    patch = (
        "diff --git a/module.py b/module.py\n"
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )

    targets, skip_reason = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert targets == []
    assert skip_reason == "no_rewriteable_function_body"


def test_ignores_protected_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_module.py").write_text(
        "def test_target():\n"
        "    return 1\n"
    )
    patch = (
        "diff --git a/tests/test_module.py b/tests/test_module.py\n"
        "--- a/tests/test_module.py\n"
        "+++ b/tests/test_module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def test_target():\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    targets, _ = extract_rewrite_targets_from_golden_patch(patch, repo)

    assert targets == []


def test_validate_rewritten_function_body_rejects_signature_changes():
    original = (
        "@decorator\n"
        "def sample(x, y):\n"
        "    return x + y\n"
    )
    rewritten = (
        "@decorator\n"
        "def sample(x, y, z=0):\n"
        "    return x + y + z\n"
    )

    ok, reason = validate_rewritten_function_body(original, rewritten)

    assert ok is False
    assert reason == "function_signature_changed"


def test_validate_rewritten_function_body_rejects_sync_to_async_drift():
    original = (
        "def sample(x):\n"
        "    return x + 1\n"
    )
    rewritten = (
        "async def sample(x):\n"
        "    return x + 1\n"
    )

    ok, reason = validate_rewritten_function_body(original, rewritten, target=("sample", 1, 2))

    assert ok is False
    assert reason == "function_type_changed"


def test_validate_rewritten_function_body_selects_same_named_nested_function(tmp_path):
    original = (
        "def sample(x):\n"
        "    return x + 1\n"
        "\n"
        "def outer(y):\n"
        "    def sample(z):\n"
        "        return z + 2\n"
        "    return sample(y)\n"
    )
    rewritten = (
        "def sample(x):\n"
        "    return x + 1\n"
        "\n"
        "def outer(y):\n"
        "    def sample(z):\n"
        "        tmp = z + 2\n"
        "        return tmp * 3\n"
        "    return sample(y)\n"
    )

    ok, reason = validate_rewritten_function_body(original, rewritten, target=("sample", 5, 6))

    assert ok is True
    assert reason is None


def test_validate_and_strength_gate_allow_nested_helper_in_single_target_function():
    original = (
        "def sample(x):\n"
        "    return x + 1\n"
    )
    rewritten = (
        "def sample(x):\n"
        "    def helper(value):\n"
        "        return value + 1\n"
        "    return helper(x)\n"
    )

    ok, reason = validate_rewritten_function_body(original, rewritten)
    assert ok is True
    assert reason is None

    ok, reason = rewrite_strength_gate(original, rewritten)
    assert ok is True
    assert reason is None


def test_rewrite_strength_gate_rejects_temp_var_extraction_only():
    original = (
        "def sample(x):\n"
        "    return x + 1\n"
    )
    rewritten = (
        "def sample(x):\n"
        "    tmp = x + 1\n"
        "    return tmp\n"
    )

    ok, reason = rewrite_strength_gate(original, rewritten)

    assert ok is False
    assert reason == "rewrite_too_trivial"


def test_rewrite_strength_gate_allows_named_target_in_multi_function_source():
    original = (
        "def helper(value):\n"
        "    return value - 1\n"
        "\n"
        "def sample(x):\n"
        "    return x + 1\n"
    )
    rewritten = (
        "def helper(value):\n"
        "    return value - 1\n"
        "\n"
        "def sample(x):\n"
        "    tmp = x + 1\n"
        "    return tmp * 2\n"
    )

    ok, reason = rewrite_strength_gate(original, rewritten, target=("sample", 4, 5))

    assert ok is True
    assert reason is None


def test_build_level4b_runtime_record_prefers_group_metadata():
    targets = [
        RewriteTarget(path=Path("pkg.py"), function_name="second", start_line=1, end_line=2),
        RewriteTarget(path=Path("pkg.py"), function_name="first", start_line=4, end_line=6),
    ]
    target_group = serialize_target_group(targets)
    second_key = target_group_item_key(target_group[0])
    first_key = target_group_item_key(target_group[1])

    record = build_level4b_runtime_record(
        seed=42,
        runtime_seed=43,
        variant_name="variant_0",
        rewritten_files=["pkg.py"],
        target_group=target_group,
        target_group_signature="group:abc",
        rewrite_summaries={
            second_key: "flattened branch structure",
            first_key: "introduced staged flags",
        },
        behavioral_invariants={
            second_key: ["same return semantics"],
            first_key: ["same exceptions"],
        },
        rewrite_strength=0.55,
    )

    assert record["mode"] == "llm_function_body_rewrite"
    assert record["status"] == "applied"
    assert record["target_group"] == target_group
    assert record["target_group_size"] == 2
    assert record["target_group_signature"] == "group:abc"
    assert record["rewrite_summaries"][second_key] == "flattened branch structure"
    assert record["behavioral_invariants"][first_key] == ["same exceptions"]
    assert record["rewrite_strength"] == 0.55
    assert "target_function" not in record
    assert "rewrite_summary" not in record


def test_build_level4b_runtime_record_includes_target_metadata():
    target_group = [
        {"path": "pkg/mod.py", "function_name": "target", "start_line": 10, "end_line": 12},
    ]
    target_key = target_group_item_key(target_group[0])
    record = build_level4b_runtime_record(
        seed=42,
        runtime_seed=43,
        variant_name="variant_0",
        rewritten_files=["pkg/mod.py"],
        target_group=target_group,
        target_group_signature="group:def",
        rewrite_summaries={target_key: "restructured control flow"},
        behavioral_invariants={target_key: ["same return semantics"]},
        rewrite_strength=0.41,
    )

    assert record["mode"] == "llm_function_body_rewrite"
    assert record["status"] == "applied"
    assert record["target_group"] == target_group
    assert record["target_group_size"] == 1
    assert record["target_group_signature"] == "group:def"
    assert record["rewrite_summaries"][target_key] == "restructured control flow"
    assert record["rewrite_strength"] == 0.41


def test_next_free_variant_index_skips_sparse_existing_slots(tmp_path):
    mode_dir = tmp_path / "instance" / "llm_function_body_rewrite"
    (mode_dir / "variant_0").mkdir(parents=True)
    (mode_dir / "variant_2").mkdir()

    assert next_free_variant_index(mode_dir) == 1


def test_verify_instance_reports_partial_when_targets_exhausted(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    mode_dir = tmp_path / "output" / "demo" / "llm_function_body_rewrite"
    tmp_root = tmp_path / "tmp"

    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=1, end_line=2)
    candidate = {
        "status": "ok",
        "seed": 42,
        "runtime_seed": 42,
        "target_group": [{"path": "module.py", "function_name": "target", "start_line": 1, "end_line": 2}],
        "target_group_size": 1,
        "target_group_signature": "group:abc",
        "rewrite_summaries": {"module.py:000000001:target": "restructured control flow"},
        "behavioral_invariants": {"module.py:000000001:target": ["same return semantics"]},
        "rewrite_strength": 0.5,
        "rewritten_files": ["module.py"],
        "files": {"module.py": "def target(value):\n    return value + 2\n"},
        "level4_patch": "diff --git a/module.py b/module.py\n",
    }

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store._build_group_candidate", lambda **kwargs: candidate)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_official_preflight",
        lambda *args, **kwargs: {
            "status": "ok",
            "returncode": 0,
            "report": {"patch_successfully_applied": True, "tests_status": {"PASS_TO_PASS": {"failure": []}}},
        },
    )

    result = verify_instance(
        instance,
        repo_root,
        mode_dir,
        tmp_root,
        seed=42,
        variant_count=1,
        max_attempts=6,
        target_variants=3,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 0.0}},
    )

    assert result["status"] == "partial_verified"
    assert result["reason"] == "exhausted_attempts"
    assert result["verified_count"] == 1
    assert (mode_dir / "level4_runtime.json").exists()
    assert json.loads((mode_dir / "level4_runtime.json").read_text())["status"] == "partial_verified"


def test_verify_instance_marks_p2p_only_skips_with_specific_reason(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=1, end_line=2)
    candidate = {
        "status": "ok",
        "seed": 42,
        "runtime_seed": 42,
        "target_group": [{"path": "module.py", "function_name": "target", "start_line": 1, "end_line": 2}],
        "target_group_size": 1,
        "target_group_signature": "group:abc",
        "rewrite_summaries": {"module.py:1:target": "rewrite"},
        "behavioral_invariants": {"module.py:1:target": ["same"]},
        "rewrite_strength": 0.5,
        "rewritten_files": ["module.py"],
        "files": {"module.py": "def target(value):\n    return value + 2\n"},
        "level4_patch": "diff --git a/module.py b/module.py\nindex a..b 100644\n",
    }

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store._build_group_candidate", lambda **kwargs: candidate)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_official_preflight",
        lambda *args, **kwargs: {
            "status": "ok",
            "returncode": 0,
            "report": {
                "patch_successfully_applied": True,
                "tests_status": {"PASS_TO_PASS": {"failure": ["test_example"]}, "FAIL_TO_PASS": {"failure": []}},
            },
        },
    )

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "llm_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=1,
        max_attempts=2,
        target_variants=3,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "p2p_only"


def test_verify_instance_marks_local_gate_only_trivial_with_specific_reason(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=1, end_line=2)

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store._build_group_candidate",
        lambda **kwargs: {"status": "skipped", "reason": "rewrite_too_trivial"},
    )

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "llm_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=1,
        max_attempts=2,
        target_variants=3,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "local_gate_only_trivial"


def test_verify_instance_passes_recent_failure_feedback_to_next_attempt(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=1, end_line=2)
    seen_feedback = []

    def fake_build_group_candidate(**kwargs):
        seen_feedback.append(kwargs.get("failure_feedback"))
        if len(seen_feedback) == 1:
            return {"status": "skipped", "reason": "rewrite_too_trivial", "failed_target": "module.py:1:target"}
        return {
            "status": "ok",
            "seed": 42,
            "runtime_seed": 43,
            "target_group": [{"path": "module.py", "function_name": "target", "start_line": 1, "end_line": 2}],
            "target_group_size": 1,
            "target_group_signature": "group:abc",
            "rewrite_summaries": {"module.py:1:target": "rewrite"},
            "behavioral_invariants": {"module.py:1:target": ["same"]},
            "rewrite_strength": 0.5,
            "rewritten_files": ["module.py"],
            "files": {"module.py": "def target(value):\n    return value + 2\n"},
            "level4_patch": "diff --git a/module.py b/module.py\nindex a..b 100644\n",
        }

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store._build_group_candidate", fake_build_group_candidate)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_official_preflight",
        lambda *args, **kwargs: {
            "status": "ok",
            "returncode": 0,
            "report": {"patch_successfully_applied": True, "tests_status": {"PASS_TO_PASS": {"failure": []}}},
        },
    )

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "llm_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=1,
        max_attempts=2,
        target_variants=1,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
    )

    assert result["status"] == "verified"
    assert seen_feedback[0] is None
    assert "rewrite_too_trivial" in seen_feedback[1]


def test_verify_instance_passes_failed_pass_to_pass_test_sources_to_next_attempt(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "module.py").write_text(
        "def target(value):\n"
        "    return value + 1\n"
    )
    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(
        "def test_example():\n"
        "    assert True\n"
    )
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/module.py b/module.py\n"
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=1, end_line=2)
    seen_feedback = []

    def fake_build_group_candidate(**kwargs):
        seen_feedback.append(kwargs.get("failure_feedback"))
        if len(seen_feedback) == 1:
            return {
                "status": "ok",
                "seed": 42,
                "runtime_seed": 42,
                "target_group": [{"path": "module.py", "function_name": "target", "start_line": 1, "end_line": 2}],
                "target_group_size": 1,
                "target_group_signature": "group:first",
                "rewrite_summaries": {"module.py:1:target": "rewrite"},
                "behavioral_invariants": {"module.py:1:target": ["same"]},
                "rewrite_strength": 0.5,
                "rewritten_files": ["module.py"],
                "files": {"module.py": "def target(value):\n    return value + 2\n"},
                "level4_patch": "diff --git a/module.py b/module.py\nindex a..b 100644\n",
            }
        return {"status": "skipped", "reason": "rewrite_too_trivial", "failed_target": "module.py:1:target"}

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store._build_group_candidate", fake_build_group_candidate)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_official_preflight",
        lambda *args, **kwargs: {
            "status": "ok",
            "returncode": 0,
            "report": {
                "patch_successfully_applied": True,
                "tests_status": {"PASS_TO_PASS": {"failure": ["tests.test_sample.test_example"]}, "FAIL_TO_PASS": {"failure": []}},
            },
        },
    )

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "llm_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=1,
        max_attempts=2,
        target_variants=2,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "mixed_local_gate_and_p2p_fail"
    assert seen_feedback[0] is None
    assert "tests.test_sample.test_example" in seen_feedback[1]
    assert "tests/test_sample.py" in seen_feedback[1]
    assert "def test_example():" in seen_feedback[1]


def test_build_candidate_rejects_invalid_full_file_after_replacement(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "class Example:\n"
        "    def target(self):\n"
        "        return 1\n"
    )
    target = SimpleNamespace(path=Path("module.py"), function_name="target", start_line=2, end_line=3)

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.generate_level4b_rewrite",
        lambda **kwargs: {
            "rewritten_function_source": "def target(self):\n    return 2\n",
            "rewrite_summary": "changed return expression",
            "behavioral_invariants": ["same signature"],
        },
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store.validate_rewritten_function_body", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr("glasses.build_verified_level4b_store.rewrite_strength_gate", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr("glasses.build_verified_level4b_store.replace_function_in_source", lambda *args, **kwargs: "def broken(:\n")

    result = _build_candidate(
        repo_root=repo,
        target=target,
        seed=42,
        runtime_seed=42,
        model_name="fake/model",
        model_config={},
        work_dir=tmp_path / "work",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "syntax_error_after_replacement"


def test_rewrite_strength_gate_selects_same_named_nested_function():
    original = (
        "def sample(x):\n"
        "    return x + 1\n"
        "\n"
        "def outer(y):\n"
        "    def sample(z):\n"
        "        return z + 2\n"
        "    return sample(y)\n"
    )
    rewritten = (
        "def sample(x):\n"
        "    return x + 1\n"
        "\n"
        "def outer(y):\n"
        "    def sample(z):\n"
        "        tmp = z + 2\n"
        "        return tmp * 3\n"
        "    return sample(y)\n"
    )

    ok, reason = rewrite_strength_gate(original, rewritten, target=("sample", 5, 6))

    assert ok is True
    assert reason is None


def test_build_level4b_prompt_contains_strong_rewrite_constraints():
    prompt = build_level4b_prompt(
        path="pkg/mod.py",
        function_name="target",
        original_function_source="def target(value):\n    return value\n",
        enclosing_context="",
    )

    assert "substantially different implementation" in prompt
    assert "do not change the function name" in prompt
    assert "do not change arguments" in prompt
    assert "do not change decorators" in prompt
    assert "do not change the return annotation" in prompt
    assert "do not change imports" in prompt
    assert "do not modify code outside this function" in prompt


def test_build_level4b_prompt_includes_recent_failure_feedback():
    prompt = build_level4b_prompt(
        path="pkg/mod.py",
        function_name="target",
        original_function_source="def target(value):\n    return value\n",
        enclosing_context="",
        failure_feedback="- local failure: syntax_error\n- produce syntactically valid Python after the rewrite",
    )

    assert "Previous attempt failed. Avoid repeating these issues:" in prompt
    assert "local failure: syntax_error" in prompt


def test_build_level4b_prompt_embeds_failed_pass_to_pass_test_sources():
    prompt = build_level4b_prompt(
        path="pkg/mod.py",
        function_name="target",
        original_function_source="def target(value):\n    return value\n",
        enclosing_context="",
        failure_feedback=(
            "Previous attempt broke these previously passing tests. The next rewrite must keep them passing.\n"
            "Test: tests.test_sample.TestThing.test_value\n"
            "Path: tests/test_sample.py\n"
            "```python\n"
            "def test_value(self):\n"
            "    assert 1 == 1\n"
            "```"
        ),
    )

    assert "previously passing tests" in prompt
    assert "tests.test_sample.TestThing.test_value" in prompt
    assert "tests/test_sample.py" in prompt
    assert "def test_value(self):" in prompt


def test_extract_pass_to_pass_test_sources_returns_test_path_and_function_source(tmp_path):
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_sample.py").write_text(
        "class TestThing:\n"
        "    def test_value(self):\n"
        "        assert 1 == 1\n"
        "\n"
        "def test_top_level():\n"
        "    assert True\n"
    )

    snippets = extract_pass_to_pass_test_sources(repo, ["tests.test_sample.TestThing.test_value", "tests.test_sample.test_top_level"])

    assert snippets == [
        {
            "test_id": "tests.test_sample.TestThing.test_value",
            "path": "tests/test_sample.py",
            "source": "def test_value(self):\n    assert 1 == 1\n",
        },
        {
            "test_id": "tests.test_sample.test_top_level",
            "path": "tests/test_sample.py",
            "source": "def test_top_level():\n    assert True\n",
        },
    ]


def test_parse_level4b_response_accepts_fenced_json():
    content = """```json
    {
      "rewritten_function_source": "def target(value):\\n    return value + 1\\n",
      "rewrite_summary": "restructured control flow",
      "behavioral_invariants": ["same input", "same output"]
    }
    ```"""

    parsed = parse_level4b_response(content)

    assert parsed["rewritten_function_source"].startswith("def target")
    assert parsed["rewrite_summary"] == "restructured control flow"
    assert parsed["behavioral_invariants"] == ["same input", "same output"]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"rewritten_function_source": "def target():\\n    pass\\n"',
        '{"rewritten_function_source": 1, "rewrite_summary": "ok", "behavioral_invariants": []}',
        '{"rewritten_function_source": "def target():\\n    pass\\n", "rewrite_summary": 1, "behavioral_invariants": []}',
        '{"rewritten_function_source": "def target():\\n    pass\\n", "rewrite_summary": "ok", "behavioral_invariants": {}}',
        '{"rewrite_summary": "ok", "behavioral_invariants": []}',
    ],
)
def test_parse_level4b_response_rejects_malformed_or_invalid_payloads(content):
    with pytest.raises(ValueError):
        parse_level4b_response(content)


def test_generate_level4b_rewrite_uses_get_model_and_parses_json():
    fake_model = MagicMock()
    fake_model.config.model_name = "openai/gpt-5.4-mini"
    fake_model.config.model_kwargs = {"temperature": 1.0, "parallel_tool_calls": True}
    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(
            message=MagicMock(
                content=(
                    '{"rewritten_function_source":"def target(value):\\n'
                    '    if value is None:\\n'
                    '        return None\\n'
                    '    cleaned = normalize(value)\\n'
                    '    return cleaned\\n",'
                    '"rewrite_summary":"restructured control flow",'
                    '"behavioral_invariants":["same return semantics"]}'
                )
            )
        )
    ]

    with patch("glasses.level4_llm_function_body_rewrite.get_model", return_value=fake_model) as get_model_mock, patch(
        "glasses.level4_llm_function_body_rewrite.litellm.completion", return_value=fake_response
    ) as completion_mock:
        result = generate_level4b_rewrite(
            model_name="openai/gpt-5.4-mini",
            model_config={"model_kwargs": {"temperature": 1.0}},
            path="pkg/mod.py",
            function_name="target",
            original_function_source="def target(value):\n    return normalize(value)\n",
            enclosing_context="",
            failure_feedback="- local failure: rewrite_too_trivial",
        )

    prompt = build_level4b_prompt(
        path="pkg/mod.py",
        function_name="target",
        original_function_source="def target(value):\n    return normalize(value)\n",
        enclosing_context="",
        failure_feedback="- local failure: rewrite_too_trivial",
    )
    call = completion_mock.call_args
    assert get_model_mock.call_count == 1
    assert get_model_mock.call_args.args == ("openai/gpt-5.4-mini", {"model_kwargs": {"temperature": 1.0}})
    assert call.kwargs["model"] == fake_model.config.model_name
    assert call.kwargs["messages"] == [{"role": "user", "content": prompt}]
    assert call.kwargs["temperature"] == 1.0
    assert "parallel_tool_calls" not in call.kwargs
    assert result["rewrite_summary"] == "restructured control flow"
    assert "cleaned = normalize(value)" in result["rewritten_function_source"]


def test_replace_function_in_source_only_changes_target_function_in_nested_source():
    source = (
        "def keep():\n"
        "    return 1\n\n"
        "def target(value):\n"
        "    return normalize(value)\n\n"
        "def outer(flag):\n"
        "    def target(value):\n"
        "        return value + 1\n"
        "    return target(flag)\n"
    )
    rewritten = (
        "def target(value):\n"
        "    if value is None:\n"
        "        return None\n"
        "    cleaned = normalize(value)\n"
        "    return cleaned\n"
    )

    new_source = replace_function_in_source(source, "target", rewritten, target=("target", 4))

    assert "def keep():" in new_source
    assert "return 1" in new_source
    assert "def outer(flag):" in new_source
    assert "def target(value):\n        return value + 1" in new_source
    assert "cleaned = normalize(value)" in new_source


def test_replace_function_in_source_preserves_method_indent_when_rewrite_is_top_level():
    source = (
        "class Example:\n"
        "    def target(self, value):\n"
        "        return value + 1\n"
        "\n"
        "    def keep(self):\n"
        "        return 0\n"
    )
    rewritten = (
        "def target(self, value):\n"
        "    adjusted = value + 1\n"
        "    return adjusted\n"
    )

    new_source = replace_function_in_source(source, "target", rewritten, target=("target", 2))

    assert "\n    def target(self, value):\n" in new_source
    assert "\n        adjusted = value + 1\n" in new_source
    assert "\n    def keep(self):\n" in new_source


def test_replace_function_in_source_fails_closed_on_mismatched_target_tuple():
    source = (
        "def target(value):\n"
        "    return value + 1\n\n"
        "def outer(flag):\n"
        "    def target(value):\n"
        "        return value + 2\n"
        "    return target(flag)\n"
    )
    rewritten = (
        "def target(value):\n"
        "    return value + 99\n"
    )

    with pytest.raises(ValueError):
        replace_function_in_source(source, "target", rewritten, target=("target", 2))


def test_replace_function_in_source_replaces_decorated_span_once():
    source = (
        "@decorator_a\n"
        "@decorator_b\n"
        "def target(value):\n"
        "    return value + 1\n\n"
        "def keep():\n"
        "    return 0\n"
    )
    rewritten = (
        "@decorator_a\n"
        "@decorator_b\n"
        "def target(value):\n"
        "    tmp = value + 1\n"
        "    return tmp\n"
    )

    new_source = replace_function_in_source(source, "target", rewritten)

    assert new_source.count("@decorator_a") == 1
    assert new_source.count("@decorator_b") == 1
    assert new_source.count("def target(value):") == 1
    assert "tmp = value + 1" in new_source
    assert "def keep():" in new_source
