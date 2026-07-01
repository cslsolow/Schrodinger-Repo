from glasses.level4_agentic_function_body_rewrite import (
    FULL_TARGET_GROUP_SCOPE,
    GOLDEN_PATCH_LOCAL_SCOPE,
    apply_agentic_submission_patch,
    build_agentic_rewrite_task,
    collect_agentic_changed_files,
    extract_agentic_changed_paths,
    get_agentic_pass_to_pass_tests,
    run_mini_agentic_rewrite,
)
from glasses.build_verified_level4b_store import _build_agentic_group_candidate, verify_instance
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_build_agentic_rewrite_task_includes_target_group_and_p2p_list():
    task = build_agentic_rewrite_task(
        instance_id="demo__repo-1",
        target_group=[
            {"path": "pkg/mod.py", "function_name": "target", "start_line": 10, "end_line": 18},
        ],
        target_sources={
            "pkg/mod.py:10:target": {
                "original_function_source": "def target(value):\n    return value + 1\n",
                "enclosing_context": "class Example:\n    def target(value):\n        return value + 1\n",
            }
        },
        pass_to_pass_tests=["tests.test_mod.test_target", "tests.test_mod.test_other"],
    )

    assert "target_group" in task
    assert "tests.test_mod.test_target" in task
    assert "First locate the target code" in task
    assert "Work around the listed PASS_TO_PASS tests" in task


def test_build_agentic_rewrite_task_supports_golden_patch_local_scope():
    task = build_agentic_rewrite_task(
        instance_id="demo__repo-1",
        target_group=[
            {"path": "pkg/mod.py", "function_name": "target", "start_line": 10, "end_line": 18},
        ],
        target_sources={
            "pkg/mod.py:10:target": {
                "original_function_source": "def target(value):\n    return value + 1\n",
                "enclosing_context": "class Example:\n    def target(value):\n        return value + 1\n",
            }
        },
        pass_to_pass_tests=["tests.test_mod.test_target"],
        rewrite_scope=GOLDEN_PATCH_LOCAL_SCOPE,
        golden_patch_local_context={"pkg/mod.py": "@@ local context 10-12 @@\nreturn value + 1\n"},
    )

    assert "produce a local, conservative, behavior-preserving patch" in task
    assert "inspect the listed PASS_TO_PASS tests" in task
    assert "fall back to safer local transformations" in task
    assert "introducing local temporary variables" in task
    assert "renaming local variables" in task
    assert "Do not change function signatures, decorators, imports" in task
    assert "valid unified diff that can be applied cleanly with git apply" in task
    assert '"rewrite_scope": "golden_patch_local"' in task
    assert '"golden_patch_local_context"' in task


def test_build_agentic_rewrite_task_includes_previous_attempt_feedback():
    task = build_agentic_rewrite_task(
        instance_id="demo__repo-1",
        target_group=[
            {"path": "pkg/mod.py", "function_name": "target", "start_line": 10, "end_line": 18},
        ],
        target_sources={
            "pkg/mod.py:10:target": {
                "original_function_source": "def target(value):\n    return value + 1\n",
                "enclosing_context": "class Example:\n    def target(value):\n        return value + 1\n",
            }
        },
        pass_to_pass_tests=["tests.test_mod.test_target"],
        rewrite_scope=GOLDEN_PATCH_LOCAL_SCOPE,
        previous_attempt_feedback={
            "previous_submission_patch": "diff --git a/pkg/mod.py b/pkg/mod.py",
            "pass_to_pass_failure": ["tests.test_mod.test_target"],
        },
    )

    assert "If previous_attempt_feedback is present" in task
    assert '"previous_attempt_feedback"' in task
    assert 'diff --git a/pkg/mod.py b/pkg/mod.py' in task


def test_collect_agentic_changed_files_filters_to_python_paths(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/mod.py").write_text("def target():\n    return 2\n")
    (repo / "README.md").write_text("after\n")

    files = collect_agentic_changed_files(repo, ["pkg/mod.py", "README.md", "missing.py"])

    assert files == {"pkg/mod.py": "def target():\n    return 2\n"}


def test_extract_agentic_changed_paths_reads_submission_patch():
    patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def target():\n"
        "-    return 1\n"
        "+    return 2\n"
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
    )

    assert extract_agentic_changed_paths(patch) == ["pkg.py", "README.md"]


def test_run_mini_agentic_rewrite_uses_inprocess_mini_agent_stack(tmp_path, monkeypatch):
    output_path = tmp_path / "agent.traj.json"
    seen = {}
    fake_agent = MagicMock()

    def fake_run(task):
        seen["task"] = task
        return {"submission": ""}

    fake_agent.run.side_effect = fake_run

    monkeypatch.setattr(
        "glasses.level4_agentic_function_body_rewrite.get_config_from_spec",
        lambda spec: {"agent": {}, "model": {}, "environment": {}},
    )
    monkeypatch.setattr("glasses.level4_agentic_function_body_rewrite.get_model", lambda config: "model")
    monkeypatch.setattr("glasses.level4_agentic_function_body_rewrite.get_sb_environment", lambda config, instance: "env")
    monkeypatch.setattr("glasses.level4_agentic_function_body_rewrite.get_agent", lambda model, env, config, default_type='': fake_agent)

    result = run_mini_agentic_rewrite(
        instance={"instance_id": "demo"},
        repo_root=tmp_path / "repo",
        task="rewrite this code",
        model_name="openai/gpt-5.4-mini",
        output_path=output_path,
        config_path=tmp_path / "mini.yaml",
    )

    assert seen["task"] == "rewrite this code"
    assert result == {"submission": "", "trajectory_path": str(output_path)}


def test_get_agentic_pass_to_pass_tests_uses_instance_pass_to_pass_field():
    tests = get_agentic_pass_to_pass_tests(
        {
            "PASS_TO_PASS": "[\"pkg.tests.TestCase.test_existing\", \"pkg.tests.TestCase.test_more\"]",
            "FAIL_TO_PASS": "[]",
        }
    )

    assert tests == ["pkg.tests.TestCase.test_existing", "pkg.tests.TestCase.test_more"]


def test_apply_agentic_submission_patch_applies_git_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess_run = __import__("subprocess").run
    subprocess_run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    target = repo / "pkg.py"
    target.write_text("def target():\n    return 1\n")
    patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def target():\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    apply_agentic_submission_patch(repo, patch, tmp_path / "submission.patch")

    assert target.read_text() == "def target():\n    return 2\n"


def test_apply_agentic_submission_patch_keeps_patch_and_git_apply_logs_on_failure(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess_run = __import__("subprocess").run
    subprocess_run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "pkg.py").write_text("def target():\n    return 1\n")
    patch_path = tmp_path / "submission.patch"
    patch = (
        "diff --git a/pkg.py b/pkg.py\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -10,2 +10,2 @@\n"
        " def target():\n"
        "-    return 0\n"
        "+    return 2\n"
    )

    try:
        apply_agentic_submission_patch(repo, patch, patch_path)
    except Exception:
        pass

    assert patch_path.read_text() == patch
    assert (tmp_path / "git_apply_stdout.txt").exists()
    assert (tmp_path / "git_apply_stderr.txt").exists()


def test_build_agentic_group_candidate_returns_existing_candidate_shape(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg.py").write_text("def target(value):\n    return value + 1\n")

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_mini_agentic_rewrite",
        lambda **kwargs: {
            "submission": "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n@@ -1,2 +1,3 @@\n def target(value):\n-    return value + 1\n+    adjusted = value + 1\n+    return adjusted\n",
        },
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store.apply_agentic_submission_patch", lambda repo_root, submission, patch_path: None)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.collect_agentic_changed_files",
        lambda repo_root, changed_paths: {"pkg.py": "def target(value):\n    adjusted = value + 1\n    return adjusted\n"},
    )

    candidate = _build_agentic_group_candidate(
        instance={"instance_id": "demo__repo-1"},
        instance_id="demo__repo-1",
        repo_root=repo,
        target_group=[SimpleNamespace(path=Path("pkg.py"), function_name="target", start_line=1, end_line=2)],
        seed=42,
        runtime_seed=42,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
        work_dir=tmp_path / "work",
        failure_feedback=None,
        rewrite_scope=FULL_TARGET_GROUP_SCOPE,
    )

    assert candidate["status"] == "ok"
    assert candidate["rewritten_files"] == ["pkg.py"]
    assert candidate["target_group_size"] == 1
    assert candidate["rewrite_strength"] is not None


def test_build_agentic_group_candidate_allows_missing_rewritten_function_for_strength(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg.py").write_text("def target(value):\n    return value + 1\n")

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.run_mini_agentic_rewrite",
        lambda **kwargs: {
            "submission": "diff --git a/pkg.py b/pkg.py\n--- a/pkg.py\n+++ b/pkg.py\n@@ -1,2 +1,4 @@\n+def helper():\n+    return 0\n+\n def target(value):\n     return value + 1\n",
        },
    )
    monkeypatch.setattr("glasses.build_verified_level4b_store.apply_agentic_submission_patch", lambda repo_root, submission, patch_path: None)
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.collect_agentic_changed_files",
        lambda repo_root, changed_paths: {"pkg.py": "def helper():\n    return 0\n\ndef target(value):\n    return value + 1\n"},
    )

    candidate = _build_agentic_group_candidate(
        instance={"instance_id": "demo__repo-1"},
        instance_id="demo__repo-1",
        repo_root=repo,
        target_group=[SimpleNamespace(path=Path("pkg.py"), function_name="target", start_line=1, end_line=2)],
        seed=42,
        runtime_seed=42,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 1.0}},
        work_dir=tmp_path / "work",
        failure_feedback=None,
        rewrite_scope=FULL_TARGET_GROUP_SCOPE,
    )

    assert candidate["status"] == "ok"
    assert candidate["rewrite_strength"] is None


def test_verify_instance_agentic_backend_runs_exactly_three_attempts(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pkg.py").write_text("def target(value):\n    return value + 1\n")
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/pkg.py b/pkg.py\n"
            "--- a/pkg.py\n"
            "+++ b/pkg.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("pkg.py"), function_name="target", start_line=1, end_line=2)
    calls = []

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store._build_agentic_group_candidate",
        lambda **kwargs: calls.append(kwargs["runtime_seed"]) or {"status": "skipped", "reason": "syntax_error"},
    )

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "agentic_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=3,
        max_attempts=99,
        target_variants=3,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 0.0}},
        mode="agentic_function_body_rewrite",
        agentic_rewrite_scope=FULL_TARGET_GROUP_SCOPE,
    )

    assert calls == [42, 43, 44]
    assert result["status"] == "skipped"


def test_verify_instance_skipped_keeps_attempt_artifacts_for_preflight_failures(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pkg.py").write_text("def target(value):\n    return value + 1\n")
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/pkg.py b/pkg.py\n"
            "--- a/pkg.py\n"
            "+++ b/pkg.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("pkg.py"), function_name="target", start_line=1, end_line=2)

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store._build_agentic_group_candidate",
        lambda **kwargs: (
            (kwargs["work_dir"] / "submission.patch").parent.mkdir(parents=True, exist_ok=True),
            (kwargs["work_dir"] / "submission.patch").write_text("diff --git a/pkg.py b/pkg.py\n"),
            {
                "status": "ok",
                "runtime_seed": kwargs["runtime_seed"],
                "target_group": [{"path": "pkg.py", "function_name": "target", "start_line": 1, "end_line": 2}],
                "target_group_size": 1,
                "target_group_signature": "sig",
                "rewrite_summaries": {"pkg.py:1:target": "agentic rewrite"},
                "behavioral_invariants": {"pkg.py:1:target": ["same behavior"]},
                "rewrite_strength": 0.8,
                "rewritten_files": ["pkg.py"],
                "files": {"pkg.py": "def target(value):\n    adjusted = value + 1\n    return adjusted\n"},
                "level4_patch": "diff --git a/pkg.py b/pkg.py\n",
                "generator_backend": "mini_swe_agent",
            },
        )[-1],
    )

    def fake_preflight(instance, overlay_root, rewritten_files, level4_patch, log_dir, timeout=None):
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "report.json").write_text('{"demo": "report"}')
        (log_dir / "test_output.txt").write_text("failed tests")
        return {
            "status": "runtime_broken",
            "returncode": 1,
            "report": {
                "patch_successfully_applied": True,
                "tests_status": {
                    "PASS_TO_PASS": {"failure": ["pkg.tests.test_target"]},
                    "FAIL_TO_PASS": {"failure": []},
                },
            },
        }

    monkeypatch.setattr("glasses.build_verified_level4b_store.run_official_preflight", fake_preflight)

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "agentic_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=3,
        max_attempts=3,
        target_variants=1,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 0.0}},
        mode="agentic_function_body_rewrite",
        agentic_rewrite_scope=FULL_TARGET_GROUP_SCOPE,
    )

    assert result["status"] == "skipped"
    artifacts = result["diagnostics"][0]["artifacts"]
    assert artifacts["submission_patch"].endswith("submission.patch")
    assert artifacts["preflight_report"].endswith("report.json")
    assert artifacts["preflight_test_output"].endswith("test_output.txt")


def test_verify_instance_verified_keeps_attempt_artifacts(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pkg.py").write_text("def target(value):\n    return value + 1\n")
    instance = {
        "instance_id": "demo",
        "patch": (
            "diff --git a/pkg.py b/pkg.py\n"
            "--- a/pkg.py\n"
            "+++ b/pkg.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def target(value):\n"
            "-    return value + 1\n"
            "+    return value + 2\n"
        ),
    }
    target = SimpleNamespace(path=Path("pkg.py"), function_name="target", start_line=1, end_line=2)

    monkeypatch.setattr(
        "glasses.build_verified_level4b_store.extract_rewrite_targets_from_golden_patch",
        lambda patch, repo: ([target], None),
    )
    monkeypatch.setattr(
        "glasses.build_verified_level4b_store._build_agentic_group_candidate",
        lambda **kwargs: (
            (kwargs["work_dir"] / "submission.patch").parent.mkdir(parents=True, exist_ok=True),
            (kwargs["work_dir"] / "submission.patch").write_text("diff --git a/pkg.py b/pkg.py\n"),
            (kwargs["work_dir"] / "agent.traj.json").write_text("{}"),
            {
                "status": "ok",
                "runtime_seed": kwargs["runtime_seed"],
                "target_group": [{"path": "pkg.py", "function_name": "target", "start_line": 1, "end_line": 2}],
                "target_group_size": 1,
                "target_group_signature": "sig",
                "rewrite_summaries": {"pkg.py:1:target": "agentic rewrite"},
                "behavioral_invariants": {"pkg.py:1:target": ["same behavior"]},
                "rewrite_strength": 0.8,
                "rewritten_files": ["pkg.py"],
                "files": {"pkg.py": "def target(value):\n    adjusted = value + 1\n    return adjusted\n"},
                "level4_patch": "diff --git a/pkg.py b/pkg.py\n",
                "generator_backend": "mini_swe_agent",
            },
        )[-1],
    )

    def fake_preflight(instance, overlay_root, rewritten_files, level4_patch, log_dir, timeout=None):
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "report.json").write_text('{"demo": "report"}')
        (log_dir / "test_output.txt").write_text("all passed")
        return {
            "status": "ok",
            "returncode": 0,
            "report": {
                "patch_successfully_applied": True,
                "tests_status": {
                    "PASS_TO_PASS": {"failure": []},
                    "FAIL_TO_PASS": {"failure": ["pkg.tests.test_bug_still_fails"]},
                },
            },
        }

    monkeypatch.setattr("glasses.build_verified_level4b_store.run_official_preflight", fake_preflight)

    result = verify_instance(
        instance,
        repo_root,
        tmp_path / "output" / "demo" / "agentic_function_body_rewrite",
        tmp_path / "tmp",
        seed=42,
        variant_count=3,
        max_attempts=3,
        target_variants=1,
        redo=True,
        model_name="openai/gpt-5.4-mini",
        model_config={"model_kwargs": {"temperature": 0.0}},
        mode="agentic_function_body_rewrite",
        agentic_rewrite_scope=FULL_TARGET_GROUP_SCOPE,
    )

    assert result["status"] == "verified"
    artifacts = result["variants"][0]["artifacts"]
    assert artifacts["submission_patch"].endswith("submission.patch")
    assert artifacts["trajectory"].endswith("agent.traj.json")
    assert artifacts["preflight_report"].endswith("report.json")
    assert artifacts["preflight_test_output"].endswith("test_output.txt")
