import json

from glasses.docker_verifier import (
    build_module_path_aliases,
    build_instance_test_command,
    build_official_python_eval_script,
    classify_official_eval_output,
    classify_verification_result,
    extract_changed_test_paths,
    get_official_python_test_directives,
)


def test_extract_changed_test_paths_reads_patch_headers():
    patch = (
        "diff --git a/tests/validators/invalid_urls.txt b/tests/validators/invalid_urls.txt\n"
        "+++ b/tests/validators/invalid_urls.txt\n"
        "diff --git a/tests/validators/valid_urls.txt b/tests/validators/valid_urls.txt\n"
        "+++ b/tests/validators/valid_urls.txt\n"
        "diff --git a/django/core/validators.py b/django/core/validators.py\n"
        "+++ b/django/core/validators.py\n"
    )

    assert extract_changed_test_paths(patch) == [
        "tests/validators/invalid_urls.txt",
        "tests/validators/valid_urls.txt",
    ]


def test_build_instance_test_command_uses_django_runtests_labels():
    instance = {
        "repo": "django/django",
        "test_patch": (
            "diff --git a/tests/validators/invalid_urls.txt b/tests/validators/invalid_urls.txt\n"
            "+++ b/tests/validators/invalid_urls.txt\n"
            "diff --git a/tests/validators/valid_urls.txt b/tests/validators/valid_urls.txt\n"
            "+++ b/tests/validators/valid_urls.txt\n"
        ),
    }

    assert build_instance_test_command(instance) == "python tests/runtests.py validators --verbosity 0 --parallel=1"


def test_build_instance_test_command_falls_back_to_pytest_paths():
    instance = {
        "repo": "matplotlib/matplotlib",
        "test_patch": (
            "diff --git a/lib/matplotlib/foo.py b/lib/matplotlib/foo.py\n"
            "+++ b/lib/matplotlib/foo.py\n"
            "diff --git a/lib/matplotlib/tests/test_foo.py b/lib/matplotlib/tests/test_foo.py\n"
            "+++ b/lib/matplotlib/tests/test_foo.py\n"
        ),
    }

    assert build_instance_test_command(instance) == "python -m pytest lib/matplotlib/tests/test_foo.py -q"


def test_classify_verification_result_marks_import_error_as_runtime_broken():
    result = {
        "status": "test_failed",
        "output": "ImportError: No module named 'django.apps.validators'\n",
    }

    classified = classify_verification_result(result)

    assert classified["status"] == "runtime_broken"
    assert classified["failure_kind"] == "import_error"


def test_classify_verification_result_marks_assertion_failures_as_runtime_ok():
    result = {
        "status": "test_failed",
        "output": "AssertionError: ValidationError not raised when validating 'http://foo/bar@example.com'\n",
    }

    classified = classify_verification_result(result)

    assert classified["status"] == "runtime_ok_tests_failed"
    assert classified["failure_kind"] == "test_assertion_failure"


def test_classify_verification_result_prefers_test_failure_over_traceback_noise():
    result = {
        "status": "test_failed",
        "output": "FAIL: test_x\nTraceback (most recent call last):\nAssertionError: boom\nFAILED (failures=1)\n",
    }

    classified = classify_verification_result(result)

    assert classified["status"] == "runtime_ok_tests_failed"
    assert classified["failure_kind"] == "test_assertion_failure"


def test_build_module_path_aliases_uses_container_testbed_prefix():
    layout_map = {
        "path_map": {
            "django/forms/widgets.py": "django/assembly/widgets.py",
            "django/forms/renderers.py": "django/assembly/renderers.py",
        }
    }

    aliases = build_module_path_aliases(layout_map)

    assert aliases == {
        "django.forms.widgets": "/testbed/django/assembly/widgets.py",
        "django.forms.renderers": "/testbed/django/assembly/renderers.py",
    }


def test_classify_official_eval_output_detects_patch_failure():
    result = classify_official_eval_output(
        ">>>>> Patch Apply Failed\nsome git apply error\n",
        returncode=1,
    )

    assert result["status"] == "runtime_broken"
    assert result["failure_kind"] == "patch_or_reset_failure"


def test_classify_official_eval_output_detects_runtime_ok_test_failure():
    output = (
        ">>>>> Start Test Output\n"
        "FAIL: test_example\n"
        "AssertionError: boom\n"
        ">>>>> End Test Output\n"
    )
    result = classify_official_eval_output(output, returncode=1)

    assert result["status"] == "runtime_ok_tests_failed"
    assert result["failure_kind"] == "test_assertion_failure"


def test_classify_official_eval_output_ignores_zero_returncode_if_test_output_failed():
    output = (
        ">>>>> Start Test Output\n"
        "FAILED something\n"
        ">>>>> End Test Output\n"
    )
    result = classify_official_eval_output(output, returncode=0)

    assert result["status"] == "runtime_ok_tests_failed"


def test_get_official_python_test_directives_matches_django_style():
    instance = {
        "repo": "django/django",
        "test_patch": (
            "diff --git a/tests/forms_tests/test_a.py b/tests/forms_tests/test_a.py\n"
            "+++ b/tests/forms_tests/test_a.py\n"
            "diff --git a/tests/forms_tests/data.txt b/tests/forms_tests/data.txt\n"
            "+++ b/tests/forms_tests/data.txt\n"
        ),
    }

    assert get_official_python_test_directives(instance) == ["forms_tests.test_a"]


def test_build_official_python_eval_script_includes_test_patch_and_markers():
    instance = {
        "repo": "django/django",
        "version": "2.2",
        "base_commit": "abc123",
        "test_patch": (
            "diff --git a/tests/forms_tests/test_a.py b/tests/forms_tests/test_a.py\n"
            "+++ b/tests/forms_tests/test_a.py\n"
        ),
    }

    script = build_official_python_eval_script(instance, runtime_alias_prefix="export PYTHONPATH=/shim:$PYTHONPATH")

    assert "conda activate testbed" in script
    assert "git checkout abc123 tests/forms_tests/test_a.py" in script
    assert "git apply -v - <<" in script
    assert ">>>>> Start Test Output" in script
    assert ">>>>> End Test Output" in script
