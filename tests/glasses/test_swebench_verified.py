import json

from glasses.swebench_verified import parse_instance_test_lists


def test_parse_instance_test_lists_decodes_json_strings():
    instance = {
        "FAIL_TO_PASS": json.dumps(["pkg.tests.TestCase.test_fix", "pkg.tests.TestCase.test_more"]),
        "PASS_TO_PASS": json.dumps(["pkg.tests.TestCase.test_existing"]),
    }

    parsed = parse_instance_test_lists(instance)

    assert parsed == {
        "fail_to_pass": ["pkg.tests.TestCase.test_fix", "pkg.tests.TestCase.test_more"],
        "pass_to_pass": ["pkg.tests.TestCase.test_existing"],
    }


def test_parse_instance_test_lists_handles_missing_fields():
    parsed = parse_instance_test_lists({})

    assert parsed == {
        "fail_to_pass": [],
        "pass_to_pass": [],
    }
