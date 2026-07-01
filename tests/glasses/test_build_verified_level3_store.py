from glasses.build_verified_level3_store import pass_to_pass_clean


def _result(*, patch_applied=True, p2p_failure=None, f2p_failure=None):
    return {
        "report": {
            "patch_successfully_applied": patch_applied,
            "tests_status": {
                "PASS_TO_PASS": {"failure": p2p_failure or []},
                "FAIL_TO_PASS": {"failure": f2p_failure or []},
            },
        }
    }


def test_accepts_only_when_p2p_passes_and_f2p_still_fails():
    assert pass_to_pass_clean(_result(f2p_failure=["test_bug"])) is True


def test_rejects_when_transform_solves_fail_to_pass_tests():
    assert pass_to_pass_clean(_result(f2p_failure=[])) is False


def test_rejects_when_fail_to_pass_status_is_missing():
    result = {
        "report": {
            "patch_successfully_applied": True,
            "tests_status": {"PASS_TO_PASS": {"failure": []}},
        }
    }
    assert pass_to_pass_clean(result) is False


def test_rejects_when_pass_to_pass_regresses():
    assert pass_to_pass_clean(_result(p2p_failure=["test_existing"], f2p_failure=["test_bug"])) is False
