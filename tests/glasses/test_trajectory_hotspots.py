import json

from glasses.trajectory_hotspots import extract_hotspot_evidence, summarize_hotspots


def test_extract_hotspot_evidence_collects_edit_read_search_and_stack(tmp_path):
    traj_path = tmp_path / "demo__demo-1.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                                {"command": "cd /testbed && grep -R \"URLValidator\" -n django/core/validators.py tests/validators"},
                            ]
                        },
                    },
                    {
                        "role": "tool",
                        "content": "<output>\ndjango/core/validators.py:74:class URLValidator\n"
                        'Traceback\n  File "/testbed/django/core/checks/security/base.py", line 1, in <module>\n'
                        "</output>",
                    },
                    {
                        "role": "exit",
                        "content": "diff --git a/django/core/validators.py b/django/core/validators.py\n"
                        "+++ b/django/core/validators.py\n",
                    },
                ]
            }
        )
    )

    evidence = extract_hotspot_evidence(traj_path)

    assert evidence == {
        "edited": ["django/core/validators.py"],
        "read": ["django/core/validators.py", "django/core/validators.py"],
        "search": ["django/core/validators.py"],
        "stack": ["django/core/checks/security/base.py"],
    }


def test_extract_hotspot_evidence_filters_protected_paths(tmp_path):
    traj_path = tmp_path / "demo__demo-2.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {
                                    "command": "cd /testbed && sed -n '1,100p' tests/validators/test_custom.py "
                                    "django/core/validators.py django/db/migrations/graph.py"
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )

    evidence = extract_hotspot_evidence(traj_path)

    assert evidence == {"read": ["django/core/validators.py"]}


def test_summarize_hotspots_scores_and_directory_rollup(tmp_path):
    traj_path = tmp_path / "django__django-10097.traj.json"
    traj_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "actions": [
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                                {"command": "cd /testbed && sed -n '1,200p' django/core/validators.py"},
                            ]
                        },
                    },
                    {
                        "role": "tool",
                        "content": "django/core/validators.py:74:class URLValidator\n"
                        'File "/testbed/django/core/checks/security/base.py", line 1, in <module>\n',
                    },
                    {
                        "role": "exit",
                        "content": "+++ b/django/core/validators.py\n",
                    },
                ]
            }
        )
    )

    summary = summarize_hotspots(traj_path)

    assert summary["instance_id"] == "django__django-10097"
    assert summary["files"][0] == {
        "path": "django/core/validators.py",
        "score": 9.5,
        "evidence": {"edited": 1, "read": 2, "search": 1},
    }
    assert summary["directories"][0] == {"path": "django/core", "score": 9.5}
