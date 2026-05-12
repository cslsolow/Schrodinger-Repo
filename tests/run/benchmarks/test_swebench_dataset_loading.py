from unittest.mock import patch

from minisweagent.run.benchmarks.swebench import load_swebench_instances


def test_load_swebench_instances_uses_local_verified_arrow(monkeypatch):
    fake_rows = [{"instance_id": "demo", "problem_statement": "x"}]

    class FakeDataset:
        @staticmethod
        def from_file(path):
            assert path.endswith("swe-bench_verified-test.arrow")
            return fake_rows

    monkeypatch.setattr("minisweagent.run.benchmarks.swebench.default_verified_arrow", lambda: __import__("pathlib").Path("/tmp/swe-bench_verified-test.arrow"))
    with (
        patch("datasets.load_dataset") as load_dataset,
        patch("datasets.Dataset", FakeDataset),
        patch("pathlib.Path.exists", return_value=True),
    ):
        rows = load_swebench_instances("princeton-nlp/SWE-Bench_Verified", "test")

    assert rows == fake_rows
    load_dataset.assert_not_called()
