from pathlib import Path


def default_verified_arrow() -> Path:
    candidates = [
        Path(
            "/data/swebench/silinchen/hf-cache/datasets/princeton-nlp___swe-bench_verified/default/0.0.0/"
            "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/swe-bench_verified-test.arrow"
        ),
        Path(
            "/home/swebench/.cache/huggingface/datasets/princeton-nlp___swe-bench_verified/default/0.0.0/"
            "c104f840cc67f8b6eec6f759ebc8b2693d585d4a/swe-bench_verified-test.arrow"
        ),
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "princeton-nlp___swe-bench_verified"
        / "default"
        / "0.0.0"
        / "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
        / "swe-bench_verified-test.arrow",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
