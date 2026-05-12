import json
from pathlib import Path

from datasets import Dataset

try:
    from glasses.verified_dataset_paths import default_verified_arrow
except ModuleNotFoundError:
    from verified_dataset_paths import default_verified_arrow

DEFAULT_VERIFIED_ARROW = default_verified_arrow()


def load_verified_instance(instance_id: str, arrow_path: Path = DEFAULT_VERIFIED_ARROW) -> dict:
    dataset = Dataset.from_file(str(arrow_path))
    for instance in dataset:
        if instance.get("instance_id") == instance_id:
            return dict(instance)
    raise KeyError(f"Instance {instance_id} not found in {arrow_path}")


def parse_instance_test_lists(instance: dict) -> dict[str, list[str]]:
    def parse_field(name: str) -> list[str]:
        value = instance.get(name)
        if not value:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [str(item) for item in json.loads(value)]
        raise TypeError(f"Unsupported {name} type: {type(value).__name__}")

    return {
        "fail_to_pass": parse_field("FAIL_TO_PASS"),
        "pass_to_pass": parse_field("PASS_TO_PASS"),
    }
