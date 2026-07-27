import csv
import json

import numpy as np

from Spyctres._serialization import (
    atomic_write_csv_rows,
    atomic_write_json,
    json_safe,
    safe_filename,
)


def test_json_safe_converts_numpy_and_nonfinite_values():
    payload = {
        "array": np.array([1.0, np.nan, np.float64(2.5)]),
        "scalar": np.int64(3),
        "nested": {"bad": float("inf")},
    }

    converted = json_safe(payload)

    assert converted == {
        "array": [1.0, None, 2.5],
        "scalar": 3,
        "nested": {"bad": None},
    }
    json.dumps(converted, allow_nan=False)


def test_atomic_json_and_csv_create_parents(tmp_path):
    json_path = tmp_path / "nested" / "payload.json"
    csv_path = tmp_path / "nested" / "payload.csv"

    atomic_write_json(json_path, {"x": np.array([1, 2])})
    atomic_write_csv_rows(csv_path, ["a", "b"], [{"a": 1, "b": 2, "extra": 3}])

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"x": [1, 2]}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "2"}]


def test_safe_filename_is_conservative():
    assert safe_filename("  HD 123/odd:name  ") == "HD_123_odd_name"
    assert safe_filename("", fallback="target") == "target"
