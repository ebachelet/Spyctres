import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


NUMBERED_EXAMPLES = (
    "example1_quickstart",
    "example2_lines_windows_and_masks",
    "example3_improving_a_phoenix_fit",
    "example4_publication_quality_fitting",
    "example5_batch_fitting",
)


def test_numbered_example_script_and_notebook_pairs_exist():
    missing = []
    for stem in NUMBERED_EXAMPLES:
        for suffix in (".py", ".ipynb"):
            path = EXAMPLES_DIR / f"{stem}{suffix}"
            if not path.exists():
                missing.append(str(path.relative_to(REPO_ROOT)))

    assert missing == []


def test_numbered_notebooks_are_clean_source_notebooks():
    for stem in NUMBERED_EXAMPLES:
        path = EXAMPLES_DIR / f"{stem}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))

        assert notebook.get("nbformat") == 4
        assert notebook.get("cells")

        for cell in notebook["cells"]:
            assert cell.get("cell_type") in {"markdown", "code"}
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []


def test_numbered_examples_do_not_use_local_external_test_paths():
    for stem in NUMBERED_EXAMPLES:
        paths = (
            EXAMPLES_DIR / f"{stem}.py",
            EXAMPLES_DIR / f"{stem}.ipynb",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

        assert "/home/" not in text
        assert "Data/Spectra" not in text
