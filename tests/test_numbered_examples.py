import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


NUMBERED_EXAMPLES = (
    "example1_quickstart",
    "example2_lines_windows_and_masks",
    "example3_improving_a_phoenix_fit",
    "example4_reviewed_balmer_analysis",
    "example4b_balmer_stability_checks",
    "example5_batch_fitting",
    "example6_multiarm_classification",
    "example7_xsl_reference_validation",
    "example8_pepsi_legacy_linefit_validation",
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


def test_numbered_notebooks_have_shared_teaching_contract():
    required_markdown = (
        "What this example teaches",
        "Requirements",
        "Expected outputs",
    )
    scope_caveats = (
        "not a",
        "not an",
        "not automatically",
        "not final",
        "not treated as final",
        "not a validation",
        "scope",
        "diagnostic",
    )
    for stem in NUMBERED_EXAMPLES:
        path = EXAMPLES_DIR / f"{stem}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        for phrase in required_markdown:
            assert phrase in markdown
        assert any(phrase in markdown.lower() for phrase in scope_caveats)


def test_numbered_examples_do_not_use_local_external_test_paths():
    for stem in NUMBERED_EXAMPLES:
        paths = (
            EXAMPLES_DIR / f"{stem}.py",
            EXAMPLES_DIR / f"{stem}.ipynb",
        )
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

        assert "/home/" not in text
        assert "Data/Spectra" not in text


def test_numbered_notebooks_use_public_import_and_no_local_helpers():
    """Keep the beginner notebooks on the public ``import Spyctres as sp`` path."""
    deprecated_or_internal_terms = (
        "instrument=",
        "readiness_intent",
        "intended_use",
        "segment.mask",
        "setup.fit_kwargs",
        "from Spyctres.",
    )
    for stem in NUMBERED_EXAMPLES:
        path = EXAMPLES_DIR / f"{stem}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        assert "import Spyctres as sp" in code
        assert "\ndef " not in f"\n{code}"
        for term in deprecated_or_internal_terms:
            assert term not in code
        assert "reader=" in code


def test_example_documentation_uses_current_reader_and_example_names():
    docs = [
        REPO_ROOT / "README.md",
        EXAMPLES_DIR / "README.md",
        REPO_ROOT / "docs" / "development_plan.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    assert "target_id,path,instrument,R" not in text
    assert "quick_example.py" not in text
    assert "example6_microlensing_source_sed" not in text
    assert "examples/simple_phoenix_fit.py" not in text
    assert "examples/full_spectrum_classification.ipynb" not in text
    assert "examples/xshooter_multiarm_classification.ipynb" not in text
    assert "examples/xsl_figure1_validation.ipynb" not in text
    assert "examples/pepsi_legacy_linefit_validation.ipynb" not in text
    assert "examples/reviewed_xshooter_uvb_analysis.py" not in text
