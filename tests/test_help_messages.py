import pytest

from Spyctres import (
    classify_spectrum,
    build_fit_collection_from_windows,
    classify_quality_flag,
    describe_public_function_group,
    describe_public_function,
    format_public_api_guide,
    fit_phoenix_spectrum,
    fit_stellar_spectrum,
    format_public_function_help,
    input_checksum_provenance,
    list_public_functions,
    list_public_function_groups,
    quality_flag_actions,
    read_spectrum,
    summarize_quality_flags,
)


def test_public_function_help_is_structured_and_formatted():
    topics = list_public_functions()

    assert "fit_stellar_spectrum" in topics
    assert "fit_line" in topics
    assert "list_known_lines" in topics
    assert "build_mask" in topics
    assert "plot_spectrum" in topics
    assert "plot_fit_referee" in topics
    assert "plot_fit_comparison_line_windows" in topics
    assert "plot_line_fit_comparison" in topics
    assert "annotate_nonstellar_features" in topics
    assert "diagnose_known_residual_windows" in topics
    assert "find_known_nonstellar_features" in topics
    assert "known_feature_masks" in topics
    assert "select_diagnostic_windows" in topics
    assert "build_fit_collection_from_windows" in topics
    assert "audit_spectrum_for_fit" in topics
    assert "compare_fits" in topics
    assert "summarize_quality_flags" in topics
    assert "quality_flag_actions" in topics
    assert "classify_quality_flag" in topics
    assert "suggest_fit_setup" in topics
    assert "input_checksum_provenance" in topics
    assert "readiness_flag_actions" in topics
    assert "read_spectrum" in topics

    record = describe_public_function("fit_stellar_spectrum")
    assert record["minimal_call"].startswith("fit_stellar_spectrum(")
    assert any(item["name"] == "spectrum" for item in record["required"])
    assert any("regions" in item["name"] for item in record["optional"])

    text = format_public_function_help("read_spectrum")
    assert "Minimal call:" in text
    assert "Optional extras:" in text
    assert "list_readers()" in text

    line_text = format_public_function_help("fit_line")
    assert 'fit_line(spec, "Hgamma")' in line_text
    line_list_text = format_public_function_help("list_known_lines")
    assert "List built-in local-line names" in line_list_text

    setup_text = format_public_function_help("suggest_fit_setup")
    assert "without running PHOENIX" in setup_text
    assert "suggest_fit_setup(spec)" in setup_text

    readiness_text = format_public_function_help("readiness_flag_actions")
    assert "Translate readiness-audit flags" in readiness_text

    checksum_text = format_public_function_help("input_checksum_provenance")
    assert "input-file checksum provenance" in checksum_text
    assert input_checksum_provenance is not None
    retained_text = format_public_function_help("build_fit_collection_from_windows")
    assert "fit-only SpectrumCollection" in retained_text
    assert build_fit_collection_from_windows is not None
    line_comparison_text = format_public_function_help("plot_line_fit_comparison")
    assert "diagnostic metrics" in line_comparison_text
    flag_text = format_public_function_help("summarize_quality_flags")
    assert "top user actions" in flag_text
    assert summarize_quality_flags is not None
    assert quality_flag_actions is not None
    assert classify_quality_flag is not None


def test_public_function_groups_expose_beginner_path_first():
    groups = list_public_function_groups()

    assert groups[0] == "beginner"
    assert "readiness_and_masks" in groups
    beginner = describe_public_function_group("quickstart")
    assert beginner["name"] == "beginner"
    assert beginner["functions"] == [
        "read_spectrum",
        "plot_spectrum",
        "suggest_fit_setup",
        "fit_stellar_spectrum",
        "plot_fit_referee",
    ]
    assert list_public_functions("readiness") == [
        "audit_spectrum_for_fit",
        "readiness_flag_actions",
        "select_diagnostic_windows",
        "build_fit_collection_from_windows",
        "plot_diagnostic_windows",
        "build_mask",
    ]
    for group in groups:
        for name in describe_public_function_group(group)["functions"]:
            assert describe_public_function(name)["name"] == name
    assert list_public_functions("line_diagnostics") == [
        "list_known_lines",
        "fit_line",
        "plot_line_fit",
        "plot_line_fit_comparison",
    ]


def test_public_api_guide_is_one_import_oriented():
    text = format_public_api_guide()

    assert "Recommended one-import path:" in text
    assert "import Spyctres as sp" in text
    assert "sp.fit_stellar_spectrum" in text
    assert "beginner: Beginner one-import path" in text
    assert "advanced: Advanced and compatibility entry points" in text

    readiness_text = format_public_api_guide("readiness")
    assert "readiness_and_masks: Readiness, masks, and diagnostic windows" in readiness_text
    assert "line_diagnostics" not in readiness_text


def test_unknown_public_function_help_lists_known_topics():
    with pytest.raises(ValueError, match="Known help topics"):
        describe_public_function("does_not_exist")


def test_read_spectrum_incomplete_call_explains_minimal_requirements(tmp_path):
    with pytest.raises(ValueError) as caught:
        read_spectrum()

    message = str(caught.value)
    assert "Minimal call: read_spectrum" in message
    assert "describe_public_function('read_spectrum')" in message

    spectrum_path = tmp_path / "spectrum.dat"
    spectrum_path.write_text("1 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        read_spectrum(spectrum_path)

    message = str(caught.value)
    assert "No spectrum reader was specified" in message
    assert "list_readers()" in message


def test_fit_entry_points_incomplete_calls_explain_minimal_requirements():
    for func, topic in (
        (fit_stellar_spectrum, "fit_stellar_spectrum"),
        (fit_phoenix_spectrum, "fit_phoenix_spectrum"),
        (classify_spectrum, "classify_spectrum"),
    ):
        with pytest.raises(ValueError) as caught:
            func()
        message = str(caught.value)
        assert "Minimal call: {0}".format(topic) in message
        assert "format_public_function_help('{0}')".format(topic) in message
