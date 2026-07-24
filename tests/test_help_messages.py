import pytest

from Spyctres import (
    classify_spectrum,
    describe_public_function,
    fit_phoenix_spectrum,
    fit_stellar_spectrum,
    format_public_function_help,
    list_public_functions,
    read_spectrum,
)


def test_public_function_help_is_structured_and_formatted():
    topics = list_public_functions()

    assert "fit_stellar_spectrum" in topics
    assert "fit_line" in topics
    assert "build_mask" in topics
    assert "plot_spectrum" in topics
    assert "read_spectrum" in topics

    record = describe_public_function("fit_stellar_spectrum")
    assert record["minimal_call"].startswith("fit_stellar_spectrum(")
    assert any(item["name"] == "spectrum" for item in record["required"])
    assert any("regions" in item["name"] for item in record["optional"])

    text = format_public_function_help("read_spectrum")
    assert "Minimal call:" in text
    assert "Optional extras:" in text
    assert "list_instruments()" in text

    line_text = format_public_function_help("fit_line")
    assert 'fit_line(spec, "Hgamma")' in line_text


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
    assert "No instrument reader was specified" in message
    assert "list_instruments()" in message


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
