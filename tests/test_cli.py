import json

from Spyctres.cli import main


def test_cli_lists_instruments_as_json(capsys):
    assert main(["instruments", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert "xshooter" in payload["instruments"]
    assert "sdss" in payload["instruments"]
    assert payload["include_aliases"] is False


def test_cli_reports_instrument_info_for_alias(capsys):
    assert main(["instrument-info", "x-shooter", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["canonical_name"] == "xshooter"
    assert "x-shooter" in payload["aliases"]
    assert payload["default_observer_frame"] == "topocentric"


def test_cli_inspects_spectrum_without_fitting(capsys):
    assert main(
        [
            "inspect-spectrum",
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
            "--instrument",
            "xshooter",
            "--json",
            "--no-warn-unknown",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["kind"] == "SpectrumSegment"
    assert payload["n_segments"] == 1
    assert payload["segments"][0]["n_pixels"] > 0
    assert payload["segments"][0]["wave_medium"] == "air"
