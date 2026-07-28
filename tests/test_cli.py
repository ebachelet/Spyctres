import json

from Spyctres.cli import main


def test_cli_lists_readers_as_json(capsys):
    assert main(["readers", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert "xshooter_merge1d" in payload["readers"]
    assert "sdss_spec" in payload["readers"]
    assert payload["include_aliases"] is False


def test_cli_reports_reader_info_for_alias(capsys):
    assert main(["reader-info", "x-shooter", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["canonical_name"] == "xshooter_merge1d"
    assert "x-shooter" in payload["aliases"]
    assert payload["default_observer_frame"] == "topocentric"


def test_cli_inspects_spectrum_without_fitting(capsys):
    assert main(
        [
            "inspect-spectrum",
            "examples/data/TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits",
            "--reader",
            "xshooter_merge1d",
            "--json",
            "--no-warn-unknown",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["kind"] == "SpectrumSegment"
    assert payload["reader"] == "xshooter_merge1d"
    assert payload["n_segments"] == 1
    assert payload["segments"][0]["n_pixels"] > 0
    assert payload["segments"][0]["wave_medium"] == "air"


def test_cli_doctor_skip_phoenix(capsys):
    assert main(["doctor", "--skip-phoenix"]) == 0

    out = capsys.readouterr().out
    assert "Spyctres setup check passed." in out
    assert "PHOENIX checks skipped" in out


def test_cli_version_reports_prerelease(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    assert "Spyctres 0.5.0a1" in capsys.readouterr().out
