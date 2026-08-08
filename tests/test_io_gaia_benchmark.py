import gzip

import numpy as np
import pytest

from Spyctres import read_gaia_benchmark_ascii
from Spyctres.io import get_reader_info, read_spectrum


def _write_gzip_text(path, text):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_gaia_benchmark_gzip_nm_flux_error_parsing(tmp_path):
    path = tmp_path / "HIP79672_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(
        path,
        "waveobs\tflux\terr\n"
        "480.0\t0.91\t0.01\n"
        "480.1\t0.92\t0.02\n",
    )

    segment = read_gaia_benchmark_ascii(path)

    assert np.allclose(segment.wave, [4800.0, 4801.0])
    assert np.allclose(segment.flux, [0.91, 0.92])
    assert np.allclose(segment.err, [0.01, 0.02])
    assert np.array_equal(segment.mask, [True, True])
    assert segment.meta["wave_unit_input"] == "nm"
    assert segment.meta["benchmark_star_id"] == "HIP79672"
    assert segment.meta["source_instrument"] == "HARPS"
    assert segment.meta["source_spectrum_index"] == 1
    assert segment.wave_medium == "air"


def test_gaia_benchmark_masks_bad_flux_and_error(tmp_path):
    path = tmp_path / "HIP69673_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(
        path,
        "waveobs flux err\n"
        "480.0 1.0 0.01\n"
        "480.1 nan 0.01\n"
        "480.2 0.9 0.0\n"
        "480.3 0.8 nan\n",
    )

    segment = read_gaia_benchmark_ascii(path)

    assert np.array_equal(segment.mask, [True, False, False, False])


def test_gaia_benchmark_can_ignore_error_column_explicitly(tmp_path):
    path = tmp_path / "HIP76976_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(path, "480.0 1.0 0.0\n480.1 1.1 nan\n")

    segment = read_gaia_benchmark_ascii(path, err_column=None)

    assert segment.err is None
    assert np.array_equal(segment.mask, [True, True])


def test_gaia_benchmark_metadata_records_resolution_and_frames(tmp_path):
    path = tmp_path / "HIP37279_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(path, "480.0 1.0 0.01\n480.1 1.1 0.02\n")

    segment = read_gaia_benchmark_ascii(path)

    assert segment.wave_medium == "air"
    assert segment.observer_frame == "barycentric"
    assert segment.stellar_rest_status == "corrected"
    assert segment.meta["wave_medium"] == "air"
    assert "line-center comparison" in segment.meta["wave_medium_source"]
    assert segment.meta["observer_frame"] == "barycentric"
    assert segment.meta["stellar_rest_status"] == "corrected"
    assert segment.meta["flux_state"] == "continuum-normalized"
    assert segment.meta["resolution_R"] == pytest.approx(42000.0)
    assert segment.resolution.quantity == "R"
    assert segment.resolution.value == pytest.approx(42000.0)


@pytest.mark.parametrize("medium", ["unknown", "vacuum"])
def test_gaia_benchmark_wave_medium_can_be_overridden(medium, tmp_path):
    path = tmp_path / "HIP37279_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(path, "480.0 1.0 0.01\n480.1 1.1 0.02\n")

    segment = read_gaia_benchmark_ascii(path, wave_medium=medium)

    assert segment.wave_medium == medium
    assert segment.meta["wave_medium"] == medium


@pytest.mark.parametrize(
    "alias",
    [
        "gaia_benchmark",
        "gbs_v3_ascii",
        "gaia-benchmark",
        "gbs",
        "gbs_v3",
        "fgk_benchmark",
        "gaia_fgk_benchmark",
    ],
)
def test_gaia_benchmark_reader_aliases(alias, tmp_path):
    path = tmp_path / "HIP79672_HARPS_1_R42KNorm.txt.gz"
    _write_gzip_text(path, "480.0 1.0 0.01\n480.1 1.1 0.02\n")

    segment = read_spectrum(path, reader=alias, warn_unknown=False)

    assert segment.meta["instrument"] == "Gaia FGK Benchmark Stars"
    assert segment.meta["resolution_R"] == pytest.approx(42000.0)
    assert segment.wave_medium == "air"


def test_gaia_benchmark_reader_info_is_discoverable():
    info = get_reader_info("gbs")

    assert info.canonical_name == "gbs_v3_ascii"
    assert "gbs" in info.aliases
    assert info.default_observer_frame == "barycentric"
    assert info.default_stellar_rest_status == "corrected"
    assert info.default_wave_medium == "air"
    assert info.resolving_power == "common-resolution R=42000"
