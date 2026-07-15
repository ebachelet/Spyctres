import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "batch_quickscan_then_refine.py"
    )
    spec = importlib.util.spec_from_file_location("batch_quickscan_then_refine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_input_is_bundled_xshooter_uvb_file():
    module = _load_example_module()

    paths = module._input_paths([])

    assert len(paths) == 1
    assert paths[0].endswith("TOO_Gaia21ccu_SCI_SLIT_FLUX_MERGE1D_UVB.fits")
    assert Path(paths[0]).is_file()


def test_focused_bounds_from_quick_result_clip_to_base_bounds():
    module = _load_example_module()
    quick = {
        "teff": 9800.0,
        "feh": -0.2,
        "logg": 2.7,
        "rv_kms": -5.0,
    }
    base_bounds = (
        (4500.0, -1.5, 2.5, -300.0),
        (10000.0, 0.5, 5.5, 300.0),
    )

    p0, bounds = module.focused_bounds_from_quick_result(
        quick,
        base_bounds,
        margins=(1000.0, 0.5, 0.8, 50.0),
    )

    assert p0 == pytest.approx((9800.0, -0.2, 2.7, -5.0))
    assert bounds[0] == pytest.approx((8800.0, -0.7, 2.5, -55.0))
    assert bounds[1] == pytest.approx((10000.0, 0.3, 3.5, 45.0))


def test_focused_bounds_reject_nonpositive_margins():
    module = _load_example_module()

    with pytest.raises(ValueError, match="positive"):
        module.focused_bounds_from_quick_result(
            {"teff": 6000.0, "feh": 0.0, "logg": 4.0, "rv_kms": 0.0},
            ((4500.0, -1.5, 2.5, -300.0), (10000.0, 0.5, 5.5, 300.0)),
            margins=(1000.0, 0.0, 0.8, 50.0),
        )


def test_make_refine_fit_kwargs_can_keep_quick_window(monkeypatch):
    module = _load_example_module()

    def fake_prepare(_spectrum, **_kwargs):
        return {
            "p0": (6000.0, 0.0, 4.0, 0.0),
            "bounds": (
                (4500.0, -1.5, 2.5, -300.0),
                (10000.0, 0.5, 5.5, 300.0),
            ),
            "regions": [(3800.0, 7000.0)],
        }, SimpleNamespace(to_dict=lambda: {"fake": True})

    monkeypatch.setattr(module, "prepare_phoenix_fit_kwargs", fake_prepare)
    args = SimpleNamespace(
        refine_defaults_mode="standard",
        refine_window="quick",
        teff_margin=500.0,
        feh_margin=0.25,
        logg_margin=0.5,
        rv_margin=20.0,
        refine_rv_grid_n=21,
        refine_multistart=2,
        refine_max_nfev=99,
    )
    quick = {
        "teff": 8000.0,
        "feh": 0.0,
        "logg": 4.0,
        "rv_kms": 10.0,
    }

    fit_kwargs, suggestion, focus = module.make_refine_fit_kwargs(
        spectrum=object(),
        quick_result=quick,
        quick_fit_kwargs={"regions": [(3800.0, 5200.0)]},
        args=args,
    )

    assert suggestion.to_dict() == {"fake": True}
    assert fit_kwargs["regions"] == [(3800.0, 5200.0)]
    assert fit_kwargs["p0"] == pytest.approx((8000.0, 0.0, 4.0, 10.0))
    assert fit_kwargs["bounds"][0] == pytest.approx((7500.0, -0.25, 3.5, -10.0))
    assert fit_kwargs["bounds"][1] == pytest.approx((8500.0, 0.25, 4.5, 30.0))
    assert fit_kwargs["rv_grid_n"] == 21
    assert fit_kwargs["multistart"] == 2
    assert fit_kwargs["max_nfev"] == 99
    assert focus["refine_window_policy"] == "quick"
