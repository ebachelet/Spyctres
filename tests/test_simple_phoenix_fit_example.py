import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "simple_phoenix_fit.py"
    spec = importlib.util.spec_from_file_location("simple_phoenix_fit_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Segment:
    def __init__(self, wave):
        self.wave = np.asarray(wave, dtype=float)


def test_line_windows_are_selected_from_fitted_pixels_only():
    module = _load_example_module()
    segment = _Segment(np.linspace(3000.0, 5200.0, 2000))
    used = (segment.wave >= 3900.0) & (segment.wave <= 4890.0)

    windows = module._line_windows_for_used_pixels(
        segment,
        used,
        groups=["balmer", "caii", "mgii"],
        half_width_A=20.0,
    )

    labels = [window[0] for window in windows]
    assert labels == [
        "Ca II K 3933.7 Å",
        "Ca II H + Hε 3968.5 Å",
        "Hδ 4101.7 Å",
        "Hγ 4340.5 Å",
        "Mg II 4481.1 Å",
        "Hβ 4861.3 Å",
    ]


def test_auto_line_groups_add_hot_lines_only_for_hot_fits():
    module = _load_example_module()

    assert module._parse_line_groups("auto", result={"teff": 9800.0}) == [
        "balmer",
        "caii",
        "mgii",
    ]
    assert module._parse_line_groups("auto", result={"teff": 11000.0}) == [
        "balmer",
        "caii",
        "mgii",
        "hei",
    ]


def test_overlapping_hot_line_windows_are_merged():
    module = _load_example_module()
    segment = _Segment(np.linspace(4400.0, 4520.0, 500))
    used = np.ones(segment.wave.size, dtype=bool)

    windows = module._line_windows_for_used_pixels(
        segment,
        used,
        groups=["mgii", "hei"],
        half_width_A=20.0,
    )

    assert len(windows) == 1
    assert windows[0][0] == "He I 4471.5 Å + Mg II 4481.1 Å"


def test_line_panel_columns_follow_identified_line_count():
    module = _load_example_module()

    assert module._line_panel_columns(1) == 1
    assert module._line_panel_columns(4) == 2
    assert module._line_panel_columns(5) == 3


def test_line_plot_path_defaults_to_companion_file():
    module = _load_example_module()

    assert (
        module._derive_line_plot_path(None, "/tmp/fit.png")
        == "/tmp/fit_lines.png"
    )
    assert (
        module._derive_line_plot_path("/tmp/custom.png", "/tmp/fit.png")
        == "/tmp/custom.png"
    )
    assert (
        module._derive_line_plot_path(None, "/tmp/fit.png", segment_index=1)
        == "/tmp/fit_lines_segment2.png"
    )


def test_append_exclusion_mask_preserves_existing_masks():
    module = _load_example_module()
    fit_kwargs = {"exclude_masks": [("existing", lambda wave: wave == wave)]}
    mask = module.nonstellar_feature_mask("dib_4428")

    module._append_exclusion_mask(fit_kwargs, mask)

    names = [
        item[0] if isinstance(item, tuple) else item.name
        for item in fit_kwargs["exclude_masks"]
    ]
    assert names == ["existing", "nonstellar:dib_4428"]
