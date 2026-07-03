def test_legacy_module_import_and_core_entry_points():
    from Spyctres import Spyctres

    expected = (
        "velocity_correction",
        "Barycentric_velocity",
        "get_element_lines",
        "star_spectrum_new",
        "load_telluric_lines",
    )

    for name in expected:
        assert callable(getattr(Spyctres, name))
