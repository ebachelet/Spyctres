"""Small internal helpers for working with Spyctres spectrum containers."""

from .io import SpectrumCollection, SpectrumSegment, coerce_spectrum


def spectrum_segments(
    spectrum,
    *,
    tuple_is_collection=False,
    coerce=True,
    warn_unknown=False,
):
    """Return a list of ``SpectrumSegment`` objects from supported inputs.

    ``tuple_is_collection`` preserves older helper semantics used by workflow
    and diagnostic code where a Python list/tuple means "already a sequence of
    segments".  Leave it false when accepting public tuple-like array inputs
    such as ``(wave, flux[, err[, mask]])``.
    """
    if isinstance(spectrum, SpectrumSegment):
        return [spectrum]
    if isinstance(spectrum, SpectrumCollection):
        return list(spectrum.segments)
    if tuple_is_collection and isinstance(spectrum, (list, tuple)):
        return list(spectrum)
    if not coerce:
        return [spectrum]

    coerced = coerce_spectrum(spectrum, warn_unknown=warn_unknown)
    if isinstance(coerced, SpectrumCollection):
        return list(coerced.segments)
    return [coerced]
