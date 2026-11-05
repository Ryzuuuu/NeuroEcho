import math

import pytest

from ml.src.ensemble import MESSAGES, band_from_prob, safe_float


@pytest.mark.parametrize(
    "p, band",
    [(0.0, "GREEN"), (0.59, "GREEN"), (0.60, "YELLOW"), (0.84, "YELLOW"),
     (0.85, "RED"), (1.0, "RED")],
)
def test_band_from_prob(p, band):
    assert band_from_prob(p) == band


def test_every_band_has_a_message():
    assert set(MESSAGES) == {"GREEN", "YELLOW", "RED"}


def test_safe_float_passes_valid_numbers():
    assert safe_float("1.5", 0.0) == 1.5
    assert safe_float(2, 0.0) == 2.0


@pytest.mark.parametrize("bad", [None, float("nan"), math.inf, -math.inf, "abc", []])
def test_safe_float_falls_back_to_default(bad):
    assert safe_float(bad, 7.0) == 7.0
