"""Tests for helpers inside :mod:`material_response`."""

import pytest

from material_response import _clamp


def test_clamp_raises_when_minimum_exceeds_maximum() -> None:
    """Ensure invalid clamp bounds raise a :class:`ValueError`."""

    with pytest.raises(ValueError):
        _clamp(0.5, minimum=0.8, maximum=0.2)
