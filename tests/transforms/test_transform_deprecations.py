"""Unit tests for deprecated transform locations in PennyLane."""

import pytest

from pennylane import noise, transforms
from pennylane.exceptions import PennyLaneDeprecationWarning

DEPRECATED_TRANSFORMS = [
    "add_noise",
    "insert",
    "mitigate_with_zne",
    "fold_global",
    "poly_extrapolate",
    "richardson_extrapolate",
    "exponential_extrapolate",
]


@pytest.mark.parametrize("name", DEPRECATED_TRANSFORMS)
def test_deprecation(name):
    """Test that deprecated transforms raise a warning and redirect to the new location."""

    with pytest.warns(
        PennyLaneDeprecationWarning,
        match=f"pennylane.{name} is no longer accessible from the transforms module",
    ):
        fn = getattr(transforms, name)
    assert fn is getattr(noise, name)
