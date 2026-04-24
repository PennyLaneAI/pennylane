import pytest

from pennylane import list_decomps
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.ffft import FFFT


@pytest.mark.parametrize("wires", [(0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7)])
def test_ffft_decomposition_new(wires):
    op = FFFT(wires)

    for rule in list_decomps(FFFT):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize(
    "wires, error_type, error_msg",
    [
        (
            tuple(),
            ValueError,
            "The number of wires must be at least 2"
        ),
        (
            (0, 1, 2),
            NotImplementedError,
            "odd numbers of wires"
        ),
    ]
)
def test_raises(wires, error_type, error_msg):
    with pytest.raises(error_type, match=error_msg):
        FFFT(wires)
