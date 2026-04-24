import pytest

from pennylane import list_decomps
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.ffft import FFFT


@pytest.mark.parametrize("wires", [(0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7)])
def test_ffft_decomposition_new(wires):
    op = FFFT(wires)

    for rule in list_decomps(FFFT):
        _test_decomposition_rule(op, rule)
