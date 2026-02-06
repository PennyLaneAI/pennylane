"""Tests for ``device_constructor``"""

import numpy as np
import pytest

import pennylane as qp


def test_my_feature_is_deprecated():
    """Test that custom_decomps is deprecated."""

    def ion_trap_cnot(wires, **_):
        return [
            qp.RY(np.pi / 2, wires=wires[0]),
            qp.IsingXX(np.pi / 2, wires=wires),
            qp.RX(-np.pi / 2, wires=wires[0]),
            qp.RY(-np.pi / 2, wires=wires[0]),
            qp.RY(-np.pi / 2, wires=wires[1]),
        ]

    with pytest.warns(
        qp.exceptions.PennyLaneDeprecationWarning,
        match="The ``custom_decomps`` keyword argument",
    ):
        _ = qp.device("default.qubit", wires=2, custom_decomps={"CNOT": ion_trap_cnot})
