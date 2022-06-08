import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.default_qutrit import DefaultQutrit
from pennylane.wires import Wires, WireError
from tests.devices.test_default_qubit import TestInverseDecomposition

OMEGA = np.exp(2 * np.pi * 1j / 3)


U_thadamard_01 = np.multiply(1 / np.sqrt(2),
                np.array(
                    [[1, 1, 0],
                     [1, -1, 0],
                     [0, 0, np.sqrt(2)]],
                    )
                )

U_tswap = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ],
    dtype=np.complex128
)


def include_inverses_with_test_data(test_data):
    return test_data + [(item[0] + ".inv", item[1], item[2]) for item in test_data]


def test_analytic_deprecation():
    """Tests if the kwarg `analytic` is used and displays error message."""
    msg = "The analytic argument has been replaced by shots=None. "
    msg += "Please use shots=None instead of analytic=True."

    with pytest.raises(
        DeviceError,
        match=msg,
    ):
        qml.device("default.qutrit", wires=1, shots=1, analytic=True)


def test_dtype_errors():
    """Test that if an incorrect dtype is provided to the device then an error is raised."""
    with pytest.raises(DeviceError, match="Real datatype must be a floating point type."):
        qml.device("default.qubit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("default.qubit", wires=1, c_dtype=np.float64)

dev = qml.device("default.qutrit", wires=1, shots=100000)



# @qml.qnode(dev)
# def test_qnode():
#     qml.TShift(wires=0)                 # |000>         -> |100>
#     qml.adjoint(qml.TShift)(wires=1)    # |100>         -> |120>
#     qml.TSWAP(wires=[0, 1])             # |120>         -> |210>
#     qml.TAdd(wires=[0, 1])              # |210>         -> |200>
#     qml.TAdd(wires=[0, 1])              # |200>         -> |220>
#     qml.TAdd(wires=[1, 0])              # |220>         -> |120>
#     qml.TAdd(wires=[0, 2])              # |120>         -> |121>
#     qml.TSWAP(wires=[2, 1])             # |121>         -> |112>
#     qml.TAdd(wires=[2, 0])              # |112>         -> |012>

#     return qml.state()


class TestApply:
    pass


class TestExpval:
    pass


class TestVar:
    pass


class TestSample:
    pass


class TestDefaultQutritIntegration:
    pass


class TestTensorExpval:
    pass


class TestTensorVar:
    pass


class TestTensorSample:
    pass


class TestDtypePreserved:
    pass


class TestProbabilityIntegration:
    pass


class TestWiresIntegration:
    pass


class TestApplyOps:
    pass


class TestInverseDecomposition:
    pass


class TestApplyOperationUnit:
    pass
