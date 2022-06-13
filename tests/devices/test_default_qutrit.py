"""
Unit tests for the :mod:`pennylane.plugin.DefaultQutrit` device.
"""
import math

import pytest
import pennylane as qml
from pennylane import numpy as np, DeviceError
from pennylane.devices.default_qutrit import DefaultQutrit
from pennylane.wires import Wires, WireError

OMEGA = np.exp(2 * np.pi * 1j / 3)


U_thadamard_01 = np.multiply(1 / np.sqrt(2),
                np.array(
                    [[1, 1, 0],
                     [1, -1, 0],
                     [0, 0, np.sqrt(2)]],
                    )
                )

U_x_02 = np.array(
    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],
    dtype=np.complex128
)

U_z_12 = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, -1]],
    dtype=np.complex128
)

U_shift = np.array(
    [[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]],
    dtype=np.complex128
)

U_clock = np.array(
    [[1, 0, 0],
     [0, OMEGA, 0],
     [0, 0, OMEGA**2]]
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

U_tadd = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0]
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
        qml.device("default.qutrit", wires=1, r_dtype=np.complex128)
    with pytest.raises(
        DeviceError, match="Complex datatype must be a complex floating point type."
    ):
        qml.device("default.qutrit", wires=1, c_dtype=np.float64)

dev = qml.device("default.qutrit", wires=1, shots=100000)


class TestApply:
    """Tests that operations and inverses of certain operations are applied correctly or that the proper
    errors are raised.
    """

    # TODO: Add tests for non-parametric ops after they're implemented

    # TODO: Add more data as parametric ops get added
    test_data_single_wire_with_parameters = [
        (qml.QutritUnitary, [1, 0, 0], [1, 1, 0] / np.sqrt(2), U_thadamard_01),
        (qml.QutritUnitary, [1, 0, 0], [0, 0, 1], U_x_02),
        (qml.QutritUnitary, [1, 0, 0], [1, 0, 0], U_z_12),
        (qml.QutritUnitary, [0, 1, 0], [0, 1, 0], U_x_02),
        (qml.QutritUnitary, [0, 0, 1], [0, 0, -1], U_z_12),
        (qml.QutritUnitary, [0, 1, 0], [0, 0, 1], U_shift),
        (qml.QutritUnitary, [0, 1, 0], [0, OMEGA, 0], U_clock),
    ]

    # TODO: Add more data as parametric ops get added
    test_data_single_wire_with_parameters_inverse = [
        (qml.QutritUnitary, [1, 0, 0], [0, 0, 1], U_shift),
        (qml.QutritUnitary, [0, 0, 1], [0, 1, 0], U_shift),
        (qml.QutritUnitary, [0, OMEGA, 0], [0, 1, 0], U_clock),
        (qml.QutritUnitary, [0, 0, OMEGA**2], [0, 0, 1], U_clock),
    ]

    @pytest.mark.parametrize(
        "operation, input, expected_output, par", test_data_single_wire_with_parameters
    )
    def test_apply_operation_single_wire_with_parameters(
        self, qutrit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qutrit_device_1_wire._state = np.array(input, dtype=qutrit_device_1_wire.C_DTYPE)

        qutrit_device_1_wire.apply([operation(par, wires=[0])])

        assert np.allclose(qutrit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qutrit_device_1_wire._state.dtype == qutrit_device_1_wire.C_DTYPE

    @pytest.mark.parametrize(
        "operation, input, expected_output, par", test_data_single_wire_with_parameters_inverse
    )
    def test_apply_operation_single_wire_with_parameters_inverse(
        self, qutrit_device_1_wire, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for single wire
        operations that have parameters."""

        qutrit_device_1_wire._state = np.array(input, dtype=qutrit_device_1_wire.C_DTYPE)

        qutrit_device_1_wire.apply([operation(par, wires=[0]).inv()])

        assert np.allclose(qutrit_device_1_wire._state, np.array(expected_output), atol=tol, rtol=0)
        assert qutrit_device_1_wire._state.dtype == qutrit_device_1_wire.C_DTYPE

    # TODO: Add more ops as parametric operations get added
    test_data_two_wires_with_parameters = [
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], U_tswap),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], U_tswap),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            U_tswap
        ),
        (
            qml.QutritUnitary,
            np.multiply(0.5, [0, 1, 1, 0, 0, 0, 0, 1, 1]),
            np.multiply(0.5, [0, 0, 0, 1, 0, 1, 1, 0, 1]),
            U_tswap
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], U_tadd),
    ]

    # TODO: Add more ops as parametric operations get added
    test_data_two_wires_with_parameters_inverse = [
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], U_tswap),
        (qml.QutritUnitary, [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], U_tswap),
        (
            qml.QutritUnitary,
            [0, 0, 1, 0, 0, 0, 0, 1, 0] / np.sqrt(2),
            [0, 0, 0, 0, 0, 1, 1, 0, 0] / np.sqrt(2),
            U_tswap
        ),
        (
            qml.QutritUnitary,
            np.multiply([0, 1, 1, 0, 0, 0, 0, 1, 1], 0.5),
            np.multiply([0, 0, 0, 1, 0, 1, 1, 0, 1], 0.5),
            U_tswap
        ),
        (qml.QutritUnitary, [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 0, 1, 0, 0], [0 ,0 ,0, 0, 0, 0, 0, 1, 0], U_tadd),
        (qml.QutritUnitary, [0, 0, 0, 0, 0, 1, 0, 0, 0], [0 ,0, 0, 0, 1, 0, 0, 0, 0], U_tadd),
    ]

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters
    )
    def test_apply_operation_two_wires_with_parameters(
        self, qutrit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qutrit_device_2_wires._state = np.array(input, dtype=qutrit_device_2_wires.C_DTYPE).reshape(
            (3, 3)
        )
        qutrit_device_2_wires.apply([operation(par, wires=[0, 1])])

        assert np.allclose(
            qutrit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qutrit_device_2_wires._state.dtype == qutrit_device_2_wires.C_DTYPE

    @pytest.mark.parametrize(
        "operation,input,expected_output,par", test_data_two_wires_with_parameters_inverse
    )
    def test_apply_operation_two_wires_with_parameters_inverse(
        self, qutrit_device_2_wires, tol, operation, input, expected_output, par
    ):
        """Tests that applying an operation yields the expected output state for two wire
        operations that have parameters."""

        qutrit_device_2_wires._state = np.array(input, dtype=qutrit_device_2_wires.C_DTYPE).reshape(
            (3, 3)
        )
        qutrit_device_2_wires.apply([operation(par, wires=[0, 1]).inv()])

        assert np.allclose(
            qutrit_device_2_wires._state.flatten(), np.array(expected_output), atol=tol, rtol=0
        )
        assert qutrit_device_2_wires._state.dtype == qutrit_device_2_wires.C_DTYPE

    # TODO: Add tests for state preperation ops


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
