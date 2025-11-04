# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for class needed to map PennyLane operations to their associated resource
operator.
"""
import numpy as np
import pytest

import pennylane as qml
import pennylane.estimator as re_ops
import pennylane.estimator.templates as re_temps
import pennylane.ops as qops
import pennylane.templates as qtemps
from pennylane.estimator.resource_mapping import _map_to_resource_op
from pennylane.operation import Operation

# pylint: disable= no-self-use,too-few-public-methods


class TestMapToResourceOp:
    """Test the class for mapping PennyLane operations to their resource operators."""

    def test_map_to_resource_op_raises_type_error_if_not_operation(self):
        """Test that a TypeError is raised if the input is not an Operation."""
        with pytest.raises(TypeError, match="is not a valid operation"):
            _map_to_resource_op("oper")

    def test_map_to_resource_op_raises_not_implemented_error(self):
        """Test that a NotImplementedError is raised for a valid Operation."""
        operation = Operation(qml.wires.Wires([4]))
        with pytest.raises(
            NotImplementedError, match="Operation doesn't have a resource equivalent"
        ):
            _map_to_resource_op(operation)

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            (qml.Identity(0), re_ops.Identity()),
            (qml.GlobalPhase(0), re_ops.GlobalPhase()),
            # Single-Qubit Gates
            (qml.Hadamard(0), re_ops.Hadamard()),
            (qml.S(0), re_ops.S()),
            (qml.T(0), re_ops.T()),
            (qml.PauliX(0), re_ops.X()),
            (qml.PauliY(0), re_ops.Y()),
            (qml.PauliZ(0), re_ops.Z()),
            (qml.PhaseShift(0.1, wires=0), re_ops.PhaseShift()),
            (qml.Rot(0.1, 0.2, 0.3, wires=0), re_ops.Rot()),
            (qml.RX(0.1, wires=0), re_ops.RX()),
            (qml.RY(0.1, wires=0), re_ops.RY()),
            (qml.RZ(0.1, wires=0), re_ops.RZ()),
            # Two-Qubit Gates
            (qml.SWAP(wires=(0, 1)), re_ops.SWAP()),
            (qml.SingleExcitation(0.1, wires=(0, 1)), re_ops.SingleExcitation()),
            (qml.CH(wires=(0, 1)), re_ops.CH()),
            (qml.CNOT(wires=(0, 1)), re_ops.CNOT()),
            (qml.ControlledPhaseShift(0.1, wires=(0, 1)), re_ops.ControlledPhaseShift()),
            (qml.CRot(0.1, 0.2, 0.3, wires=(0, 1)), re_ops.CRot()),
            (qml.CRX(0.1, wires=(0, 1)), re_ops.CRX()),
            (qml.CRY(0.1, wires=(0, 1)), re_ops.CRY()),
            (qml.CRZ(0.1, wires=(0, 1)), re_ops.CRZ()),
            (qml.CY(wires=(0, 1)), re_ops.CY()),
            (qml.CZ(wires=(0, 1)), re_ops.CZ()),
            # Three-Qubit Gates
            (qml.CCZ(wires=(0, 1, 2)), re_ops.CCZ()),
            (qml.CSWAP(wires=(0, 1, 2)), re_ops.CSWAP()),
            (qml.Toffoli(wires=(0, 1, 2)), re_ops.Toffoli()),
            # Multi-Qubit Gates
            (qml.MultiRZ(0.1, wires=[0, 1, 2]), re_ops.MultiRZ(num_wires=3)),
            (qml.PauliRot(0.1, "XYZ", wires=[0, 1, 2]), re_ops.PauliRot("XYZ")),
            (
                qml.MultiControlledX(wires=[0, 1, 2]),
                re_ops.MultiControlledX(num_ctrl_wires=2, num_zero_ctrl=0),
            ),
            # Custom/Template Gates
            (qtemps.TemporaryAND(wires=[0, 1, 2]), re_ops.TemporaryAND()),
        ],
    )
    def test_map_to_resource_op(self, operator, expected_res_op):
        """Test that _map_to_resource_op maps to the appropriate resource operator"""
        assert _map_to_resource_op(operator) == expected_res_op

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            (
                qtemps.OutMultiplier(x_wires=[0, 1], y_wires=[2], output_wires=[3, 4]),
                re_temps.OutMultiplier(a_num_wires=2, b_num_wires=1, wires=[0, 1, 2, 3, 4]),
            ),
            (
                qml.SemiAdder(x_wires=[0, 1, 2], y_wires=[3, 4], work_wires=[5]),
                re_temps.SemiAdder(max_register_size=3, wires=[0, 1, 2, 3, 4]),
            ),
            (qtemps.QFT(wires=[0, 1, 2]), re_temps.QFT(num_wires=3, wires=[0, 1, 2])),
            (
                qtemps.AQFT(order=3, wires=[0, 1, 2, 3, 4]),
                re_temps.AQFT(order=3, num_wires=5, wires=[0, 1, 2, 3, 4]),
            ),
            (
                qtemps.BasisRotation(wires=[0, 1, 2, 3], unitary_matrix=np.eye(4)),
                re_temps.BasisRotation(dim=4, wires=[0, 1, 2, 3]),
            ),
            (
                qtemps.Select([qml.PauliX(2), qml.PauliY(2)], control=[0, 1]),
                re_temps.Select(ops=[re_ops.X(), re_ops.Y()], wires=[0, 1, 2]),
            ),
            (
                qtemps.QROM(
                    bitstrings=["01", "11", "10"],
                    control_wires=[0, 1],
                    target_wires=[2, 3],
                    work_wires=[4],
                    clean=False,
                ),
                re_temps.QROM(
                    num_bitstrings=3, size_bitstring=2, restored=False, wires=[0, 1, 2, 3, 4]
                ),
            ),
            (
                qtemps.SelectPauliRot(
                    angles=np.array([0.1, 0.2, 0.3, 0.4]),
                    control_wires=[0, 1],
                    target_wire=[2],
                    rot_axis="Y",
                ),
                re_temps.SelectPauliRot(
                    rot_axis="Y", num_ctrl_wires=2, precision=None, wires=[0, 1, 2]
                ),
            ),
            (
                qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2]),
                re_temps.ControlledSequence(
                    base=re_ops.RX(wires=3), num_control_wires=3, wires=[0, 1, 2]
                ),
            ),
            (
                qml.QubitUnitary(np.eye(2), wires=0),
                re_ops.QubitUnitary(num_wires=1, precision=None, wires=[0]),
            ),
            (
                qtemps.SelectPauliRot(
                    angles=np.array([0.1, 0.2, 0.3, 0.4]),
                    control_wires=[0, 1],
                    target_wire=[2],
                    rot_axis="Y",
                ),
                re_temps.SelectPauliRot(
                    rot_axis="Y", num_ctrl_wires=2, precision=None, wires=[0, 1, 2]
                ),
            ),
            (
                qml.QuantumPhaseEstimation(qml.PauliZ(2), estimation_wires=[0, 1]),
                re_temps.QPE(base=re_ops.Z(), num_estimation_wires=2, wires=[2, 0, 1]),
            ),
            (
                qml.TrotterProduct(
                    qml.dot(
                        [0.25, 0.75, -1],
                        [qml.X(0), qml.Z(1), qml.prod(qml.Y(0), qml.Y(2), qml.Y(1))],
                    ),
                    time=1.0,
                    n=10,
                    order=2,
                ),
                re_temps.TrotterProduct(
                    first_order_expansion=[
                        re_ops.RX(wires=[0]),
                        re_ops.RZ(wires=[1]),
                        re_ops.PauliRot("YYY", wires=[0, 2, 1]),
                    ],
                    num_steps=10,
                    order=2,
                ),
            ),
            (
                qml.IntegerComparator(5, geq=False, wires=[0, 1, 2, 3]),
                re_temps.IntegerComparator(value=5, register_size=3, geq=False, wires=[0, 1, 2, 3]),
            ),
            (
                qtemps.MPSPrep(
                    [np.ones((2, 4)), np.ones((4, 2, 2)), np.ones((2, 2))], wires=[0, 1, 2]
                ),
                re_temps.MPSPrep(
                    num_mps_matrices=3, max_bond_dim=4, precision=None, wires=[0, 1, 2]
                ),
            ),
            (
                qtemps.QROMStatePreparation(
                    np.array([0.25] * 16), wires=[0, 1, 2, 3], precision_wires=[4, 5]
                ),
                re_temps.QROMStatePreparation(
                    num_state_qubits=4,
                    precision=np.pi / 4,
                    positive_and_real=True,
                    select_swap_depths=1,
                    wires=[0, 1, 2, 3],
                ),
            ),
            (
                qtemps.QROMStatePreparation(
                    1j * np.array([0.25] * 16), wires=[0, 1, 2, 3], precision_wires=[4, 5]
                ),
                re_temps.QROMStatePreparation(
                    num_state_qubits=4,
                    precision=np.pi / 4,
                    positive_and_real=False,
                    select_swap_depths=1,
                    wires=[0, 1, 2, 3],
                ),
            ),
        ],
    )
    def test_map_to_resource_op_templates(self, operator, expected_res_op):
        """Test that _map_to_resource_op maps templates to the appropriate resource operator"""
        mapped_op = _map_to_resource_op(operator)
        assert mapped_op == expected_res_op
        assert mapped_op.wires == expected_res_op.wires

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        (
            (
                qops.ChangeOpBasis(
                    compute_op=qops.T(wires=0),
                    target_op=qops.Z(wires=0),
                ),
                re_ops.ChangeOpBasis(
                    compute_op=re_ops.T(wires=0),
                    target_op=re_ops.Z(wires=0),
                    uncompute_op=re_ops.Adjoint(re_ops.T(wires=0)),
                ),
            ),
            (
                qops.ChangeOpBasis(
                    compute_op=qtemps.QFT(wires=[0, 1, 2]),
                    target_op=qtemps.ControlledSequence(qops.S(wires=2), control=[0, 1]),
                    uncompute_op=qops.adjoint(qtemps.AQFT(order=3, wires=[0, 1, 2, 3, 4])),
                ),
                re_ops.ChangeOpBasis(
                    compute_op=re_temps.QFT(num_wires=3, wires=[0, 1, 2]),
                    target_op=re_temps.ControlledSequence(
                        re_ops.S(wires=2), num_control_wires=2, wires=[0, 1]
                    ),
                    uncompute_op=re_ops.Adjoint(
                        re_temps.AQFT(order=3, num_wires=5, wires=[0, 1, 2, 3, 4])
                    ),
                ),
            ),
            (
                qops.prod(
                    qops.X(0),
                    qops.H(0),
                    qops.Z(1),
                    qops.CNOT([0, 1]),
                    qops.PhaseShift(1.23, wires=2),
                    qops.CRY(1.23, wires=[0, 2]),
                ),
                re_ops.Prod(
                    (
                        re_ops.X(wires=0),
                        re_ops.Hadamard(wires=0),
                        re_ops.Z(wires=1),
                        re_ops.CNOT(wires=[0, 1]),
                        re_ops.PhaseShift(wires=2),
                        re_ops.CRY(wires=[0, 2]),
                    ),
                    wires=[0, 1, 2],
                ),
            ),
            (
                qops.adjoint(
                    qtemps.QROMStatePreparation(
                        state_vector=np.array([1] * (2**3)) / (2 ** (3 / 2)),
                        wires=[0, 1, 2],
                        precision_wires=[3, 4, 5],
                    )
                ),
                re_ops.Adjoint(
                    re_temps.QROMStatePreparation(
                        num_state_qubits=3,
                        precision=np.pi / (2**3),
                        positive_and_real=True,
                        wires=[0, 1, 2],
                    ),
                ),
            ),
            (
                qops.adjoint(qops.T(wires=0)),
                re_ops.Adjoint(re_ops.T(wires=0)),
            ),
            (
                qops.pow(qops.Hadamard(wires=0), z=7),
                re_ops.Pow(re_ops.Hadamard(wires=0), pow_z=7),
            ),
            (
                qops.ctrl(
                    qtemps.SemiAdder(x_wires=[0, 1, 2], y_wires=[3, 4, 5], work_wires=[6, 7, 8]),
                    control=["c1", "c2"],
                    work_wires=["w1", "w2"],
                    work_wire_type="zeroed",
                ),
                re_ops.Controlled(
                    re_temps.SemiAdder(3, wires=[0, 1, 2, 3, 4, 5]),
                    num_ctrl_wires=2,
                    num_zero_ctrl=0,
                    wires=["c1", "c2"],
                ),
            ),
            (
                qops.ctrl(
                    qtemps.AQFT(order=4, wires=[0, 1, 2, 3, 4, 5, 6]),
                    control=["c1", "c2", "c3"],
                    control_values=[0, 1, 0],
                    work_wires=["w1", "w2"],
                    work_wire_type="borrowed",
                ),
                re_ops.Controlled(
                    re_temps.AQFT(order=4, num_wires=7, wires=[0, 1, 2, 3, 4, 5, 6]),
                    num_ctrl_wires=3,
                    num_zero_ctrl=2,
                    wires=["c1", "c2", "c3"],
                ),
            ),
        ),
    )
    def test_map_to_resource_op_symbolic(self, operator, expected_res_op):
        """Test that _map_to_resource_op maps templates to the appropriate resource operator"""
        mapped_op = _map_to_resource_op(operator)
        assert mapped_op == expected_res_op
        assert mapped_op.wires == expected_res_op.wires

    def test_map_from_decomposition(self):
        """Test that the decomposition is used when a resource equivalent is not defined"""

        class DummyOp(Operation):
            num_wires = 2

            def __init__(self, theta, wires=None, id=None):
                super().__init__(theta, wires=wires, id=id)

            @staticmethod
            def compute_decomposition(theta, wires):
                return [qml.RX(theta, wires[0]), qml.CNOT(wires)]

        def actual_circ():
            DummyOp(theta=1.23, wires=[0, 1])

        def expected_circ():
            re_ops.RX(wires=0)
            re_ops.CNOT(wires=[0, 1])

        assert re_ops.estimate(actual_circ)() == re_ops.estimate(expected_circ)()
