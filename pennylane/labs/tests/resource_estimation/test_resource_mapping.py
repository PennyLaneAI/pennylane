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
This module contains tests for class needed to map PennyLane operations to their ResourceOperator.
"""
import pytest

import pennylane as qml
import numpy as np
import pennylane.labs.resource_estimation.ops as re_ops
import pennylane.labs.resource_estimation.templates as re_temps
import pennylane.templates as qtemps
from pennylane.labs.resource_estimation import map_to_resource_op
from pennylane.operation import Operation

# pylint: disable= no-self-use


class Test_map_to_resource_op:
    """Test the class for mapping PennyLane operations to their ResourceOperators."""

    def test_map_to_resource_op_raises_type_error_if_not_operation(self):
        """Test that a TypeError is raised if the input is not an Operation."""
        with pytest.raises(TypeError, match="The op oper is not a valid operation"):
            map_to_resource_op("oper")

    def test_map_to_resource_op_raises_not_implemented_error(self):
        """Test that a NotImplementedError is raised for a valid Operation."""
        operation = Operation(qml.wires.Wires([4]))
        with pytest.raises(
            NotImplementedError, match="Operation doesn't have a resource equivalent"
        ):
            map_to_resource_op(operation)

    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            (qml.Identity(0), re_ops.ResourceIdentity()),
            # Single-Qubit Gates
            (qml.Hadamard(0), re_ops.ResourceHadamard()),
            (qml.S(0), re_ops.ResourceS()),
            (qml.T(0), re_ops.ResourceT()),
            (qml.PauliX(0), re_ops.ResourceX()),
            (qml.PauliY(0), re_ops.ResourceY()),
            (qml.PauliZ(0), re_ops.ResourceZ()),
            (qml.PhaseShift(0.1, wires=0), re_ops.ResourcePhaseShift()),
            (qml.Rot(0.1, 0.2, 0.3, wires=0), re_ops.ResourceRot()),
            (qml.RX(0.1, wires=0), re_ops.ResourceRX()),
            (qml.RY(0.1, wires=0), re_ops.ResourceRY()),
            (qml.RZ(0.1, wires=0), re_ops.ResourceRZ()),
            # Two-Qubit Gates
            (qml.SWAP(wires=(0, 1)), re_ops.ResourceSWAP()),
            (qml.IsingXX(0.1, wires=(0, 1)), re_ops.ResourceIsingXX()),
            (qml.IsingYY(0.1, wires=(0, 1)), re_ops.ResourceIsingYY()),
            (qml.IsingXY(0.1, wires=(0, 1)), re_ops.ResourceIsingXY()),
            (qml.IsingZZ(0.1, wires=(0, 1)), re_ops.ResourceIsingZZ()),
            (qml.PSWAP(0.1, wires=(0, 1)), re_ops.ResourcePSWAP()),
            (qml.SingleExcitation(0.1, wires=(0, 1)), re_ops.ResourceSingleExcitation()),
            (qml.CH(wires=(0, 1)), re_ops.ResourceCH()),
            (qml.CNOT(wires=(0, 1)), re_ops.ResourceCNOT()),
            (qml.ControlledPhaseShift(0.1, wires=(0, 1)), re_ops.ResourceControlledPhaseShift()),
            (qml.CRot(0.1, 0.2, 0.3, wires=(0, 1)), re_ops.ResourceCRot()),
            (qml.CRX(0.1, wires=(0, 1)), re_ops.ResourceCRX()),
            (qml.CRY(0.1, wires=(0, 1)), re_ops.ResourceCRY()),
            (qml.CRZ(0.1, wires=(0, 1)), re_ops.ResourceCRZ()),
            (qml.CY(wires=(0, 1)), re_ops.ResourceCY()),
            (qml.CZ(wires=(0, 1)), re_ops.ResourceCZ()),
            # Three-Qubit Gates
            (qml.CCZ(wires=(0, 1, 2)), re_ops.ResourceCCZ()),
            (qml.CSWAP(wires=(0, 1, 2)), re_ops.ResourceCSWAP()),
            (qml.Toffoli(wires=(0, 1, 2)), re_ops.ResourceToffoli()),
            # Multi-Qubit Gates
            (qml.MultiRZ(0.1, wires=[0, 1, 2]), re_ops.ResourceMultiRZ(num_wires=3)),
            (qml.PauliRot(0.1, "XYZ", wires=[0, 1, 2]), re_ops.ResourcePauliRot("XYZ")),
            (
                qml.MultiControlledX(wires=[0, 1, 2]),
                re_ops.ResourceMultiControlledX(num_ctrl_wires=2, num_ctrl_values=2),
            ),
            # Custom/Template Gates
            (qtemps.TemporaryAND(wires=[0, 1, 2]), re_ops.ResourceTempAND()),
        ],
    )
    def test_map_to_resource_op(self, operator, expected_res_op):
        """Test that map_to_resource_op maps to the appropriate resource operator"""

        assert map_to_resource_op(operator) == expected_res_op


    @pytest.mark.parametrize(
        "operator, expected_res_op",
        [
            (
                qtemps.OutMultiplier(x_wires=[0, 1], y_wires=[2], output_wires=[3, 4]),
                re_temps.ResourceOutMultiplier(a_num_qubits=2, b_num_qubits=1),
            ),
            (
                qml.SemiAdder(x_wires=[0, 1, 2], y_wires=[3, 4], work_wires=[5]),
                re_temps.ResourceSemiAdder(max_register_size=3),
            ),
            # (qtemps.QFT(wires=[0, 1, 2]), re_temps.ResourceQFT(num_wires=3)),
            # (
            #     qtemps.AQFT(order=3, wires=[0, 1, 2, 3]),
            #     re_temps.ResourceAQFT(order=3, num_wires=4),
            # ),
            # (
            #     qtemps.BasisRotation(np.eye(4), wires=[0, 1, 2, 3]),
            #     re_temps.ResourceBasisRotation(dim_N=4),
            # ),
            # (
            #     qtemps.Select([qml.PauliX(2), qml.PauliY(2)], control=[0, 1]),
            #     re_temps.ResourceSelect(select_ops=[re_ops.ResourceX(), re_ops.ResourceY()]),
            # ),
            # (
            #     qtemps.QROM([[0, 1], [1, 1], [1, 0]], wires=[0, 1, 2, 3]),
            #     re_temps.ResourceQROM(num_bitstrings=3, size_bitstring=2),
            # ),
            # (
            #     qtemps.SelectPauliRot(0.5, "Z", wires=[0, 1, 2]),
            #     re_temps.ResourceSelectPauliRot(rotation_axis="Z", num_ctrl_wires=2, precision=None),
            # ),
            # (
            #     qml.QubitUnitary(np.eye(2), wires=0),
            #     re_temps.ResourceQubitUnitary(num_wires=1, precision=None),
            # ),
            # (
            #     qtemps.ControlledSequence(qml.RY(0.5, wires=2), wires=[0, 1]),
            #     re_temps.ResourceControlledSequence(base=re_ops.ResourceRY(), num_control_wires=2),
            # ),
            # (
            #     qml.QuantumPhaseEstimation(qml.PauliZ(2), estimation_wires=[0, 1]),
            #     re_temps.ResourceQPE(base=re_ops.ResourceZ(), num_estimation_wires=2),
            # ),
            # (
            #     qml.TrotterProduct(
            #         qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.PauliY(1)]),
            #         time=1.0,
            #         n=10,
            #         order=2,
            #     ),
            #     re_temps.ResourceTrotterProduct(
            #         first_order_expansion=[re_ops.ResourceX(), re_ops.ResourceY()],
            #         num_steps=10,
            #         order=2,
            #     ),
            # ),
            # (
            #     qml.IntegerComparator(5, geq=False, wires=[0, 1, 2, 3]),
            #     re_temps.ResourceIntegerComparator(value=5, register_size=3, geq=False),
            # ),
        ],
    )
    def test_map_to_resource_op_templates(self, operator, expected_res_op):
        """Test that map_to_resource_op maps templates to the appropriate resource operator"""
        assert map_to_resource_op(operator) == expected_res_op
