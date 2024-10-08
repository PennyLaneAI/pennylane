# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the ``decompose`` transform"""

import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane.transforms.decompose import decompose

# pylint: disable=too-few-public-methods


class InfiniteOp(qml.operation.Operation):
    """An op with an infinite decomposition."""

    num_wires = 1

    def decomposition(self):
        return [InfiniteOp(*self.parameters, self.wires)]


class TestDecompose:
    """Unit tests for decompose function"""

    gate_set_inputs = [None, "RX", ["RX"], ("RX",), {"RX"}, qml.RX, [qml.RX], (qml.RX,), {qml.RX}]

    iterables_test = [
        (
            [qml.Hadamard(0)],
            {qml.RX, qml.RZ},
            [qml.RZ(qnp.pi / 2, 0), qml.RX(qnp.pi / 2, 0), qml.RZ(qnp.pi / 2, 0)],
            None,
        ),
        (
            [qml.SWAP(wires=[0, 1])],
            {qml.CNOT},
            [qml.CNOT([0, 1]), qml.CNOT([1, 0]), qml.CNOT([0, 1])],
            None,
        ),
        (
            [qml.Toffoli([0, 1, 2])],
            {qml.Toffoli},
            [qml.Toffoli([0, 1, 2])],
            None,
        ),
        (
            [qml.measurements.MidMeasureMP(0)],
            {},
            [qml.measurements.MidMeasureMP(0)],
            "MidMeasureMP",
        ),
    ]

    callables_test = [
        (
            [qml.Hadamard(0)],
            lambda op: "RX" in op.name,
            [qml.RZ(qnp.pi / 2, 0), qml.RX(qnp.pi / 2, 0), qml.RZ(qnp.pi / 2, 0)],
            "RZ",
        ),
        (
            [qml.Toffoli([0, 1, 2])],
            lambda op: len(op.wires) <= 2,
            [
                qml.Hadamard(wires=[2]),
                qml.CNOT(wires=[1, 2]),
                qml.ops.op_math.Adjoint(qml.T(wires=[2])),
                qml.CNOT(wires=[0, 2]),
                qml.T(wires=[2]),
                qml.CNOT(wires=[1, 2]),
                qml.ops.op_math.Adjoint(qml.T(wires=[2])),
                qml.CNOT(wires=[0, 2]),
                qml.T(wires=[1]),
                qml.T(wires=[2]),
                qml.CNOT(wires=[0, 1]),
                qml.Hadamard(wires=[2]),
                qml.T(wires=[0]),
                qml.ops.op_math.Adjoint(qml.T(wires=[1])),
                qml.CNOT(wires=[0, 1]),
            ],
            None,
        ),
    ]

    @pytest.mark.parametrize("gate_set", gate_set_inputs)
    def test_different_input_formats(self, gate_set):
        """Tests that gate sets of different types are handled correctly"""
        tape = qml.tape.QuantumScript([qml.RX(0, wires=[0])])
        decompose(tape, gate_set=gate_set)

    def test_user_warning(self):
        """Tests that user warning is raised if operator does not have a valid decomposition"""
        tape = qml.tape.QuantumScript([qml.RX(0, wires=[0])])
        with pytest.warns(UserWarning, match="has no supported decomposition"):
            decompose(tape, gate_set=lambda op: op.name not in {"RX"})

    def test_infinite_decomposition_loop(self):
        """Test that a recursion error is raised if decomposition enters an infinite loop."""
        tape = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(RecursionError, match=r"Reached recursion limit trying to decompose"):
            decompose(tape, lambda obj: obj.has_matrix)

    @pytest.mark.parametrize("initial_ops, gate_set, expected_ops, warning_pattern", iterables_test)
    def test_iterable_gate_set(self, initial_ops, gate_set, expected_ops, warning_pattern):
        """Tests that gate sets defined with iterables decompose correctly"""
        tape = qml.tape.QuantumScript(initial_ops)

        if warning_pattern is not None:
            with pytest.warns(UserWarning, match=warning_pattern):
                (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
        else:
            (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)

        expected_tape = qml.tape.QuantumScript(expected_ops)

        qml.assert_equal(decomposed_tape, expected_tape)

    @pytest.mark.parametrize("initial_ops, gate_set, expected_ops, warning_pattern", iterables_test)
    def test_callable_gate_set(self, initial_ops, gate_set, expected_ops, warning_pattern):
        """Tests that gate sets defined by callables decompose correctly"""
        tape = qml.tape.QuantumScript(initial_ops)

        if warning_pattern is not None:
            with pytest.warns(UserWarning, match=warning_pattern):
                (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
        else:
            (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)

        expected_tape = qml.tape.QuantumScript(expected_ops)

        qml.assert_equal(decomposed_tape, expected_tape)
