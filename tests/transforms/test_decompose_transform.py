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

import warnings

import pytest

import pennylane as qml
import pennylane.numpy as qnp
from pennylane.operation import Operation
from pennylane.transforms.decompose import _operator_decomposition_gen, decompose

# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=too-few-public-methods


@pytest.fixture(autouse=True)
def warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class NoMatOp(Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


class InfiniteOp(Operation):
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
                qml.T(wires=[2]),
                qml.T(wires=[1]),
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
        (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
        qml.assert_equal(tape, decomposed_tape)

    def test_user_warning(self):
        """Tests that user warning is raised if operator does not have a valid decomposition"""
        tape = qml.tape.QuantumScript([qml.RX(0, wires=[0])])
        with pytest.warns(UserWarning, match="does not define a decomposition"):
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

    @pytest.mark.parametrize("initial_ops, gate_set, expected_ops, warning_pattern", callables_test)
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


def test_null_postprocessing():
    """Tests the null postprocessing function in the decompose transform"""
    tape = qml.tape.QuantumScript([qml.Hadamard(0), qml.RX(0, 0)])
    (_,), fn = qml.transforms.decompose(tape, gate_set={qml.RX, qml.RZ})
    assert fn((1,)) == 1


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, wires=0), qml.QFT(wires=range(3))))
    def test_operator_decomposition_gen_accepted_operator(self, op):
        """Test the _operator_decomposition_gen function on an operator that is accepted."""

        def stopping_condition(op):
            return op.has_matrix

        casted_to_list = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(casted_to_list) == 1
        assert casted_to_list[0] is op

    def test_operator_decomposition_gen_decomposed_operators_single_nesting(self):
        """Assert _operator_decomposition_gen turns into a list with the operators decomposition
        when only a single layer of expansion is necessary."""

        def stopping_condition(op):
            return op.has_matrix

        op = NoMatOp("a")
        casted_to_list = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(casted_to_list) == 2
        qml.assert_equal(casted_to_list[0], qml.PauliX("a"))
        qml.assert_equal(casted_to_list[1], qml.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        def stopping_condition(op):
            return op.has_matrix

        class RaggedDecompositionOp(Operation):
            """class with a ragged decomposition."""

            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qml.S(self.wires), qml.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(final_decomp) == 5
        qml.assert_equal(final_decomp[0], qml.PauliX("a"))
        qml.assert_equal(final_decomp[1], qml.PauliY("a"))
        qml.assert_equal(final_decomp[2], qml.S("a"))
        qml.assert_equal(final_decomp[3], qml.adjoint(qml.PauliY("a")))
        qml.assert_equal(final_decomp[4], qml.adjoint(qml.PauliX("a")))

    def test_operator_decomposition_gen_max_depth_reached(self):
        """Tests whether max depth reached flag gets activated"""

        stopping_condition = lambda op: False
        op = InfiniteOp(1.23, 0)
        final_decomp = list(_operator_decomposition_gen(op, stopping_condition, max_expansion=5))

        qml.assert_equal(op, final_decomp[0])
