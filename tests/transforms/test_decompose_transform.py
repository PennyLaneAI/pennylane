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

import numpy as np
import pytest

import pennylane as qp
import pennylane.numpy as qnp
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.operation import Operation
from pennylane.ops import Conditional, MidMeasure
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
        return [qp.PauliX(self.wires), qp.PauliY(self.wires)]


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

    gate_set_inputs = [
        None,
        "RX",
        ["RX"],
        ("RX",),
        {"RX"},
        qp.RX,
        [qp.RX],
        (qp.RX,),
        {qp.RX},
        {qp.RX: 1.0},
    ]

    iterables_test = [
        (
            [qp.Hadamard(0)],
            {qp.RX, qp.RZ},
            [qp.RZ(qnp.pi / 2, 0), qp.RX(qnp.pi / 2, 0), qp.RZ(qnp.pi / 2, 0)],
            None,
        ),
        (
            [qp.SWAP(wires=[0, 1])],
            {qp.CNOT},
            [qp.CNOT([0, 1]), qp.CNOT([1, 0]), qp.CNOT([0, 1])],
            None,
        ),
        ([qp.Toffoli([0, 1, 2])], {qp.Toffoli}, [qp.Toffoli([0, 1, 2])], None),
        (
            [qp.Hadamard(0)],
            {qp.RX: 1, qp.RZ: 2},
            [qp.RZ(qnp.pi / 2, 0), qp.RX(qnp.pi / 2, 0), qp.RZ(qnp.pi / 2, 0)],
            {
                "type": UserWarning,
                "msg": "Gate weights were provided to a non-graph-based decomposition.",
            },
        ),
        (
            [qp.Toffoli([0, 1, 2]), qp.ops.MidMeasure(0)],
            {qp.Toffoli},
            [qp.Toffoli([0, 1, 2]), qp.ops.MidMeasure(0)],
            {
                "type": UserWarning,
                "msg": "MidMeasure",
            },
        ),
    ]

    callables_test = [
        (
            [qp.Hadamard(0)],
            lambda op: "RX" in op.name,
            [qp.RZ(qnp.pi / 2, 0), qp.RX(qnp.pi / 2, 0), qp.RZ(qnp.pi / 2, 0)],
            "RZ",
        ),
        (
            [qp.Toffoli([0, 1, 2])],
            lambda op: len(op.wires) <= 2,
            [
                qp.Hadamard(wires=[2]),
                qp.CNOT(wires=[1, 2]),
                qp.ops.op_math.Adjoint(qp.T(wires=[2])),
                qp.CNOT(wires=[0, 2]),
                qp.T(wires=[2]),
                qp.CNOT(wires=[1, 2]),
                qp.ops.op_math.Adjoint(qp.T(wires=[2])),
                qp.CNOT(wires=[0, 2]),
                qp.T(wires=[2]),
                qp.T(wires=[1]),
                qp.CNOT(wires=[0, 1]),
                qp.Hadamard(wires=[2]),
                qp.T(wires=[0]),
                qp.ops.op_math.Adjoint(qp.T(wires=[1])),
                qp.CNOT(wires=[0, 1]),
            ],
            None,
        ),
    ]

    @pytest.mark.parametrize("gate_set", gate_set_inputs)
    def test_different_input_formats(self, gate_set):
        """Tests that gate sets of different types are handled correctly"""
        tape = qp.tape.QuantumScript([qp.RX(0, wires=[0])])
        if isinstance(gate_set, dict):
            with pytest.raises(
                UserWarning, match="Gate weights were provided to a non-graph-based decomposition."
            ):
                (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
                qp.assert_equal(tape, decomposed_tape)
        else:
            (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
            qp.assert_equal(tape, decomposed_tape)

    def test_stopping_cond_without_gate_set(self):
        gate_set = None

        def stopping_condition(op):
            return op.name in ("RX")

        tape = qp.tape.QuantumScript([qp.RX(0, wires=[0])])

        (decomposed_tape,), _ = decompose(
            tape, gate_set=gate_set, stopping_condition=stopping_condition
        )
        qp.assert_equal(tape, decomposed_tape)

        def stopping_condition_2(op):
            return op.name in ("CX")

        with pytest.raises(
            UserWarning, match="Operator RX does not define a decomposition to the target gate set"
        ):
            decompose(tape, gate_set=gate_set, stopping_condition=stopping_condition_2)

    def test_user_warning(self):
        """Tests that user warning is raised if operator does not have a valid decomposition"""
        tape = qp.tape.QuantumScript([qp.RX(0, wires=[0])])
        with pytest.warns(UserWarning, match="does not define a decomposition"):
            decompose(tape, stopping_condition=lambda op: op.name not in {"RX"})

    def test_infinite_decomposition_loop(self):
        """Test that a recursion error is raised if decomposition enters an infinite loop."""
        tape = qp.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(RecursionError, match=r"Reached recursion limit trying to decompose"):
            decompose(tape, stopping_condition=lambda obj: obj.has_matrix)

    @pytest.mark.parametrize(
        "initial_ops, gate_set, expected_ops, warning_or_error_pattern", iterables_test
    )
    def test_iterable_gate_set(self, initial_ops, gate_set, expected_ops, warning_or_error_pattern):
        """Tests that gate sets defined with iterables decompose correctly"""
        tape = qp.tape.QuantumScript(initial_ops)

        if warning_or_error_pattern is not None:
            with pytest.raises(
                warning_or_error_pattern["type"], match=warning_or_error_pattern["msg"]
            ):
                (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
                expected_tape = qp.tape.QuantumScript(expected_ops)
                qp.assert_equal(decomposed_tape, expected_tape)
        else:
            (decomposed_tape,), _ = decompose(tape, gate_set=gate_set)
            expected_tape = qp.tape.QuantumScript(expected_ops)
            qp.assert_equal(decomposed_tape, expected_tape)

    @pytest.mark.parametrize("initial_ops, gate_set, expected_ops, warning_pattern", callables_test)
    def test_callable_stopping_condition(
        self, initial_ops, gate_set, expected_ops, warning_pattern
    ):
        """Tests that stopping_condition defined by callables decompose correctly"""
        tape = qp.tape.QuantumScript(initial_ops)

        if warning_pattern is not None:
            with pytest.warns(UserWarning, match=warning_pattern):
                (decomposed_tape,), _ = decompose(tape, stopping_condition=gate_set)
        else:
            (decomposed_tape,), _ = decompose(tape, stopping_condition=gate_set)

        expected_tape = qp.tape.QuantumScript(expected_ops)

        qp.assert_equal(decomposed_tape, expected_tape)

    def test_callable_gate_set_deprecated(self):
        """Tests that passing a callable to gate_set is deprecated."""

        with pytest.warns(PennyLaneDeprecationWarning, match="Passing a function to the gate_set"):
            tape = qp.tape.QuantumScript([qp.Hadamard(0)])
            [decomp], _ = decompose(tape, gate_set=lambda op: op.name in {"RZ", "RX"})

        expected = qp.tape.QuantumScript(
            [qp.RZ(qnp.pi / 2, 0), qp.RX(qnp.pi / 2, 0), qp.RZ(qnp.pi / 2, 0)]
        )
        qp.assert_equal(decomp, expected)

    def test_decompose_with_mcm(self):
        """Tests that circuits and decomposition rules containing MCMs are supported."""

        class CustomOp(Operation):  # pylint: disable=too-few-public-methods
            resource_keys = set()

            @property
            def resource_params(self) -> dict:
                return {}

            def decomposition(self):
                ops = [qp.H(0)]
                m = qp.measure(0)
                ops += m.measurements
                ops.append(qp.ops.Conditional(m0, qp.H(1)))
                return ops

        m0 = qp.measure(0)
        tape = qp.tape.QuantumScript(
            [
                CustomOp(wires=(0, 1)),
                m0.measurements[0],
                qp.ops.Conditional(m0, qp.X(0)),
                qp.ops.Conditional(m0, qp.RX(0.5, wires=0)),
            ]
        )
        [decomposed_tape], _ = qp.transforms.decompose(
            [tape], gate_set={qp.RX, qp.RZ, MidMeasure}
        )
        assert len(decomposed_tape.operations) == 10

        with qp.queuing.AnnotatedQueue() as q:
            qp.RZ(np.pi / 2, wires=0)
            qp.RX(np.pi / 2, wires=0)
            qp.RZ(np.pi / 2, wires=0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.RZ)(np.pi / 2, wires=1)
            qp.cond(m0, qp.RX)(np.pi / 2, wires=1)
            qp.cond(m0, qp.RZ)(np.pi / 2, wires=1)
            m1 = qp.measure(0)
            qp.cond(m1, qp.RX)(np.pi, wires=0)
            qp.cond(m1, qp.RX)(0.5, wires=0)

        qp.assert_equal(decomposed_tape.operations[0], q.queue[0])
        qp.assert_equal(decomposed_tape.operations[1], q.queue[1])
        qp.assert_equal(decomposed_tape.operations[2], q.queue[2])
        assert isinstance(decomposed_tape.operations[4], Conditional)
        assert isinstance(decomposed_tape.operations[5], Conditional)
        assert isinstance(decomposed_tape.operations[6], Conditional)
        assert isinstance(decomposed_tape.operations[8], Conditional)
        assert isinstance(decomposed_tape.operations[9], Conditional)
        qp.assert_equal(decomposed_tape.operations[4].base, q.queue[4].base)
        qp.assert_equal(decomposed_tape.operations[5].base, q.queue[5].base)
        qp.assert_equal(decomposed_tape.operations[6].base, q.queue[6].base)
        qp.assert_equal(decomposed_tape.operations[8].base, q.queue[8].base)
        qp.assert_equal(decomposed_tape.operations[9].base, q.queue[9].base)
        assert isinstance(decomposed_tape.operations[3], MidMeasure)
        assert isinstance(decomposed_tape.operations[7], MidMeasure)


def test_null_postprocessing():
    """Tests the null postprocessing function in the decompose transform"""
    tape = qp.tape.QuantumScript([qp.Hadamard(0), qp.RX(0, 0)])
    (_,), fn = qp.transforms.decompose(tape, gate_set={qp.RX, qp.RZ})
    assert fn((1,)) == 1


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @pytest.mark.parametrize("op", (qp.PauliX(0), qp.RX(1.2, wires=0), qp.QFT(wires=range(3))))
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
        qp.assert_equal(casted_to_list[0], qp.PauliX("a"))
        qp.assert_equal(casted_to_list[1], qp.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        def stopping_condition(op):
            return op.has_matrix

        class RaggedDecompositionOp(Operation):
            """class with a ragged decomposition."""

            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qp.S(self.wires), qp.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(final_decomp) == 5
        qp.assert_equal(final_decomp[0], qp.PauliX("a"))
        qp.assert_equal(final_decomp[1], qp.PauliY("a"))
        qp.assert_equal(final_decomp[2], qp.S("a"))
        qp.assert_equal(final_decomp[3], qp.adjoint(qp.PauliY("a")))
        qp.assert_equal(final_decomp[4], qp.adjoint(qp.PauliX("a")))

    def test_operator_decomposition_gen_max_depth_reached(self):
        """Tests whether max depth reached flag gets activated"""

        stopping_condition = lambda op: False
        op = InfiniteOp(1.23, 0)
        final_decomp = list(_operator_decomposition_gen(op, stopping_condition, max_expansion=5))

        qp.assert_equal(op, final_decomp[0])

    @pytest.mark.unit
    def test_invalid_gate_set(self):
        """Tests that an invalid gate set raises a TypeError."""

        tape = qp.tape.QuantumScript([])
        with pytest.raises(TypeError, match="Invalid gate_set type."):
            qp.transforms.decompose(tape, gate_set=123)
