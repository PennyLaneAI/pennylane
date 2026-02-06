# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the the arithmetic qubit operations
"""
# pylint: disable=too-many-arguments
import copy
import itertools

import numpy as np
import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.wires import Wires

label_data = [
    (qp.QubitCarry(wires=(0, 1, 2, 3)), "QubitCarry"),
    (qp.QubitSum(wires=(0, 1, 2)), "Σ"),
]


@pytest.mark.parametrize("op, label", label_data)
def test_label(op, label):
    assert op.label() == label
    assert op.label(decimals=2) == label


class TestQubitCarry:
    """Tests the QubitCarry operator."""

    @pytest.mark.parametrize(
        "wires,input_string,output_string,expand",
        [
            ([0, 1, 2, 3], "0000", "0000", True),
            ([0, 1, 2, 3], "0001", "0001", True),
            ([0, 1, 2, 3], "0010", "0010", True),
            ([0, 1, 2, 3], "0011", "0011", True),
            ([0, 1, 2, 3], "0100", "0110", True),
            ([0, 1, 2, 3], "0101", "0111", True),
            ([0, 1, 2, 3], "0110", "0101", True),
            ([0, 1, 2, 3], "0111", "0100", True),
            ([0, 1, 2, 3], "1000", "1000", True),
            ([0, 1, 2, 3], "1001", "1001", True),
            ([0, 1, 2, 3], "1010", "1011", True),
            ([0, 1, 2, 3], "1011", "1010", True),
            ([0, 1, 2, 3], "1100", "1111", True),
            ([0, 1, 2, 3], "1101", "1110", True),
            ([0, 1, 2, 3], "1110", "1101", True),
            ([0, 1, 2, 3], "1111", "1100", True),
            ([3, 1, 2, 0], "0110", "1100", True),
            ([3, 2, 0, 1], "1010", "0110", True),
            ([0, 1, 2, 3], "0000", "0000", False),
            ([0, 1, 2, 3], "0001", "0001", False),
            ([0, 1, 2, 3], "0010", "0010", False),
            ([0, 1, 2, 3], "0011", "0011", False),
            ([0, 1, 2, 3], "0100", "0110", False),
            ([0, 1, 2, 3], "0101", "0111", False),
            ([0, 1, 2, 3], "0110", "0101", False),
            ([0, 1, 2, 3], "0111", "0100", False),
            ([0, 1, 2, 3], "1000", "1000", False),
            ([0, 1, 2, 3], "1001", "1001", False),
            ([0, 1, 2, 3], "1010", "1011", False),
            ([0, 1, 2, 3], "1011", "1010", False),
            ([0, 1, 2, 3], "1100", "1111", False),
            ([0, 1, 2, 3], "1101", "1110", False),
            ([0, 1, 2, 3], "1110", "1101", False),
            ([0, 1, 2, 3], "1111", "1100", False),
            ([3, 1, 2, 0], "0110", "1100", False),
            ([3, 2, 0, 1], "1010", "0110", False),
        ],
    )
    def test_output(self, wires, input_string, output_string, expand, mocker):
        """Test if ``QubitCarry`` produces the right output and is expandable."""
        dev = qp.device("default.qubit", wires=4)
        spy = mocker.spy(qp.QubitCarry, "decomposition")

        with qp.queuing.AnnotatedQueue() as q:
            for i, letter in enumerate(input_string):
                if letter == "1":
                    qp.PauliX(i)
            qp.QubitCarry(wires=wires)
            qp.probs(wires=[0, 1, 2, 3])

        tape = qp.tape.QuantumScript.from_queue(q)
        if expand:
            tape = tape.expand()
        result = dev.execute(tape)
        result = np.argmax(result)
        result = format(result, "04b")
        assert result == output_string

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_superposition(self):
        """Test if ``QubitCarry`` works for superposition input states."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit():
            qp.PauliX(wires=1)
            qp.Hadamard(wires=2)
            qp.QubitCarry(wires=[0, 1, 2, 3])
            return qp.probs(wires=3)

        result = circuit()
        assert np.allclose(result, 0.5)

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""

        res_static = qp.QubitCarry.compute_matrix()
        res_dynamic = qp.QubitCarry(wires=[0, 1, 2, 3]).matrix()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)


class TestQubitSum:
    """Tests for the QubitSum operator"""

    # fmt: off
    @pytest.mark.parametrize(
        "wires,input_state,output_state,expand",
        [
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], True),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], True),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], True),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], True),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], True),
            ([0, 1, 2], [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
                         np.sqrt(1 / 8), np.sqrt(1 / 8)],
             [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
              np.sqrt(1 / 8), np.sqrt(1 / 8)], True),
            ([0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], False),
            ([0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], False),
            ([2, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], False),
            ([1, 2, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], False),
            ([0, 1, 2], [0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0], [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0], False),
            ([0, 1, 2], [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
                         np.sqrt(1 / 8), np.sqrt(1 / 8)],
             [np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8), np.sqrt(1 / 8),
              np.sqrt(1 / 8), np.sqrt(1 / 8)], False),
        ],
    )
    # fmt: on
    def test_output(self, wires, input_state, output_state, expand, mocker):
        """Test if ``QubitSum`` produces the correct output"""
        dev = qp.device("default.qubit", wires=3)
        spy = mocker.spy(qp.QubitSum, "decomposition")

        if expand:
            ops = [
                qp.StatePrep(input_state, wires=[0, 1, 2]),
                *qp.tape.QuantumScript(qp.QubitSum(wires=wires).decomposition()),
            ]
        else:
            ops = [qp.StatePrep(input_state, wires=[0, 1, 2]), qp.QubitSum(wires=wires)]

        tape = qp.tape.QuantumScript(ops, [qp.state()])
        result = dev.execute(tape)
        assert np.allclose(result, output_state)

        # checks that decomposition is only used when intended
        assert expand is (len(spy.call_args_list) != 0)

    def test_adjoint(self):
        """Test the adjoint method of QubitSum by reconstructing the unitary matrix and checking
        if it is equal to qp.QubitSum's matrix representation (recall that the operation is self-adjoint)
        """
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def f(state):
            qp.StatePrep(state, wires=range(3))
            qp.adjoint(qp.QubitSum)(wires=range(3))
            return qp.probs(wires=range(3))

        u = np.array([f(state) for state in np.eye(2**3)]).T
        assert np.allclose(u, qp.QubitSum.compute_matrix())

    def test_matrix_representation(self, tol):
        """Test that the matrix representation is defined correctly"""

        res_static = qp.QubitSum.compute_matrix()
        res_dynamic = qp.QubitSum(wires=[0, 1, 2]).matrix()
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)


class TestIntegerComparator:
    """Tests for the IntegerComparator"""

    # pylint: disable=protected-access
    def test_flatten_unflatten(self):
        """Tests the flatten and unflatten methods"""
        wires = qp.wires.Wires((0, 1, 2, 3))
        work_wires = qp.wires.Wires(4)
        op = qp.IntegerComparator(
            2,
            geq=False,
            wires=(0, 1, 2, 3),
            work_wires=(4),
        )

        data, metadata = op._flatten()
        assert data == tuple()
        assert len(metadata) == 2
        assert metadata[0] == wires
        assert metadata[1][0] == ("work_wires", work_wires)
        assert metadata[1][1] == ("value", 2)
        assert metadata[1][2] == ("geq", False)

        # check hashable
        assert hash(metadata)

        new_op = type(op)._unflatten(*op._flatten())
        qp.assert_equal(new_op, op)
        assert new_op is not op

    @pytest.mark.parametrize(
        "value,geq,wires,work_wires,expected_error_message",
        [
            (
                4.20,
                False,
                [0, 1, 2],
                None,
                "The compared value must be an int. Got <class 'float'>.",
            ),
            (2, True, None, None, "Must specify wires that the operation acts on."),
            (
                2,
                True,
                [1],
                None,
                r"IntegerComparator: wrong number of wires. 1 wire\(s\) given. Need at least 2.",
            ),
            (
                2,
                False,
                [0, 1, 2],
                [2],
                "The work wires must be different from the control and target wires",
            ),
        ],
    )
    def test_invalid_mixed_polarity_controls(
        self, value, geq, wires, work_wires, expected_error_message
    ):
        """Test if IntegerComparator properly handles invalid mixed-polarity
        control values."""
        with pytest.raises(ValueError, match=expected_error_message):
            qp.IntegerComparator(value, geq=geq, wires=wires, work_wires=work_wires).matrix()

    def test_compute_matrix_geq_True(self):
        """Test compute_matrix for geq=True"""
        mat1 = qp.IntegerComparator.compute_matrix(value=2, control_wires=[0, 1], geq=True)
        mat2 = np.zeros((8, 8))

        mat2[0, 0] = 1
        mat2[1, 1] = 1
        mat2[2, 2] = 1
        mat2[3, 3] = 1
        mat2[4, 5] = 1
        mat2[5, 4] = 1
        mat2[6, 7] = 1
        mat2[7, 6] = 1

        assert np.allclose(mat1, mat2)

    def test_compute_matrix_geq_False(self):
        """Test compute_matrix for geq=False"""
        mat1 = qp.IntegerComparator.compute_matrix(value=2, control_wires=[0, 1], geq=False)
        mat2 = np.zeros((8, 8))

        mat2[0, 1] = 1
        mat2[1, 0] = 1
        mat2[2, 3] = 1
        mat2[3, 2] = 1
        mat2[4, 4] = 1
        mat2[5, 5] = 1
        mat2[6, 6] = 1
        mat2[7, 7] = 1

        assert np.allclose(mat1, mat2)

    @pytest.mark.parametrize(
        "value,control_wires,geq,expected_error_message",
        [
            (None, [0, 1], True, "The value to compare to must be specified."),
            (4.20, [0, 1], False, "The compared value must be an int. Got <class 'float'>."),
            (4, None, True, "Must specify the control wires."),
        ],
    )
    def test_invalid_args_compute_matrix(self, value, control_wires, geq, expected_error_message):
        """Test if compute_matrix properly handles invalid arguments."""
        with pytest.raises(ValueError, match=expected_error_message):
            qp.IntegerComparator.compute_matrix(value=value, control_wires=control_wires, geq=geq)

    def test_compute_matrix_large_value(self):
        """Test if compute_matrix properly handles values exceeding the Hilbert space of the control
        wires."""

        mat1 = qp.IntegerComparator.compute_matrix(value=10, control_wires=[0, 1], geq=True)
        mat2 = np.eye(8)

        assert np.allclose(mat1, mat2)

    def test_compute_matrix_value_zero(self):
        """Test if compute_matrix properly handles value=0 when geq=False."""

        mat1 = qp.IntegerComparator.compute_matrix(value=0, control_wires=[0, 1], geq=False)
        mat2 = np.eye(8)

        assert np.allclose(mat1, mat2)

    def test_adjoint_method(self):
        """Test ``adjoint()`` method."""
        op = qp.IntegerComparator(2, wires=(0, 1, 2, 3))
        adj_op = copy.copy(op)
        for _ in range(4):
            adj_op = adj_op.adjoint()

            assert adj_op.name == op.name

    def test_control_wires(self):
        """Test ``control_wires`` attribute for non-parametrized operations."""
        op, control_wires = qp.IntegerComparator(2, wires=(0, 1, 2, 3)), Wires([0, 1, 2])
        assert op.control_wires == control_wires

    def test_label_method(self):
        """Test label method"""

        op = qp.IntegerComparator(2, wires=(0, 1, 2, 3))
        label = ">=2"

        assert op.label() == label
        assert op.label(decimals=2) == label

    @pytest.mark.parametrize(
        "value,wires,geq,expected_error_message",
        [
            (4.20, [0, 1, 2], False, "The compared value must be an int. Got <class 'float'>."),
            (2, None, True, "Must specify the wires that the operation acts on."),
            (
                2,
                [0],
                False,
                r"IntegerComparator: wrong number of wires. 1 wire\(s\) given. Need at least 2.",
            ),
        ],
    )
    def test_invalid_args_compute_decomposition(self, value, wires, geq, expected_error_message):
        """Test if compute_matrix properly handles invalid arguments."""
        with pytest.raises(ValueError, match=expected_error_message):
            qp.IntegerComparator.compute_decomposition(value, wires=wires, geq=geq)

    @pytest.mark.parametrize("geq", [True, False])
    @pytest.mark.parametrize("value", list(range(9)))
    @pytest.mark.parametrize("initial_state", list(itertools.product([0, 1], repeat=3)))
    def test_decomposition(self, value, geq, initial_state):
        """Tests the decomposition of the IntegerComparator operator."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.Projector(initial_state, wires=[0, 1, 2])
            qp.IntegerComparator.compute_decomposition(value, wires=[0, 1, 2, 3], geq=geq)
        tape = qp.tape.QuantumScript.from_queue(q)

        with qp.queuing.AnnotatedQueue() as q:
            qp.Projector(initial_state, wires=[0, 1, 2])
            val = int("".join(str(i) for i in initial_state), 2)
            if (geq and val >= value) or (not geq and val < value):
                qp.PauliX(3)
        equivalent_tape = qp.tape.QuantumScript.from_queue(q)

        matrix = qp.matrix(tape, wire_order=[0, 1, 2, 3])
        equivalent_matrix = qp.matrix(equivalent_tape, wire_order=[0, 1, 2, 3])
        assert qp.math.allclose(matrix, equivalent_matrix)

    @pytest.mark.parametrize("geq", [True, False])
    @pytest.mark.parametrize("value", list(range(9)))
    def test_decomposition_qfuncs(self, value, geq):
        """Tests the qfunc decompositions of the IntegerComparator operator."""
        op = qp.IntegerComparator(value, wires=[0, 1, 2, 3], geq=geq)
        for rule in qp.list_decomps(qp.IntegerComparator):
            _test_decomposition_rule(op, rule)

    @pytest.mark.integration
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_decomposition_graph_integration(self):
        """Tests that the decomposition is correct under the new system."""

        tape = qp.tape.QuantumScript([qp.IntegerComparator(42, wires=[0, 1, 2, 3, 4, 5, 6])])
        [decomp], _ = qp.transforms.decompose(
            tape,
            gate_set={qp.X, qp.CNOT, qp.Hadamard, qp.T, "Adjoint(T)", qp.RX, qp.RY, qp.RZ},
        )

        mat = qp.matrix(decomp, wire_order=[0, 1, 2, 3, 4, 5, 6])
        expected_mat = qp.matrix(tape, wire_order=[0, 1, 2, 3, 4, 5, 6])
        assert qp.math.allclose(mat, expected_mat)

    def test_decomposition_extraneous_value(self):
        """Test operator's ``compute_decomposition()`` method when ``value`` is such that
        decomposition is the identity matrix."""

        with qp.queuing.AnnotatedQueue() as q1:
            qp.IntegerComparator.compute_decomposition(10, wires=[0, 1, 2])
        tape1 = qp.tape.QuantumScript.from_queue(q1)
        assert all(isinstance(op, qp.Identity) for op in tape1.operations)

        with qp.queuing.AnnotatedQueue() as q2:
            qp.IntegerComparator.compute_decomposition(0, wires=[0, 1, 2], geq=False)
        tape2 = qp.tape.QuantumScript.from_queue(q2)
        assert all(isinstance(op, qp.Identity) for op in tape2.operations)

    # pylint: disable=use-implicit-booleaness-not-comparison
    def test_power(self):
        """Test ``pow`` method."""
        op = qp.IntegerComparator(3, wires=[0, 1, 2, 3])

        powers = range(8)
        for power in powers:
            op_pow = op.pow(power)
            assert op_pow == [] if power % 2 == 0 else [op]
