# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.utils` module.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access
import pytest
from unittest.mock import MagicMock

import numpy as np

import pennylane as qml
import pennylane.utils as pu
import functools
import itertools

flat_dummy_array = np.linspace(-1, 1, 64)
test_shapes = [
    (64,),
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
    (16, 2, 2),
    (8, 2, 2, 2),
    (4, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2),
]

# global variables and functions
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)


U_toffoli = np.diag([1 for i in range(8)])
U_toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])


class TestFlatten:
    """Tests the flatten and unflatten functions"""

    @pytest.mark.parametrize("shape", test_shapes)
    def test_flatten(self, shape):
        """Tests that _flatten successfully flattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        flattened = np.array([x for x in pu._flatten(reshaped)])

        assert flattened.shape == flat_dummy_array.shape
        assert np.array_equal(flattened, flat_dummy_array)

    @pytest.mark.parametrize("shape", test_shapes)
    def test_unflatten(self, shape):
        """Tests that _unflatten successfully unflattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        unflattened = np.array([x for x in pu.unflatten(flat_dummy_array, reshaped)])

        assert unflattened.shape == reshaped.shape
        assert np.array_equal(unflattened, reshaped)

    def test_unflatten_error_unsupported_model(self):
        """Tests that unflatten raises an error if the given model is not supported"""

        with pytest.raises(TypeError, match="Unsupported type in the model"):
            model = lambda x: x  # not a valid model for unflatten
            pu.unflatten(flat_dummy_array, model)

    def test_unflatten_error_too_many_elements(self):
        """Tests that unflatten raises an error if the given iterable has
           more elements than the model"""

        reshaped = np.reshape(flat_dummy_array, (16, 2, 2))

        with pytest.raises(ValueError, match="Flattened iterable has more elements than the model"):
            pu.unflatten(np.concatenate([flat_dummy_array, flat_dummy_array]), reshaped)


class TestPauliEigs:
    """Tests for the auxiliary function to return the eigenvalues for Paulis"""

    paulix = np.array([[0, 1], [1, 0]])
    pauliy = np.array([[0, -1j], [1j, 0]])
    pauliz = np.array([[1, 0], [0, -1]])
    hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])

    standard_observables = [paulix, pauliy, pauliz, hadamard]

    matrix_pairs = [
        np.kron(x, y)
        for x, y in list(itertools.product(standard_observables, standard_observables))
    ]

    @pytest.mark.parametrize("pauli", standard_observables)
    def test_correct_eigenvalues_paulis(self, pauli):
        """Test the paulieigs function for one qubit"""
        assert np.array_equal(pu.pauli_eigs(1), np.diag(self.pauliz))

    @pytest.mark.parametrize("pauli_product", matrix_pairs)
    def test_correct_eigenvalues_pauli_kronecker_products_two_qubits(self, pauli_product):
        """Test the paulieigs function for two qubits"""
        assert np.array_equal(pu.pauli_eigs(2), np.diag(np.kron(self.pauliz, self.pauliz)))

    @pytest.mark.parametrize("pauli_product", matrix_pairs)
    def test_correct_eigenvalues_pauli_kronecker_products_three_qubits(self, pauli_product):
        """Test the paulieigs function for three qubits"""
        assert np.array_equal(
            pu.pauli_eigs(3), np.diag(np.kron(self.pauliz, np.kron(self.pauliz, self.pauliz)))
        )

    @pytest.mark.parametrize("depth", list(range(1, 6)))
    def test_cache_usage(self, depth):
        """Test that the right number of cachings have been executed after clearing the cache"""
        pu.pauli_eigs.cache_clear()
        pu.pauli_eigs(depth)
        total_runs = sum([2 ** x for x in range(depth)])
        assert functools._CacheInfo(depth - 1, depth, 128, depth) == pu.pauli_eigs.cache_info()


class TestArgumentHelpers:
    """Tests for auxiliary functions to help with parsing
    Python function arguments"""

    def test_no_default_args(self):
        """Test that empty dict is returned if function has
        no default arguments"""

        def dummy_func(a, b):  # pylint: disable=unused-argument
            pass

        res = pu._get_default_args(dummy_func)
        assert not res

    def test_get_default_args(self):
        """Test that default arguments are correctly extracted"""

        def dummy_func(
            a, b, c=8, d=[0, 0.65], e=np.array([4]), f=None
        ):  # pylint: disable=unused-argument,dangerous-default-value
            pass

        res = pu._get_default_args(dummy_func)
        expected = {"c": (2, 8), "d": (3, [0, 0.65]), "e": (4, np.array([4])), "f": (5, None)}

        assert res == expected

    def test_inv_dict(self):
        """Test _inv_dict correctly inverts a dictionary"""
        test_data = {"c": 8, "d": (0, 0.65), "e": "hi", "f": None, "g": 8}
        res = pu._inv_dict(test_data)
        expected = {8: {"g", "c"}, (0, 0.65): {"d"}, "hi": {"e"}, None: {"f"}}

        assert res == expected

    def test_inv_dict_unhashable_key(self):
        """Test _inv_dict raises an exception if a dictionary value is unhashable"""
        test_data = {"c": 8, "d": [0, 0.65], "e": "hi", "f": None, "g": 8}

        with pytest.raises(TypeError, match="unhashable type"):
            pu._inv_dict(test_data)


class TestExpand:
    """Tests multi-qubit operator expansion"""

    def test_expand_one(self, tol):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        # test applied to wire 0
        res = pu.expand(U, [0], 3)
        expected = np.kron(np.kron(U, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1
        res = pu.expand(U, [1], 3)
        expected = np.kron(np.kron(I, U), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 2
        res = pu.expand(U, [2], 3)
        expected = np.kron(np.kron(I, I), U)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_one_wires_list(self, tol):
        """Test that a 1 qubit gate correctly expands to 3 qubits."""
        # test applied to wire 0
        res = pu.expand(U, [0], [0, 4, 9])
        expected = np.kron(np.kron(U, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 4
        res = pu.expand(U, [4], [0, 4, 9])
        expected = np.kron(np.kron(I, U), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 9
        res = pu.expand(U, [9], [0, 4, 9])
        expected = np.kron(np.kron(I, I), U)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_consecutive_wires(self, tol):
        """Test that a 2 qubit gate on consecutive wires correctly
        expands to 4 qubits."""

        # test applied to wire 0+1
        res = pu.expand(U2, [0, 1], 4)
        expected = np.kron(np.kron(U2, I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1+2
        res = pu.expand(U2, [1, 2], 4)
        expected = np.kron(np.kron(I, U2), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 2+3
        res = pu.expand(U2, [2, 3], 4)
        expected = np.kron(np.kron(I, I), U2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_two_reversed_wires(self, tol):
        """Test that a 2 qubit gate on reversed consecutive wires correctly
        expands to 4 qubits."""

        # CNOT with target on wire 1
        res = pu.expand(CNOT, [1, 0], 4)
        rows = np.array([0, 2, 1, 3])
        expected = np.kron(np.kron(CNOT[:, rows][rows], I), I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_invalid_wires(self):
        """test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError, match="Invalid target subsystems provided in 'original_wires' argument"
        ):
            pu.expand(U2, [-1, 5], 4)

    def test_expand_invalid_matrix(self):
        """test exception raised if incorrect sized matrix provided/"""
        with pytest.raises(ValueError, match="Matrix parameter must be of size"):
            pu.expand(U, [0, 1], 4)

    def test_expand_three_consecutive_wires(self, tol):
        """Test that a 3 qubit gate on consecutive
        wires correctly expands to 4 qubits."""

        # test applied to wire 0,1,2
        res = pu.expand(U_toffoli, [0, 1, 2], 4)
        expected = np.kron(U_toffoli, I)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 1,2,3
        res = pu.expand(U_toffoli, [1, 2, 3], 4)
        expected = np.kron(I, U_toffoli)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_ascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive but ascending
        wires correctly expands to 4 qubits."""

        # test applied to wire 0,2,3
        res = pu.expand(U_toffoli, [0, 2, 3], 4)
        expected = (
            np.kron(SWAP, np.kron(I, I)) @ np.kron(I, U_toffoli) @ np.kron(SWAP, np.kron(I, I))
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 0,1,3
        res = pu.expand(U_toffoli, [0, 1, 3], 4)
        expected = (
            np.kron(np.kron(I, I), SWAP) @ np.kron(U_toffoli, I) @ np.kron(np.kron(I, I), SWAP)
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_three_nonconsecutive_nonascending_wires(self, tol):
        """Test that a 3 qubit gate on non-consecutive non-ascending
        wires correctly expands to 4 qubits"""

        # test applied to wire 3, 1, 2
        res = pu.expand(U_toffoli, [3, 1, 2], 4)
        # change the control qubit on the Toffoli gate
        rows = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        expected = np.kron(I, U_toffoli[:, rows][rows])
        assert np.allclose(res, expected, atol=tol, rtol=0)

        # test applied to wire 3, 0, 2
        res = pu.expand(U_toffoli, [3, 0, 2], 4)
        # change the control qubit on the Toffoli gate
        rows = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        expected = (
            np.kron(SWAP, np.kron(I, I))
            @ np.kron(I, U_toffoli[:, rows][rows])
            @ np.kron(SWAP, np.kron(I, I))
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    VECTOR1 = np.array([1, -1])
    ONES = np.array([1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0], 3, np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([1], 3, np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([2], 3, np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [0, 4, 7], np.kron(np.kron(ONES, VECTOR1), ONES)),
            ([7], [0, 4, 7], np.kron(np.kron(ONES, ONES), VECTOR1)),
            ([0], [0, 4, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([4], [4, 0, 7], np.kron(np.kron(VECTOR1, ONES), ONES)),
            ([7], [7, 4, 0], np.kron(np.kron(VECTOR1, ONES), ONES)),
        ],
    )
    def test_expand_vector_single_wire(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = pu.expand_vector(TestExpand.VECTOR1, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    VECTOR2 = np.array([1, 2, 3, 4])
    ONES = np.array([1, 1])

    @pytest.mark.parametrize(
        "original_wires,expanded_wires,expected",
        [
            ([0, 1], 3, np.kron(VECTOR2, ONES)),
            ([1, 2], 3, np.kron(ONES, VECTOR2)),
            ([0, 2], 3, np.array([1, 2, 1, 2, 3, 4, 3, 4])),
            ([0, 5], [0, 5, 9], np.kron(VECTOR2, ONES)),
            ([5, 9], [0, 5, 9], np.kron(ONES, VECTOR2)),
            ([0, 9], [0, 5, 9], np.array([1, 2, 1, 2, 3, 4, 3, 4])),
            ([9, 0], [0, 5, 9], np.array([1, 3, 1, 3, 2, 4, 2, 4])),
        ],
    )
    def test_expand_vector_two_wires(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = pu.expand_vector(TestExpand.VECTOR2, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_vector_invalid_wires(self):
        """Test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError, match="Invalid target subsystems provided in 'original_wires' argument"
        ):
            pu.expand_vector(TestExpand.VECTOR2, [-1, 5], 4)

    def test_expand_vector_invalid_vector(self):
        """Test exception raised if incorrect sized vector provided."""
        with pytest.raises(ValueError, match="Vector parameter must be of length"):
            pu.expand_vector(TestExpand.VECTOR1, [0, 1], 4)


class TestOperationRecorder:
    """Test the OperationRecorder class."""

    def test_context_adding(self, monkeypatch):
        """Test that the OperationRecorder is added to the list of contexts."""
        with pu.OperationRecorder() as recorder:
            assert recorder in qml.QueuingContext._active_contexts

        assert recorder not in qml.QueuingContext._active_contexts

    def test_circuit_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "PauliY(wires=[0])\n"
            + "PauliY(wires=[1])\n"
            + "RZ(0.4, wires=[0])\n"
            + "RZ(0.4, wires=[1])\n"
            + "CNOT(wires=[0, 1])\n"
            + "\n"
            + "Observables\n"
            + "==========\n"
        )

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(a, b, c):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)

            with pu.OperationRecorder() as recorder:
                ops = [
                    qml.PauliY(0),
                    qml.PauliY(1),
                    qml.RZ(c, wires=0),
                    qml.RZ(c, wires=1),
                    qml.CNOT(wires=[0, 1]),
                ]

            assert str(recorder) == expected_output
            assert recorder.queue == ops

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        circuit(0.1, 0.2, 0.4)

    def test_template_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "==========\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

        with pu.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output

    def test_template_with_return_integration(self):
        """Tests that the OperationRecorder integrates well with the
        core behaviour of PennyLane."""
        expected_output = (
            "Operations\n"
            + "==========\n"
            + "RZ(0, wires=[0])\n"
            + "RZ(3, wires=[0])\n"
            + "RZ(6, wires=[0])\n"
            + "RZ(9, wires=[0])\n"
            + "RZ(12, wires=[0])\n"
            + "\n"
            + "Observables\n"
            + "==========\n"
            + "var(PauliZ(wires=[0]))\n"
            + "sample(PauliX(wires=[1]))\n"
        )

        def template(x):
            for i in range(5):
                qml.RZ(i * x, wires=0)

            return qml.var(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        with pu.OperationRecorder() as recorder:
            template(3)

        assert str(recorder) == expected_output


@qml.template
def dummy_template(wires):
    """Dummy template for inv tests."""
    for wire in wires:
        qml.RX(1, wires=[wire])
        qml.RY(-1, wires=[wire])


def inverted_dummy_template_operations(wires):
    """The expected inverted operations for the dummy template."""
    ops = []

    for wire in reversed(wires):
        ops.append(qml.RY(-1, wires=[wire]).inv())
        ops.append(qml.RX(1, wires=[wire]).inv())

    return ops


class TestInv:
    """Test the template inversion function."""

    def test_inversion_without_context(self):
        """Test that a sequence of operations is properly inverted."""
        op_queue = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        inv_queue = [qml.PauliZ(0).inv(), qml.PauliY(0).inv(), qml.PauliX(0).inv()]

        inv_ops = pu.inv(op_queue)

        for inv_op, exp_op in zip(inv_ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_template_inversion_without_context(self):
        """Test that a template is properly inverted."""
        inv_queue = inverted_dummy_template_operations([0, 1, 2])

        inv_ops = pu.inv(dummy_template([0, 1, 2]))

        for inv_op, exp_op in zip(inv_ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_double_inversion(self):
        """Test that inverting twice changes nothing."""
        op_queue = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]

        inv_inv_ops = pu.inv(pu.inv(op_queue))

        for inv_inv_op, exp_op in zip(inv_inv_ops, op_queue):
            assert inv_inv_op.name == exp_op.name
            assert inv_inv_op.wires == exp_op.wires
            assert inv_inv_op.params == exp_op.params

    def test_template_double_inversion(self):
        """Test that inverting twice changes nothing for a template."""
        inv_inv_ops = pu.inv(pu.inv(dummy_template([0, 1, 2])))

        for inv_inv_op, exp_op in zip(inv_inv_ops, dummy_template([0, 1, 2])):
            assert inv_inv_op.name == exp_op.name
            assert inv_inv_op.wires == exp_op.wires
            assert inv_inv_op.params == exp_op.params

    def test_inversion_with_context(self):
        """Test that a sequence of operations is properly inverted when a context is present."""
        with pu.OperationRecorder() as rec:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv([qml.RX(1, wires=[0]), qml.RY(2, wires=[0]), qml.RZ(3, wires=[0])])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(3, wires=[0]).inv(),
            qml.RY(2, wires=[0]).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for inv_op, exp_op in zip(rec.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_non_queued_inversion_with_context(self):
        """Test that a sequence of operations is properly inverted when a context is present.
        Test that this also works for operations that were not queued."""
        inv_ops = [qml.RX(1, wires=[0]), qml.RY(2, wires=[0]), qml.RZ(3, wires=[0])]

        with pu.OperationRecorder() as rec:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv(inv_ops)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(3, wires=[0]).inv(),
            qml.RY(2, wires=[0]).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for inv_op, exp_op in zip(rec.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_mixed_inversion_with_context(self):
        """Test that a sequence of operations is properly inverted when a context is present.
        Test that this also works for operations that were not queued."""
        X0 = qml.PauliX(0)
        Z0 = qml.PauliZ(0)

        with pu.OperationRecorder() as rec:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv([X0, qml.RX(1, wires=[0]), Z0, qml.RY(2, wires=[0])])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RY(2, wires=[0]).inv(),
            qml.PauliZ(0).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.PauliX(0).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for inv_op, exp_op in zip(rec.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_mixed_inversion_with_nested_context(self):
        """Test that a sequence of operations is properly inverted when a nested context is present.
        Test that this also works for operations that were not queued."""
        X0 = qml.PauliX(0)
        Z0 = qml.PauliZ(0)

        with pu.OperationRecorder() as rec1:
            with pu.OperationRecorder() as rec2:
                qml.Hadamard(wires=[0])
                qml.CNOT(wires=[0, 1])
                pu.inv([X0, qml.RX(1, wires=[0]), Z0, qml.RY(2, wires=[0])])
                qml.CNOT(wires=[0, 1])
                qml.Hadamard(wires=[0])

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RY(2, wires=[0]).inv(),
            qml.PauliZ(0).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.PauliX(0).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for inv_op, exp_op in zip(rec1.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

        for inv_op, exp_op in zip(rec2.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_template_inversion_with_context(self):
        """Test that a template is properly inverted when a context is present."""
        with pu.OperationRecorder() as rec:
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv(dummy_template([0, 1, 2]))
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            *inverted_dummy_template_operations([0, 1, 2]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        for inv_op, exp_op in zip(rec.queue, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_inversion_with_qnode(self):
        """Test that a sequence of operations is properly inverted when inside a QNode."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qfunc():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv([qml.RX(1, wires=[0]), qml.RY(2, wires=[0]), qml.RZ(3, wires=[0])])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

            return qml.expval(qml.PauliZ(0))

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(3, wires=[0]).inv(),
            qml.RY(2, wires=[0]).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        qfunc()

        for inv_op, exp_op in zip(qfunc.ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_non_queued_inversion_with_qnode(self):
        """Test that a sequence of operations is properly inverted inside a QNode.
        Test that this also works for operations that were not queued."""
        inv_ops = [qml.RX(1, wires=[0]), qml.RY(2, wires=[0]), qml.RZ(3, wires=[0])]

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qfunc():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv(inv_ops)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

            return qml.expval(qml.PauliZ(0))

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(3, wires=[0]).inv(),
            qml.RY(2, wires=[0]).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        qfunc()

        for inv_op, exp_op in zip(qfunc.ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_mixed_inversion_with_qnode(self):
        """Test that a sequence of operations is properly inverted inside a QNode.
        Test that this also works for operations of queued and non-queued operations."""
        X0 = qml.PauliX(0)
        Z0 = qml.PauliZ(0)
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qfunc():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv([X0, qml.RX(1, wires=[0]), Z0, qml.RY(2, wires=[0])])
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

            return qml.expval(qml.PauliZ(0))

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            qml.RY(2, wires=[0]).inv(),
            qml.PauliZ(0).inv(),
            qml.RX(1, wires=[0]).inv(),
            qml.PauliX(0).inv(),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        qfunc()

        for inv_op, exp_op in zip(qfunc.ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_template_inversion_with_qnode(self):
        """Test that a template is properly inverted when inside a QNode."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qfunc():
            qml.Hadamard(wires=[0])
            qml.CNOT(wires=[0, 1])
            pu.inv(dummy_template([0, 1]))
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=[0])

            return qml.expval(qml.PauliZ(0))

        inv_queue = [
            qml.Hadamard(wires=[0]),
            qml.CNOT(wires=[0, 1]),
            *inverted_dummy_template_operations([0, 1]),
            qml.CNOT(wires=[0, 1]),
            qml.Hadamard(wires=[0]),
        ]

        qfunc()

        for inv_op, exp_op in zip(qfunc.ops, inv_queue):
            assert inv_op.name == exp_op.name
            assert inv_op.wires == exp_op.wires
            assert inv_op.params == exp_op.params

    def test_argument_wrapping(self):
        """Test that a single operation can be given to inv and is properly inverted."""
        op = qml.PauliX(0)
        exp_op = qml.PauliX(0).inv()

        inv_ops = pu.inv(op)

        assert inv_ops[0].name == exp_op.name
        assert inv_ops[0].wires == exp_op.wires
        assert inv_ops[0].params == exp_op.params

    @pytest.mark.parametrize("arg", [2.3, object()])
    def test_argument_type_error(self, arg):
        """Test that the proper error is raised when the argument type is wrong."""
        with pytest.raises(ValueError, match="The provided operation_list is not iterable"):
            pu.inv(arg)

    def test_argument_none_error(self):
        """Test that the proper error is raised when the argument type is wrong."""
        with pytest.raises(
            ValueError,
            match="None was passed as an argument to inv. "
            + "This could happen if inversion of a template without the template decorator is attempted",
        ):
            pu.inv(None)

    def test_callable_argument_error(self):
        """Test that the proper error is raised when the argument is a function."""

        def func(x):
            return x

        with pytest.raises(
            ValueError, match="A function was passed as an argument to inv. ",
        ):
            pu.inv(func)

    @pytest.mark.parametrize("arg", [[1, 2, 3], [qml.PauliX(0), qml.PauliY(1), "Test"], "Test",])
    def test_non_operations_in_list(self, arg):
        """Test that the proper error is raised when the argument does not only contain operations."""
        with pytest.raises(
            ValueError, match="The given operation_list does not only contain Operations"
        ):
            pu.inv(arg)
