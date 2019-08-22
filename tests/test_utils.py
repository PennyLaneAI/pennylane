# Copyright 2018 Xanadu Quantum Technologies Inc.

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

import numpy as np

import pennylane as qml
import pennylane.utils as pu

from pennylane.operation import Sample, Variance, Expectation


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


class TestArgumentHelpers:
    """Tests for auxiliary functions to help with parsing
    Python function arguments"""

    def test_no_default_args(self):
        """Test that empty dict is returned if function has
        no default arguments"""

        def dummy_func(a, b): # pylint: disable=unused-argument
            pass

        res = pu._get_default_args(dummy_func)
        assert not res

    def test_get_default_args(self):
        """Test that default arguments are correctly extracted"""

        def dummy_func(a, b, c=8, d=[0, 0.65], e=np.array([4]), f=None):  # pylint: disable=unused-argument,dangerous-default-value
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
            ValueError, match="Invalid target subsystems provided in 'wires' argument."
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


class TestCircuitGraph:
    """Test conversion of queues to DAGs"""

    def test_no_dependence(self):
        """Test case where operations do not depend on each other.
        This should result in a graph with no edges."""

        queue = [qml.RX(0.43, wires=0, do_queue=False), qml.RY(0.35, wires=1, do_queue=False)]

        obs = []

        res = pu.CircuitGraph(queue, obs).graph
        assert len(res) == 2
        assert not res.edges()

    def test_dependence(self):
        """Test a more complex example containing operations
        that do depend on the result of previous operations"""

        queue = [
            qml.RX(0.43, wires=0, do_queue=False),
            qml.RY(0.35, wires=1, do_queue=False),
            qml.RZ(0.35, wires=2, do_queue=False),
            qml.CNOT(wires=[0, 1], do_queue=False),
            qml.Hadamard(wires=2, do_queue=False),
            qml.CNOT(wires=[2, 0], do_queue=False),
            qml.PauliX(wires=1, do_queue=False),
        ]

        obs = [
            qml.expval(qml.PauliX(wires=0, do_queue=False)),
            qml.expval(qml.Hermitian(np.identity(4), wires=[1, 2], do_queue=False)),
        ]

        circuit = pu.CircuitGraph(queue, obs)
        res = circuit.graph
        assert len(res) == 9

        nodes = res.nodes().data()

        # the three rotations should be starting nodes in the graph
        assert nodes[0]["name"] == "RX"
        assert nodes[0]["op"] == queue[0]

        assert nodes[1]["name"] == "RY"
        assert nodes[1]["op"] == queue[1]

        assert nodes[2]["name"] == "RZ"
        assert nodes[2]["op"] == queue[2]

        # node 0 and node 1 should then connect to the CNOT gate
        assert nodes[3]["name"] == "CNOT"
        assert nodes[3]["op"] == queue[3]
        assert (0, 3) in res.edges()
        assert (1, 3) in res.edges()

        # RZ gate connects directly to a hadamard gate
        assert nodes[4]["name"] == "Hadamard"
        assert nodes[4]["op"] == queue[4]
        assert (2, 4) in res.edges()

        # hadamard gate and CNOT gate feed straight into another CNOT
        assert nodes[5]["name"] == "CNOT"
        assert nodes[5]["op"] == queue[5]
        assert (4, 5) in res.edges()
        assert (3, 5) in res.edges()

        # finally, the first CNOT also connects to a PauliX gate
        assert nodes[6]["name"] == "PauliX"
        assert nodes[6]["op"] == queue[6]
        assert (3, 6) in res.edges()

        # Measurements
        # PauliX is measured on the output of the second CNOT
        assert nodes[7]["name"] == "PauliX"
        assert nodes[7]["op"] == obs[0]
        assert nodes[7]["op"].return_type == Expectation
        assert (5, 7) in res.edges()

        # Hermitian is measured on the output of the second CNOT and the PauliX
        assert nodes[8]["name"] == "Hermitian"
        assert nodes[8]["op"] == obs[1]
        assert nodes[8]["op"].return_type == Expectation
        assert (5, 8) in res.edges()
        assert (6, 8) in res.edges()

        # Finally, checking the adjacency of the returned DAG:
        assert sorted(res.edges()) == [
            (0, 3),
            (1, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (5, 7),
            (5, 8),
            (6, 8),
        ]
