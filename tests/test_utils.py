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

import pennylane.utils as pu

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
