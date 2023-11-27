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
import functools
import itertools
import pytest

import numpy as np

import pennylane as qml
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


class TestFlatten:
    """Tests the flatten and unflatten functions"""

    @pytest.mark.parametrize("shape", test_shapes)
    def test_flatten(self, shape):
        """Tests that _flatten successfully flattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        flattened = np.array(list(pu._flatten(reshaped)))

        assert flattened.shape == flat_dummy_array.shape
        assert np.array_equal(flattened, flat_dummy_array)

    @pytest.mark.parametrize("shape", test_shapes)
    def test_unflatten(self, shape):
        """Tests that _unflatten successfully unflattens multidimensional arrays."""

        reshaped = np.reshape(flat_dummy_array, shape)
        unflattened = np.array(list(pu.unflatten(flat_dummy_array, reshaped)))

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

    def test_flatten_wires(self):
        """Tests flattening a Wires object."""
        wires = qml.wires.Wires([3, 4])
        wires_int = [3, 4]

        wires = qml.utils._flatten(wires)
        for i, wire in enumerate(wires):
            assert wires_int[i] == wire


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

    def test_correct_eigenvalues_paulis(self):
        """Test the paulieigs function for one qubit"""
        assert np.array_equal(pu.pauli_eigs(1), np.diag(self.pauliz))

    def test_correct_eigenvalues_pauli_kronecker_products_two_qubits(self):
        """Test the paulieigs function for two qubits"""
        assert np.array_equal(pu.pauli_eigs(2), np.diag(np.kron(self.pauliz, self.pauliz)))

    def test_correct_eigenvalues_pauli_kronecker_products_three_qubits(self):
        """Test the paulieigs function for three qubits"""
        assert np.array_equal(
            pu.pauli_eigs(3),
            np.diag(np.kron(self.pauliz, np.kron(self.pauliz, self.pauliz))),
        )

    @pytest.mark.parametrize("depth", list(range(1, 6)))
    def test_cache_usage(self, depth):
        """Test that the right number of cachings have been executed after clearing the cache"""
        pu.pauli_eigs.cache_clear()
        pu.pauli_eigs(depth)
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
        expected = {
            "c": (2, 8),
            "d": (3, [0, 0.65]),
            "e": (4, np.array([4])),
            "f": (5, None),
        }

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


class TestExpandVector:
    """Tests vector expansion to more wires"""

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

        res = pu.expand_vector(TestExpandVector.VECTOR1, original_wires, expanded_wires)

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
            ([0, 1], [0, 1], VECTOR2),
        ],
    )
    def test_expand_vector_two_wires(self, original_wires, expanded_wires, expected, tol):
        """Test that expand_vector works with a single-wire vector."""

        res = pu.expand_vector(TestExpandVector.VECTOR2, original_wires, expanded_wires)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_expand_vector_invalid_wires(self):
        """Test exception raised if unphysical subsystems provided."""
        with pytest.raises(
            ValueError,
            match="Invalid target subsystems provided in 'original_wires' argument",
        ):
            pu.expand_vector(TestExpandVector.VECTOR2, [-1, 5], 4)

    def test_expand_vector_invalid_vector(self):
        """Test exception raised if incorrect sized vector provided."""
        with pytest.raises(ValueError, match="Vector parameter must be of length"):
            pu.expand_vector(TestExpandVector.VECTOR1, [0, 1], 4)
