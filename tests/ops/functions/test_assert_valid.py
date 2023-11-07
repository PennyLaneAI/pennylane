# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains unit tests for ``qml.ops.functions.assert_valid``.
"""
# pylint: disable=too-few-public-methods, unused-argument
import pytest

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.functions import assert_valid


class TestDecompositionErrors:
    """Test assertions involving decompositions."""

    def test_bad_decomposition_output(self):
        """Test decomposition output must be a list of operators."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                qml.RX(1.2, wires=0)

        with pytest.raises(AssertionError, match=r"decomposition must be a list"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_bad_decomposition_queuing(self):
        """Test that output must match queued contents."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                qml.RX(1.2, wires=0)
                return [qml.RY(2.3, 0)]

        with pytest.raises(AssertionError, match="decomposition must match queued operations"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_expand_must_be_qscript(self):
        """Test that an error is raised if expand does not return a QuantumScript"""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qml.RY(2.3, 0)]

            def expand(self):
                return [qml.S(0)]

        with pytest.raises(AssertionError, match=r"expand must return a QuantumScript"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_decomposition_must_match_expand(self):
        """Test that decomposition and expand must match."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qml.RY(2.3, 0)]

            def expand(self):
                return qml.tape.QuantumScript([qml.S(0)])

        with pytest.raises(AssertionError, match="decomposition must match expansion"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_error_not_raised(self):
        """Test if has_decomposition is False but decomposition defined."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qml.RY(2.3, 0)]

            has_decomposition = False

        with pytest.raises(AssertionError, match="If has_decomposition is False"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)


class TestBadMatrix:
    """Tests involving matrix validation."""

    def test_error_not_raised(self):
        """Test that if has_matrix if False, then an error must be raised."""

        class BadMat(Operator):
            has_matrix = False

            def matrix(self):
                return np.eye(2)

        with pytest.raises(
            AssertionError, match="If has_matrix is False, the matrix method must raise"
        ):
            assert_valid(BadMat(wires=0), skip_pickle=True)

    def test_bad_matrix_shape(self):
        """Test an error if the matrix is of the wrong shape."""

        class BadMat(Operator):
            @staticmethod
            def compute_matrix():
                return np.eye(2)

        with pytest.raises(
            AssertionError, match=r"matrix must be two dimensional with shape \(4, 4\)"
        ):
            assert_valid(BadMat(wires=(0, 1)), skip_pickle=True)

    def test_matrix_not_tensorlike(self):
        """Test an error is raised if the matrix is not a TensorLike"""

        class BadMat(Operator):
            @staticmethod
            def compute_matrix():
                return "a"

        with pytest.raises(AssertionError, match=r"matrix must be a TensorLike"):
            assert_valid(BadMat(0), skip_pickle=True)


class TestBadCopyComparison:
    """Check errors invovling copy, deepcopy, and comparison."""

    def test_bad_comparison(self):
        """Test an operator that cannot be compared with standard qml.equal."""

        class BadComparison(Operator):
            def __init__(self, wires, val):
                self.hyperparameters["val"] = val
                super().__init__(wires)

        with pytest.raises(ValueError, match=r"The truth value of an array with more than one"):
            assert_valid(BadComparison(0, val=np.eye(2)))


def test_mismatched_mat_decomp():
    """Test that an error is raised if the matrix does not match the decomposition if both are defined."""

    class MisMatchedMatDecomp(Operator):
        @staticmethod
        def compute_matrix():
            return np.eye(2)

        def decomposition(self):
            return [qml.PauliX(0)]

    with pytest.raises(AssertionError, match=r"matrix and matrix from decomposition must match"):
        assert_valid(MisMatchedMatDecomp(0), skip_pickle=True)


def test_bad_eigenvalues_order():
    """Test that an error is raised if the order of eigenvalues does not match the diagonalizing gates."""

    class BadEigenDecomp(qml.PauliX):
        @staticmethod
        def compute_eigvals():
            return [-1, 1]

    with pytest.raises(
        AssertionError, match=r"eigenvalues and diagonalizing gates must be able to"
    ):
        assert_valid(BadEigenDecomp(0), skip_pickle=True)


class BadPickling0(Operator):
    def __init__(self, f, wires):
        super().__init__(wires)

        self.hyperparameters["f"] = f


def test_bad_pickling():
    """Test an error is raised in an operator cant be pickled."""

    with pytest.raises(AttributeError, match="Can't pickle local object"):
        assert_valid(BadPickling0(lambda x: x, wires=0))


def test_bad_bind_new_parameters():
    """Test a function that is not working with bind_new_parameters."""

    class NoBindNewParameters(Operator):
        num_params = 1

        def __init__(self, x, wires):
            super().__init__(x, wires)
            self.data = (1.0,)  # different x will not change data attribute

    with pytest.raises(
        AssertionError, match=r"bind_new_parameters must be able to update the operator"
    ):
        assert_valid(NoBindNewParameters(2.0, wires=0), skip_pickle=True)


def test_bad_wire_mapping():
    """Test that an error is raised if the wires cant be mapped with map_wires."""

    class BadWireMap(Operator):
        def __init__(self, op1):
            self.hyperparameters["op1"] = op1
            super().__init__(wires=op1.wires)

        @property
        def wires(self):
            return self.hyperparameters["op1"].wires

    with pytest.raises(AssertionError, match=r"wires must be mappable"):
        assert_valid(BadWireMap(qml.PauliX(0)), skip_pickle=True)


class TestPytree:
    def test_not_hashable_metadata(self):
        """Assert that an error is raised if metadata is not hashable."""

        class BadMetadata(Operator):
            def __init__(self, wires):
                super().__init__(wires)
                self.hyperparameters["test"] = ["a"]

        op = BadMetadata(0)
        with pytest.raises(AssertionError, match=r"metadata output from _flatten must be hashable"):
            assert_valid(op, skip_pickle=True)

    def test_badpytree(self):
        """Assert that an unpacking and repacking a pytree reproduces the original operation."""

        class BadPytree(Operator):
            def _flatten(self):
                return tuple(), (self.wires, tuple())

            def __init__(self, wires, val="a"):
                super().__init__(wires)
                self.hyperparameters["val"] = val

        op = BadPytree(wires=0, val="b")
        with pytest.raises(
            AssertionError,
            match=r"metadata and data must be able to reproduce the original operation",
        ):
            assert_valid(op)

        op = qml.adjoint(BadPytree(wires=0, val="b"))
        with pytest.raises(AssertionError, match=r"op must be a valid pytree."):
            assert_valid(op)

    def test_bad_leaves_ordering(self):
        """Test an error is raised if data and pytree leaves have a different ordering convention."""

        class BadLeavesOrdering(qml.ops.op_math.SProd):
            def _flatten(self):
                return (self.base, self.scalar), tuple()

            @classmethod
            def _unflatten(cls, data, _):
                return cls(data[1], data[0])

        op = BadLeavesOrdering(2.0, qml.RX(1.2, wires=0))

        with pytest.raises(AssertionError, match=r"data must be the terminal leaves of the pytree"):
            assert_valid(op)


def test_data_is_tuple():
    """Check that the data property is a tuple."""

    class BadData(Operator):
        num_params = 1

        def __init__(self, x, wires):
            super().__init__(x, wires)
            self.data = [x]

    with pytest.raises(AssertionError, match=r"op.data must be a tuple"):
        assert_valid(BadData(2.0, wires=0))
