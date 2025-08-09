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
import string

import numpy as np

# pylint: disable=too-few-public-methods, unused-argument
import pytest

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.functions import assert_valid
from pennylane.ops.functions.assert_valid import _check_capture


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

    def test_decomposition_wires_must_be_mapped(self):
        """Test that the operators in decomposition have mapped wires after mapping the op."""

        class BadDecompositionWireMap(Operator):

            @staticmethod
            def compute_decomposition(*args, **kwargs):
                if kwargs["wires"][0] == 0:
                    return [qml.RX(0.2, wires=0)]
                return [qml.RX(0.2, wires="not the ops wire")]

        with pytest.raises(AssertionError, match=r"Operators in decomposition of wire\-mapped"):
            assert_valid(BadDecompositionWireMap(wires=0), skip_pickle=True)
        assert_valid(BadDecompositionWireMap(wires=0), skip_pickle=True, skip_wire_mapping=True)

    def test_error_not_raised(self):
        """Test if has_decomposition is False but decomposition defined."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qml.RY(2.3, 0)]

            has_decomposition = False

        with pytest.raises(AssertionError, match="If has_decomposition is False"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_decomposition_must_not_contain_op(self):
        """Test that the decomposition of an operator doesn't include the operator itself"""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [BadDecomp(wires)]

        with pytest.raises(AssertionError, match="should not be included in its own decomposition"):
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
            return [qml.PauliX(self.wires)]

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

    with pytest.raises(AttributeError):
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
    """Pytree related checks."""

    def test_not_hashable_metadata(self):
        """Assert that an error is raised if metadata is not hashable."""

        class BadMetadata(Operator):
            def __init__(self, wires):
                super().__init__(wires)
                self.hyperparameters["test"] = ["a"]

        op = BadMetadata(0)
        with pytest.raises(AssertionError, match=r"metadata output from _flatten must be hashable"):
            assert_valid(op, skip_pickle=True)

    def test_bad_pytree(self):
        """Check an operation that errors out of the _unflatten call."""

        class BadPytree(qml.operation.Operator):
            def __init__(self, wires1, wires2):
                super().__init__(wires=wires1 + wires2)

        op = BadPytree([0], [1])

        with pytest.raises(AssertionError, match=r"BadPytree._unflatten must be able to reproduce"):
            assert_valid(op, skip_pickle=True)

    def test_badpytree_incomplete_info(self):
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
            assert_valid(op, skip_pickle=True)

    @pytest.mark.jax
    def test_nested_bad_pytree(self):
        """Test that an operator with a bad leaf will raise an error."""

        class BadPytree(Operator):
            def _flatten(self):
                return tuple(), (self.wires, tuple())

            def __init__(self, wires, val="a"):
                super().__init__(wires)
                self.hyperparameters["val"] = val

        op = qml.adjoint(BadPytree(wires=0, val="b"))
        with pytest.raises(AssertionError, match=r"op must be a valid pytree."):
            assert_valid(op, skip_pickle=True)

    @pytest.mark.jax
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
            assert_valid(op, skip_pickle=True)


@pytest.mark.jax
def test_bad_capture():
    """Tests that the correct error is raised when something goes wrong with program capture."""

    class MyBadOp(qml.operation.Operator):

        def _flatten(self):
            return (self.hyperparameters["target_op"], self.data[0]), ()

        @classmethod
        def _unflatten(cls, data, metadata):
            return cls(*data)

        def __init__(self, target_op, val):
            super().__init__(val, wires=target_op.wires)
            self.hyperparameters["target_op"] = target_op

    op = MyBadOp(qml.X(0), 2)
    with pytest.raises(ValueError, match=r"The capture of the operation into jaxpr failed"):
        _check_capture(op)


def test_data_is_tuple():
    """Check that the data property is a tuple."""

    class BadData(Operator):
        num_params = 1

        def __init__(self, x, wires):
            super().__init__(x, wires)
            self.data = [x]

    with pytest.raises(AssertionError, match=r"op.data must be a tuple"):
        assert_valid(BadData(2.0, wires=0))


def create_op_instance(c, str_wires=False):
    """Given an Operator class, create an instance of it."""
    n_wires = c.num_wires
    if n_wires is None:
        n_wires = 1

    wires = qml.wires.Wires(range(n_wires))
    if str_wires and len(wires) < 26:
        wires = qml.wires.Wires([string.ascii_lowercase[i] for i in wires])
    if (num_params := c.num_params) == 0:
        return c(wires) if wires else c()
    if isinstance(num_params, property):
        num_params = 1

    # get ndim_params
    if isinstance((ndim_params := c.ndim_params), property):
        ndim_params = (0,) * num_params

    # turn ndim_params into valid params
    [dim] = set(ndim_params)
    if dim == 0:
        params = [1] * len(ndim_params)
    elif dim == 1:
        params = [[1] * 2**n_wires] * len(ndim_params)
    elif dim == 2:
        params = [np.eye(2)] * len(ndim_params)
    else:
        raise ValueError("unexpected dim:", dim)

    return c(*params, wires=wires) if wires else c(*params)


@pytest.mark.jax
@pytest.mark.parametrize("str_wires", (True, False))
def test_generated_list_of_ops(class_to_validate, str_wires):
    """Test every auto-generated operator instance."""
    if class_to_validate.__module__[14:20] == "qutrit":
        pytest.xfail(reason="qutrit ops fail matrix validation")

    if class_to_validate.__module__[10:14] == "ftqc":
        pytest.skip(reason="skip tests for ftqc ops")

    # If you defined a new Operator and this call to `create_op_instance` failed, it might
    # be the fault of the test and not your Operator. Please do one of the following things:
    #   1. Update your Operator to meet PL standards so it passes
    #   2. Improve `create_op_instance` so it can create an instance of your op (it is quite hacky)
    #   3. Add an instance of your class to `_INSTANCES_TO_TEST` in ./conftest.py
    #       Note: if it then fails validation, move it to `_INSTANCES_TO_FAIL` as described below.
    op = create_op_instance(class_to_validate, str_wires)

    # If you defined a new Operator and this call to `assert_valid` failed, the Operator doesn't
    # follow PL standards. Please do one of the following things:
    #   1. Preferred action: Update your Operator to meet PL standards so it passes
    #   2. Add an instance of your class to `_INSTANCES_TO_FAIL` in ./conftest.py, along with the
    #       exception type raised by the assertion and a comment explaining the assumption your
    #       operator makes.
    assert_valid(op)


@pytest.mark.jax
def test_explicit_list_of_ops(valid_instance_and_kwargs):
    """Test the validity of operators that could not be auto-generated."""
    valid_instance, kwargs = valid_instance_and_kwargs
    assert_valid(valid_instance, **kwargs)


@pytest.mark.jax
def test_explicit_list_of_failing_ops(invalid_instance_and_error):
    """Test instances of ops that fail validation."""
    op, exc_type = invalid_instance_and_error
    with pytest.raises(exc_type):
        assert_valid(op)
