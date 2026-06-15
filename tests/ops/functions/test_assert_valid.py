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
This module contains unit tests for ``qp.ops.functions.assert_valid``.
"""

import string
from pickle import PicklingError

import numpy as np

# pylint: disable=too-few-public-methods, unused-argument
import pytest
import scipy.sparse

import pennylane as qp
from pennylane.core import Operator2
from pennylane.core.operator import Operator
from pennylane.ops.functions import assert_valid
from pennylane.ops.functions.assert_valid import _check_capture, _test_decomposition_rule
from pennylane.wires import Wires


class TestDecompositionErrors:
    """Test assertions involving decompositions."""

    def test_bad_decomposition_output(self):
        """Test that an error is raised if decomposition output is not a list."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                qp.RX(1.2, wires=0)

        with pytest.raises(AssertionError, match=r"decomposition must be a list"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_bad_decomposition_lengths(self):
        """Test that an error is raised if decomposition, compute_decomposition and queuing
        do not have the same number of ops."""

        class BadDecompLength(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qp.X(wires[0]), qp.Y(wires[1])]

            def decomposition(self):
                return [qp.X(self.wires[0])]

        with pytest.raises(AssertionError, match="decomposition must match compute_decomposition"):
            assert_valid(BadDecompLength(wires=[0, 1]), skip_pickle=True)

        class BadDecompQueueLength(Operator):
            @staticmethod
            def compute_decomposition(wires):
                qp.X(wires[0])
                return [qp.Y(wires[1])]

        with pytest.raises(AssertionError, match="decomposition must match queued operations"):
            assert_valid(BadDecompQueueLength(wires=[0, 1]), skip_pickle=True)

    def test_bad_decomposition_queuing(self):
        """Test that an error is raised if decomposition and queuing do not have the same op."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                qp.RX(1.2, wires=0)
                with qp.QueuingManager.stop_recording():
                    other_op = qp.RY(2.3, 0)
                return [other_op]

        with pytest.raises(AssertionError, match="decomposition must match queued operations"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_bad_decomposition_with_mcm(self):
        """Test that an error is raised if decomposition and compute_decomposition involve
        mid-circuit measurements and return different ops."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                mcm0 = qp.measure(wires[0])
                mcm1 = qp.measure(wires[1])
                return mcm0.measurements + mcm1.measurements

            def decomposition(self):
                mcm0 = qp.measure(self.wires[0])
                mcm1 = qp.measure(self.wires[1])
                return mcm1.measurements + mcm0.measurements

        with pytest.raises(AssertionError, match="decomposition must match compute_decomposition"):
            assert_valid(BadDecomp(wires=[0, 1]), skip_pickle=True)

        class BadDecompQueue(Operator):
            @staticmethod
            def compute_decomposition(wires):
                """Return other ops than are queued, but the same number of ops."""
                meas0 = qp.ops.MidMeasure(wires[0], meas_uid=251)
                qp.ops.MidMeasure(wires[1], meas_uid=252)
                with qp.QueuingManager.stop_recording():
                    meas2 = qp.ops.MidMeasure(wires[0], meas_uid=253)
                return [meas0, meas2]

        with pytest.raises(AssertionError, match="decomposition must match queued operations"):
            assert_valid(BadDecompQueue(wires=[0, 1]), skip_pickle=True)

    @pytest.mark.jax
    def test_decomposition_wires_must_be_mapped(self):
        """Test that an error is raised if the operators in decomposition do not have mapped
        wires after mapping the op."""

        class BadDecompositionWireMap(Operator):

            @staticmethod
            def compute_decomposition(*args, **kwargs):
                if kwargs["wires"][0] == 0:
                    return [qp.RX(0.2, wires=0)]
                return [qp.RX(0.2, wires="not the ops wire")]

        with pytest.raises(AssertionError, match=r"Operators in decomposition of wire\-mapped"):
            assert_valid(BadDecompositionWireMap(wires=0), skip_pickle=True)
        assert_valid(BadDecompositionWireMap(wires=0), skip_pickle=True, skip_wire_mapping=True)

    def test_error_has_decomposition_but_claims_not_to(self):
        """Test that an error is raised if has_decomposition is False but a decomposition is
        defined."""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [qp.RY(2.3, 0)]

            has_decomposition = False

        with pytest.raises(AssertionError, match="If has_decomposition is False"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    def test_decomposition_must_not_contain_op(self):
        """Test that an error is raised if the decomposition of an operator includes
        the operator itself"""

        class BadDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                return [BadDecomp(wires)]

        with pytest.raises(AssertionError, match="should not be included in its own decomposition"):
            assert_valid(BadDecomp(wires=0), skip_pickle=True)

    @pytest.mark.jax
    def test_mcms_can_be_compared(self):
        """Tests that decompositions with mid-circuit measurements can be compared correctly."""

        class ValidMCMDecomp(Operator):
            @staticmethod
            def compute_decomposition(wires):
                mcm = qp.measure(wires[0])
                return mcm.measurements

        assert_valid(ValidMCMDecomp(wires=0), skip_pickle=True)

    def test_bad_new_decomposition_rule_exact(self):
        """Test that an informative error is raised if the
        claimed-to-be-exact resources of a decomposition rule are not correct."""

        class MyOp(Operator):
            num_wires = 2

        op = MyOp([0, 1])

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_wrong_numbers = qp.register_resources({qp.X: 2, qp.Y: 3})(rule)
        with pytest.raises(AssertionError, match="The numbers are off"):
            _test_decomposition_rule(op, rule_wrong_numbers)

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2})(rule)
        with pytest.raises(AssertionError, match="Missing entirely in gate counts"):
            _test_decomposition_rule(op, rule_wrong_ops)

    def test_bad_new_decomposition_rule_inexact(self):
        """Test that an informative error is raised if the
        inexact resources of a decomposition rule are not correct."""

        class MyOp(Operator):
            num_wires = 2

        def rule(wires):
            qp.X(wires[0])
            qp.X(wires[1])
            qp.Y(wires[0])
            qp.Y(wires[1])

        rule_wrong_ops = qp.register_resources({qp.X: 2, qp.Z: 2}, exact=False)(rule)
        op = MyOp([0, 1])
        with pytest.raises(AssertionError, match="Gate counts expected from"):
            _test_decomposition_rule(op, rule_wrong_ops)


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

    def test_bad_copy(self):
        """Test an operator that cannot be compared with its copy."""

        class BadComparison(Operator):
            def __copy__(self):
                return self

        with pytest.raises(AssertionError, match=r"copied op must be a separate instance"):
            assert_valid(BadComparison(0), skip_pickle=True)

    def test_bad_deepcopy(self):
        """Test an operator that cannot be compared with its deepcopy."""

        class BadDeepComparison(Operator):
            def __deepcopy__(self, memo):
                return BadDeepComparison(1)

        with pytest.raises(AssertionError, match=r"deep copied op must also be equal"):
            assert_valid(BadDeepComparison(0), skip_pickle=True)


def test_mismatched_mat_decomp():
    """Test that an error is raised if the matrix does not match the decomposition if both are defined."""

    class MisMatchedMatDecomp(Operator):
        @staticmethod
        def compute_matrix():
            return np.eye(2)

        def decomposition(self):
            return [qp.PauliX(self.wires)]

    with pytest.raises(AssertionError, match=r"matrix and matrix from decomposition must match"):
        assert_valid(MisMatchedMatDecomp(0), skip_pickle=True)


def test_bad_eigenvalues_order():
    """Test that an error is raised if the order of eigenvalues does not match the diagonalizing gates."""

    class BadEigenDecomp(qp.PauliX):
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

    with pytest.raises((AttributeError, PicklingError)):
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
        def __init__(self, wires):
            super().__init__(wires=wires)

        @property
        def wires(self):
            return Wires(0)

    with pytest.raises(AssertionError, match=r"wires must be mappable"):
        assert_valid(BadWireMap(1), skip_pickle=True)


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

        class BadPytree(qp.operation.Operator):
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

        op = qp.adjoint(BadPytree(wires=0, val="b"))
        with pytest.raises(AssertionError, match=r"op must be a valid pytree."):
            assert_valid(op, skip_pickle=True)

    @pytest.mark.jax
    def test_bad_leaves_ordering(self):
        """Test an error is raised if data and pytree leaves have a different ordering convention."""

        class BadLeavesOrdering(qp.ops.op_math.SProd):
            def _flatten(self):
                return (self.base, self.scalar), tuple()

            @classmethod
            def _unflatten(cls, data, _):
                return cls(data[1], data[0])

        op = BadLeavesOrdering(2.0, qp.RX(1.2, wires=0))

        with pytest.raises(AssertionError, match=r"data must be the terminal leaves of the pytree"):
            assert_valid(op, skip_pickle=True)


@pytest.mark.jax
def test_bad_capture():
    """Tests that the correct error is raised when something goes wrong with program capture."""

    class MyBadOp(qp.operation.Operator):

        def _flatten(self):
            return (self.hyperparameters["target_op"], self.data[0]), ()

        @classmethod
        def _unflatten(cls, data, metadata):
            return cls(*data)

        def __init__(self, target_op, val):
            super().__init__(val, wires=target_op.wires)
            self.hyperparameters["target_op"] = target_op

    op = MyBadOp(qp.X(0), 2)
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


class SingleRZ(Operator2):
    """A fully-featured ``Operator2`` defining a matrix, a decomposition,
    eigenvalues, diagonalizing gates, and a generator.
    """

    dynamic_argnames = ("phi",)
    wire_argnames = ("wires",)
    ndim_params = ((),)

    def __init__(self, phi, wires):
        assert isinstance(wires, int) or len(wires) == 1
        super().__init__(phi, wires=wires)

    @staticmethod
    def compute_matrix(phi, wires):
        return qp.math.array(
            [
                [qp.math.exp(-0.5j * phi), 0],
                [0, qp.math.exp(0.5j * phi)],
            ]
        )

    @staticmethod
    def compute_decomposition(phi, wires):
        return [qp.RZ(phi, wires=wires[0])]

    @staticmethod
    def compute_eigvals(phi, wires=None):
        return qp.math.array([qp.math.exp(-0.5j * phi), qp.math.exp(0.5j * phi)])

    def generator(self):
        return qp.Hamiltonian([-0.5], [qp.PauliZ(wires=self.wires)])


class TestOperator2AssertValid:
    """Tests showing that ``assert_valid`` works on :class:`~.core.Operator2` instances thanks to
    the backwards-compatible ``data``/``parameters``/``num_params``/``hyperparameters`` attributes.

    The first test validates a fully-featured ``Operator2``. The remaining tests each violate the
    criteria of a single internal ``assert_valid`` check and confirm the corresponding failure.
    """

    def test_full_featured_operator_is_valid(self):
        """A fully-featured, self-consistent ``Operator2`` passes ``assert_valid``."""
        op = SingleRZ(np.array(0.5), wires=0)

        # differentiation is skipped: the Operator2 pytree leaves include its wires
        assert_valid(op, skip_differentiation=True)

    def test_invalid_dyn_arg_dimension(self):
        """``_assert_valid_operator2`` fails if a dynamic argument has the wrong shape."""

        class BadDims(Operator2):
            dynamic_argnames = ("angles",)
            wire_argnames = ("wires",)
            ndim_params = (2,)

            def __init__(self, angles, wires):
                super().__init__(angles, wires=wires)

        with pytest.raises(AssertionError, match=r"is not equal to dimension in ndim_params"):
            assert_valid(BadDims(np.array([0.1, 0.2, 0.3]), wires=[0, 1]), skip_pickle=True)

    def test_check_decomposition(self):
        """``_check_decomposition`` fails if ``compute_decomposition`` does not return a list."""

        class BadDecomp(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_decomposition(phi, wires):
                qp.RX(phi, wires=wires[0])  # queues but returns ``None``

        with pytest.raises(AssertionError, match=r"decomposition must be a list"):
            assert_valid(BadDecomp(1.2, wires=0), skip_pickle=True)

    def test_check_matrix(self):
        """``_check_matrix`` fails if the matrix does not have the expected shape."""

        class BadMatrix(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return np.eye(2)  # should be (4, 4) for two wires

        with pytest.raises(
            AssertionError, match=r"matrix must be two dimensional with shape \(4, 4\)"
        ):
            assert_valid(BadMatrix(1.0, wires=[0, 1]), skip_pickle=True)

    def test_check_matrix_matches_decomposition(self):
        """``_check_matrix_matches_decomp`` fails if the matrix and decomposition disagree."""

        class MatDecompMismatch(Operator2):
            wire_argnames = ("wires",)
            static_argnames = ("phi",)

            def __init__(self, wires, phi):
                super().__init__(wires=wires, phi=phi)

            @staticmethod
            def compute_matrix(wires, phi):
                return np.eye(2)

            @staticmethod
            def compute_decomposition(wires, phi):
                return [qp.RX(phi, wires[0])]

        with pytest.raises(
            AssertionError, match=r"matrix and matrix from decomposition must match"
        ):
            assert_valid(MatDecompMismatch(wires=0, phi=1.0), skip_pickle=True)

    def test_check_sparse_matrix(self):
        """``_check_sparse_matrix`` fails if the sparse matrix does not have the expected shape."""

        class BadSparse(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_sparse_matrix(phi, wires, format="csr"):
                return scipy.sparse.eye(2, format=format) * phi  # should be (4, 4) for two wires

        with pytest.raises(
            AssertionError, match=r"matrix must be two dimensional with shape \(4, 4\)"
        ):
            assert_valid(BadSparse(0.5, wires=[0, 1]), skip_pickle=True)

    def test_check_eigendecomposition(self):
        """``_check_eigendecomposition`` fails if the eigenvalues and diagonalizing gates cannot
        reproduce the operator."""

        class BadEigen(Operator2):
            wire_argnames = ("wires",)
            static_argnames = ("phi",)
            hybrid_argnames = ("tree",)

            def __init__(self, phi, wires, tree):
                super().__init__(phi, wires=wires, tree=tree)

            @staticmethod
            def compute_matrix(phi, wires, tree):
                return qp.RX.compute_matrix(phi)

            @staticmethod
            def compute_eigvals(phi, wires=None, tree=None):
                return np.array([1, 1])  # PauliX = RX(pi) has eigenvalues [1, -1]

            @staticmethod
            def compute_diagonalizing_gates(phi, wires, tree):
                return tree

        with pytest.raises(
            AssertionError, match=r"eigenvalues and diagonalizing gates must be able to reproduce"
        ):
            assert_valid(BadEigen(np.pi, wires=0, tree=[qp.Hadamard(0)]), skip_pickle=True)

    def test_check_generator(self):
        """``_check_generator`` fails if the generator does not reproduce the operator."""

        class BadGen(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

            @staticmethod
            def compute_matrix(phi, wires):
                return qp.RZ.compute_matrix(phi)

            def generator(self):
                return qp.X(self.wires[0])  # the generator of RZ is ``-0.5 * Z``

        with pytest.raises(AssertionError):
            assert_valid(BadGen(np.pi, wires=0), skip_pickle=True, skip_differentiation=True)

    def test_check_pickle(self):
        """``_check_pickle`` fails if the operator cannot be pickled (e.g. a local class)."""

        class LocalOp(Operator2):
            dynamic_argnames = ("phi",)
            wire_argnames = ("wires",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        with pytest.raises((AttributeError, PicklingError)):
            assert_valid(LocalOp(np.pi, wires=0))


def create_op_instance(c, str_wires=False):
    """Given an Operator class, create an instance of it."""
    n_wires = c.num_wires
    if n_wires is None:
        n_wires = 1

    wires = qp.wires.Wires(range(n_wires))
    if str_wires and len(wires) < 26:
        wires = qp.wires.Wires([string.ascii_lowercase[i] for i in wires])
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
