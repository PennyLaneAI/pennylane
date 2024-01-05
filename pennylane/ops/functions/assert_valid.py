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
This module contains the qml.ops.functions.check_validity function for determining whether or not an
Operator class is correctly defined.
"""

from string import ascii_lowercase
import copy
import pickle

import numpy as np

import pennylane as qml
from pennylane.operation import EigvalsUndefinedError


def _assert_error_raised(func, error, failure_comment):
    def inner_func(*args, **kwargs):
        error_raised = False
        try:
            func(*args, **kwargs)
        except error:
            error_raised = True
        assert error_raised, failure_comment

    return inner_func


def _check_decomposition(op):
    """Checks involving the decomposition."""
    if op.has_decomposition:
        decomp = op.decomposition()
        try:
            compute_decomp = type(op).compute_decomposition(
                *op.data, wires=op.wires, **op.hyperparameters
            )
        except (qml.operation.DecompositionUndefinedError, TypeError):
            # sometimes decomposition is defined but not compute_decomposition
            # Also  sometimes compute_decomposition can have a different signature
            compute_decomp = decomp
        with qml.queuing.AnnotatedQueue() as queued_decomp:
            op.decomposition()
        processed_queue = qml.tape.QuantumTape.from_queue(queued_decomp)
        expand = op.expand()

        assert isinstance(decomp, list), "decomposition must be a list"
        assert isinstance(compute_decomp, list), "decomposition must be a list"
        assert isinstance(expand, qml.tape.QuantumScript), "expand must return a QuantumScript"

        for o1, o2, o3, o4 in zip(decomp, compute_decomp, processed_queue, expand):
            assert o1 == o2, "decomposition must match compute_decomposition"
            assert o1 == o3, "decomposition must match queued operations"
            assert o1 == o4, "decomposition must match expansion"
            assert isinstance(o1, qml.operation.Operator), "decomposition must contain operators"
    else:
        failure_comment = "If has_decomposition is False, then decomposition must raise a ``DecompositionUndefinedError``."
        _assert_error_raised(
            op.decomposition,
            qml.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )()
        _assert_error_raised(
            op.expand, qml.operation.DecompositionUndefinedError, failure_comment=failure_comment
        )()
        _assert_error_raised(
            op.compute_decomposition,
            qml.operation.DecompositionUndefinedError,
            failure_comment=failure_comment,
        )(*op.data, wires=op.wires, **op.hyperparameters)


def _check_matrix(op):
    """Check that if the operation says it has a matrix, it does. Otherwise a ``MatrixUndefinedError`` should be raised."""
    if op.has_matrix:
        mat = op.matrix()
        assert isinstance(mat, qml.typing.TensorLike), "matrix must be a TensorLike"
        l = 2 ** len(op.wires)
        failure_comment = f"matrix must be two dimensional with shape ({l}, {l})"
        assert qml.math.shape(mat) == (l, l), failure_comment
    else:
        failure_comment = (
            "If has_matrix is False, the matrix method must raise a ``MatrixUndefinedError``."
        )
        _assert_error_raised(
            op.matrix, qml.operation.MatrixUndefinedError, failure_comment=failure_comment
        )()


def _check_matrix_matches_decomp(op):
    """Check that if both the matrix and decomposition are defined, they match."""
    if op.has_matrix and op.has_decomposition:
        mat = op.matrix()
        decomp_mat = qml.matrix(qml.tape.QuantumScript(op.decomposition()), wire_order=op.wires)
        failure_comment = (
            f"matrix and matrix from decomposition must match. Got \n{mat}\n\n {decomp_mat}"
        )
        assert qml.math.allclose(mat, decomp_mat), failure_comment


def _check_eigendecomposition(op):
    """Checks involving diagonalizing gates and eigenvalues."""
    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            compute_dg = type(op).compute_diagonalizing_gates(
                *op.data, wires=op.wires, **op.hyperparameters
            )
        except (qml.operation.DiagGatesUndefinedError, TypeError):
            # sometimes diagonalizing gates is defined but not compute_diagonalizing_gates
            # compute_diagonalizing_gates might also have a different call signature
            compute_dg = dg

        for op1, op2 in zip(dg, compute_dg):
            assert op1 == op2, "diagonalizing_gates and compute_diagonalizing_gates must match"
    else:
        failure_comment = "If has_diagonalizing_gates is False, diagonalizing_gates must raise a DiagGatesUndefinedError"
        _assert_error_raised(
            op.diagonalizing_gates, qml.operation.DiagGatesUndefinedError, failure_comment
        )()

    try:
        eg = op.eigvals()
    except EigvalsUndefinedError:
        eg = None

    has_eigvals = True
    try:
        compute_eg = type(op).compute_eigvals(*op.data, **op.hyperparameters)
    except EigvalsUndefinedError:
        compute_eg = eg
        has_eigvals = False

    if has_eigvals:
        assert qml.math.allclose(eg, compute_eg), "eigvals and compute_eigvals must match"

    if has_eigvals and op.has_diagonalizing_gates:
        dg = qml.prod(*dg[::-1]) if len(dg) > 0 else qml.Identity(op.wires)
        eg = qml.QubitUnitary(np.diag(eg), wires=op.wires)
        decomp = qml.prod(qml.adjoint(dg), eg, dg)
        decomp_mat = qml.matrix(decomp)
        original_mat = qml.matrix(op)
        failure_comment = f"eigenvalues and diagonalizing gates must be able to reproduce the original operator. Got \n{decomp_mat}\n\n{original_mat}"
        assert qml.math.allclose(decomp_mat, original_mat), failure_comment


def _check_copy(op):
    """Check that copies and deep copies give identical objects."""
    copied_op = copy.copy(op)
    assert qml.equal(copied_op, op), "copied op must be equal with qml.equal"
    assert copied_op == op, "copied op must be equivalent to original operation"
    assert copied_op is not op, "copied op must be a separate instance from original operaiton"
    assert qml.equal(copy.deepcopy(op), op), "deep copied op must also be equal"


# pylint: disable=import-outside-toplevel, protected-access
def _check_pytree(op):
    """Check that the operator is a pytree."""
    data, metadata = op._flatten()
    try:
        assert hash(metadata), "metadata must be hashable"
    except Exception as e:
        raise AssertionError(
            f"metadata output from _flatten must be hashable. Got metadata {metadata}"
        ) from e
    try:
        new_op = type(op)._unflatten(data, metadata)
    except Exception as e:
        message = (
            f"{type(op).__name__}._unflatten must be able to reproduce the original operation from "
            f"{data} and {metadata}. You may need to override either the _unflatten or _flatten method. "
            f"\nFor local testing, try type(op)._unflatten(*op._flatten())"
        )
        raise AssertionError(message) from e
    assert op == new_op, "metadata and data must be able to reproduce the original operation"

    try:
        import jax
    except ImportError:
        return
    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened_op == op, f"op must be a valid pytree. Got {unflattened_op} instead of {op}."

    for d1, d2 in zip(op.data, leaves):
        assert qml.math.allclose(
            d1, d2
        ), f"data must be the terminal leaves of the pytree. Got {d1}, {d2}"


def _check_pickle(op):
    """Check that an operation can be dumped and reloaded with pickle."""
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, "operation must be able to be pickled and unpickled"


# pylint: disable=no-member
def _check_bind_new_parameters(op):
    """Check that bind new parameters can create a new op with different data."""
    new_data = [d * 0.0 for d in op.data]
    new_data_op = qml.ops.functions.bind_new_parameters(op, new_data)
    failure_comment = "bind_new_parameters must be able to update the operator with new data."
    for d1, d2 in zip(new_data_op.data, new_data):
        assert qml.math.allclose(d1, d2), failure_comment


def _check_wires(op):
    """Check that wires are a ``Wires`` class and can be mapped."""
    assert isinstance(op.wires, qml.wires.Wires), "wires must be a wires instance"

    wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    new_wires = qml.wires.Wires(list(ascii_lowercase[: len(op.wires)]))
    assert mapped_op.wires == new_wires, "wires must be mappable with map_wires"


def assert_valid(op: qml.operation.Operator, skip_pickle=False) -> None:
    """Runs basic validation checks on an :class:`~.operation.Operator` to make
    sure it has been correctly defined.

    Args:
        op (.Operator): an operator instance to validate

    Keyword Args:
        skip_pickle=False : If ``True``, pickling tests are not run. Set to ``True`` when
            testing a locally defined operator, as pickle cannot handle local objects

    **Examples:**

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, data, wires):
                self.data = data
                super().__init__(wires=wires)

        op = MyOp(qml.numpy.array(0.5), wires=0)

    .. code-block::

        >>> assert_valid(op)
        AssertionError: op.data must be a tuple

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, wires):
                self.hyperparameters["unhashable_list"] = []
                super().__init__(wires=wires)

        op = MyOp(wires = 0)
        assert_valid(op)

    .. code-block::

        ValueError: metadata output from _flatten must be hashable. This also applies to hyperparameters

    """

    assert isinstance(op.data, tuple), "op.data must be a tuple"
    assert isinstance(op.parameters, list), "op.parameters must be a list"
    for d, p in zip(op.data, op.parameters):
        assert isinstance(d, qml.typing.TensorLike), "each data element must be tensorlike"
        assert qml.math.allclose(d, p), "data and parameters must match."

    if len(op.wires) <= 26:
        _check_wires(op)
    _check_copy(op)
    _check_pytree(op)
    if not skip_pickle:
        _check_pickle(op)
    _check_bind_new_parameters(op)

    _check_decomposition(op)
    _check_matrix(op)
    _check_matrix_matches_decomp(op)
    _check_eigendecomposition(op)
