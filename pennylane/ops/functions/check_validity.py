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

def _check_decomposition(op):
    if op.has_decomposition:
        decomp = op.decomposition()
        try:
            compute_decomp = op.compute_decomposition(*op.data, wires=op.wires, **op.hyperparameters)
        except qml.operation.DecompositionUndefinedError:
            # sometimes decomposition is defined but not compute_decomposition
            compute_decomp = decomp
        with qml.queuing.AnnotatedQueue() as queued_decomp:
            op.decomposition()
        processed_queue = qml.tape.QuantumTape.from_queue(queued_decomp)
        expand = op.expand()

        assert isinstance(decomp, list), "decomposition must be a list"
        assert isinstance(compute_decomp, list), "decomposition must be a list"
        assert isinstance(expand, qml.tape.QuantumScript), "expansion must be a QuantumScript"

        for o1, o2, o3, o4 in zip(decomp, compute_decomp, processed_queue , expand):
            assert o1 == o2, "decomposition must match compute_decomposition"
            assert o1 == o3, "decomposition must matched queued operations"
            assert o1 == o4, "decomposition must match expansion"
            assert isinstance(o1, qml.operation.Operator), "decomposition must contain operators"
    else:
        error_raised=False
        try:
            op.decomposition()
        except qml.operation.DecompositionUndefinedError as e:
            error_raised=True
        assert error_raised, "error must be raised if decomposition isnt defined"

        error_raised=False
        try:
            op.expand()
        except qml.operation.DecompositionUndefinedError as e:
            error_raised=True
        assert error_raised, "error must be raised if decomposition isnt defined"

        error_raised=False
        try:
            op.compute_decomposition(*op.data, wires=op.wires, **op.hyperparameters)
        except qml.operation.DecompositionUndefinedError as e:
            error_raised=True
        assert error_raised, "error must be raised if decomposition isnt defined"

def _check_matrix(op):
    if op.has_matrix:
        mat = op.matrix()
        mat2 = qml.matrix(op)
        assert qml.math.allclose(mat, mat2), "op.matrix must match qml.matrix"
    else:
        error_raised = False
        try:
            op.matrix()
        except qml.operation.MatrixUndefinedError:
            error_raised = True
        assert error_raised, "error must be raised if matrix isnt defined"

def _check_matrix_matches_decomp(op):
    if op.has_matrix and op.has_decomposition:
        mat = op.matrix()
        decomp_mat = qml.matrix(op.decomposition, wire_order=op.wires)()
        assert qml.math.allclose(mat, decomp_mat), "matrix and matrix from decomposition must match"

def _check_eigendecomposition(op):

    if op.has_diagonalizing_gates:
        dg = op.diagonalizing_gates()
        try:
            compute_dg = op.compute_diagonalizing_gates(*op.data, op.wires, **op.hyperparameters)
        except qml.operation.DiagGatesUndefinedError as e:
            compute_dg = dg

        for op1, op2 in zip(dg, compute_dg):
            assert op1 == op2, "diagonalizing_gates and compute_diagonalizing_gates must match"
    else:
        error_raised = False
        try:
            op.diagonalizing_gates()
        except qml.operation.DiagGatesUndefinedError:
            error_raised = True
        assert error_raised, "error must be raised if diagonalizing_gates isnt defined."

    has_eigvals = True
    try:
        eg = op.eigvals()
        compute_eg = op.compute_eigvals(*op.data, **op.hyperparameters)
        assert qml.math.allclose(eg, compute_eg), "eigvals and compute_eigvals must match"
    except EigvalsUndefinedError:
        has_eigvals = False

    if has_eigvals and op.has_diagonalizing_gates:
        dg = qml.prod(*dg) if len(dg) > 0 else qml.Identity(op.wires)
        eg = qml.DiagonalQubitUnitary(eg, wires=op.wires)
        decomp = dg @ qml.DiagonalQubitUnitary(eg, wires=op.wires) @ qml.adjoint(dg)
        decomp_mat = qml.matrix(decomp)
        assert qml.math.allclose(decomp_mat, qml.matrix(op)), "eigenvalues and diagonalizing gates must be able to reproduce the original operator"


# pylint: disable=import-outside-toplevel
def _check_jax(op):
    try:
        import jax
    except ImportError:
        return
    leaves, struct = jax.tree_util.tree_flatten(op)
    unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)
    assert unflattened_op == op, "op must be a valid pytree"

def _check_copy(op):

    copied_op = copy.copy(op)
    assert copied_op == op, "copied op must be equivalent to original operation"
    assert copied_op is not op, "copied op must be a separate instance from original operaiton"
    assert qml.equal(copied_op, op), "copied op must also be equal with qml.equal"
    assert copy.deepcopy(op) == op, "deep copied op must also be equal"

def _check_pytree(op):
    data, metadata = op._flatten()
    try:
        assert hash(metadata), "metadata must be hashable"
    except Exception as e:
        raise ValueError("metadata output from _flatten must be hashable. This also applies to hyperparameters") from e
    new_op = type(op)._unflatten(data, metadata)
    assert op == new_op, "metadata and data must be able to replicate the original operation"

def _check_pickle(op):
    pickled = pickle.dumps(op)
    unpickled = pickle.loads(pickled)
    assert unpickled == op, "operation must be able to be pickled and unpickled"

def _check_bind_new_parameters(op):
    new_data = [d+1.0 for d in op.data]
    new_data_op = qml.ops.functions.bind_new_parameters(op, new_data)
    for d1, d2 in zip(new_data_op.data, new_data):
        assert qml.math.allclose(d1, d2), "new data must match the data set with."

def _check_wires(op):
    assert isinstance(op.wires, qml.wires.Wires), "wires must be a wires instance"
    if op.num_wires != -1:
        assert len(op.wires) == op.num_wires, "num_wires must match the number of wires"

    wire_map = {w: ascii_lowercase[i] for i, w in enumerate(op.wires)}
    mapped_op = op.map_wires(wire_map)
    assert mapped_op.wires == qml.wires.Wires(list(ascii_lowercase[:len(op.wires)])), "wires must be mappable"

def check_validity(op: qml.operation.Operator) -> None:
    """Runs basic validation checks on an :class:`~.operation.Opeartor` to make
    sure it has been correctly defined.

    Args:
        op (qml.opeartion.Operator): a instance to validate

    **Examples:**

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, data, wires):
                self.data = data
                super().__init__(wires=wires)

        op = MyOp(qml.numpy.array(0.5), wires=0)
        check_validity(op)

    .. code-block::

        AssertionError: op.data must be a tuple

    .. code-block:: python
        class MyOp(qml.operation.Operator):

            def __init__(self, wires):
                self.hyperparameters["unhashable_list"] = []
                super().__init__(wires=wires)

        op = MyOp(wires = 0)
        check_validity(op)

    .. code-block::

        ValueError: metadata output from _flatten must be hashable. This also applies to hyperparameters

    """


    assert isinstance(op.data, tuple), "op.data must be a tuple"
    assert isinstance(op.parameters, list), "op.parameters must be a list"
    assert len(op.data) == op.num_params, "length of data must match num_params"
    for d, p in zip(op.data, op.parameters):
        assert isinstance(d, qml.typing.TensorLike), "each data element must be tensorlike"
        assert qml.math.allclose(d, p), "data and parameters must match."

    _check_wires(op)
    _check_copy(op)
    _check_pytree(op)
    _check_pickle(op)
    _check_bind_new_parameters(op)

    _check_jax(op)

    _check_decomposition(op)
    _check_matrix(op)
    _check_matrix_matches_decomp(op)
    _check_eigendecomposition(op)