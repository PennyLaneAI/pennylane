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
This module contains the qml.matrix function.
"""
from collections.abc import Callable, Sequence
from functools import partial

from pennylane import math
from pennylane.exceptions import MatrixUndefinedError, TransformError
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn, TensorLike
from pennylane.workflow.qnode import QNode


def catalyst_qjit(qnode):
    """A method checking whether a qnode is compiled by catalyst.qjit"""
    return qnode.__class__.__name__ == "QJIT" and hasattr(qnode, "user_function")


# pylint: disable=unused-argument
@transform
def matrix(
    op: Operator | QuantumScript | QNode | Callable, wire_order: Sequence | None = None
) -> TensorLike:
    pass


@matrix.register
def _matrix_op(op: Operator, wire_order: Sequence | None = None) -> TensorLike:
    if wire_order and not set(op.wires).issubset(wire_order):
        raise TransformError(
            f"Wires in circuit {list(op.wires)} are inconsistent with "
            f"those in wire_order {list(wire_order)}"
        )

    QueuingManager.remove(op)

    if op.has_matrix:
        return op.matrix(wire_order=wire_order)
    if op.has_sparse_matrix:
        return op.sparse_matrix(wire_order=wire_order).todense()
    if op.has_decomposition:
        with QueuingManager.stop_recording():
            ops = op.decomposition()
        return matrix(QuantumScript(ops), wire_order=wire_order or op.wires)

    raise MatrixUndefinedError(
        "Operator must define a matrix, sparse matrix, or decomposition for use with qml.matrix."
    )


@matrix.register
def _matrix_pauli(op: PauliWord | PauliSentence, wire_order: Sequence | None = None) -> TensorLike:
    if wire_order is None and len(op.wires) > 1:
        raise ValueError(
            "wire_order is required by qml.matrix() for PauliWords "
            "or PauliSentences with more than one wire."
        )
    return op.to_mat(wire_order=wire_order)


@matrix.register
def _matrix_tape(op: QuantumScript, wire_order: Sequence | None = None) -> TensorLike:
    if wire_order is None:
        error_base_str = "wire_order is required by qml.matrix() for tapes"
        if len(op.wires) > 1:
            raise ValueError(error_base_str + " with more than one wire.")
        if len(op.wires) == 0:
            raise ValueError(error_base_str + " without wires.")

    return _matrix_transform(op, wire_order=wire_order)


@matrix.register
def _matrix_qnode(op: QNode, wire_order: Sequence | None = None) -> TensorLike:
    if catalyst_qjit(op):
        op = op.user_function

    if wire_order is None and op.device.wires is None:
        raise ValueError(
            "wire_order is required by qml.matrix() for QNodes if the device does "
            "not have wires specified."
        )
    return _matrix_transform(op, wire_order=wire_order)


@matrix.register
def _matrix_function(op: Callable, wire_order: Sequence | None = None) -> TensorLike:
    if getattr(op, "num_wires", 0) != 1 and wire_order is None:
        raise ValueError("wire_order is required by qml.matrix() for quantum functions.")

    return _matrix_transform(op, wire_order=wire_order)


# Dispatching is ordered from most to least specific, so the fallback must be last.
@matrix.register
def _matrix_fallback(op: object, **_kwargs) -> TensorLike:
    raise TransformError(
        f"No matrix transform registered for type {type(op).__name__}. "
        "The qml.matrix transform only supports Operators, PauliWord, PauliSentence, "
        "QuantumScripts, QNodes, and callables."
    )


@partial(transform, is_informative=True)
def _matrix_transform(
    tape: QuantumScript, wire_order: Sequence | None = None, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:

    if wire_order and not set(tape.wires).issubset(wire_order):
        raise TransformError(
            f"Wires in circuit {list(tape.wires)} are inconsistent with "
            f"those in wire_order {list(wire_order)}"
        )

    wires = kwargs.get("device_wires", None) or tape.wires
    wire_order = wire_order or wires

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""

        params = res[0].get_parameters(trainable_only=False)
        interface = math.get_interface(*params)

        # initialize the unitary matrix
        if len(res[0].operations) == 0:
            result = math.eye(2 ** len(wire_order), like=interface)
        else:
            result = matrix(res[0].operations[0], wire_order=wire_order)

        for op in res[0].operations[1:]:
            U = matrix(op, wire_order=wire_order)
            # Coerce the matrices U and result and use matrix multiplication. Broadcasted axes
            # are handled correctly automatically by ``matmul`` (See e.g. NumPy documentation)
            result = math.matmul(*math.coerce([U, result], like=interface), like=interface)

        return result

    return [tape], processing_fn


@_matrix_transform.register
def _matrix_transform_qnode(qnode: QNode, *targs, **tkwargs):
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return _matrix_transform.generic_apply_transform(qnode, *targs, **tkwargs)
