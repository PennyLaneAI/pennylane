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
# pylint: disable=protected-access,too-many-branches
from typing import Sequence, Callable, Union
from functools import partial
from warnings import warn

import pennylane as qml
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane import transform
from pennylane.typing import TensorLike
from pennylane.operation import Operator
from pennylane.pauli import PauliWord, PauliSentence

_wire_order_none_warning = (
    "Calling qml.matrix() on a QuantumTape, quantum functions, or certain QNodes without "
    "specifying a value for wire_order is deprecated. Please provide a wire_order value to avoid "
    "future issues. Note that a wire order is not required for QNodes with devices that have "
    "wires specified, because the wire order is taken from the device."
)


def matrix(op: Union[Operator, PauliWord, PauliSentence], wire_order=None) -> TensorLike:
    r"""The matrix representation of an operation or quantum circuit.

    Args:
        op (Operator or QNode or QuantumTape or Callable or PauliWord or PauliSentence): A quantum operator or quantum circuit.
        wire_order (Sequence[Any], optional): Order of the wires in the quantum circuit.
            The default wire order depends on the type of ``op``:

            - If ``op`` is a :class:`~.QNode`, then the wire order is determined by the
              associated device's wires, if provided.

            - Otherwise, the wire order is determined by the order in which wires
              appear in the circuit.

            - See the usage details for more information.

    Returns:
        TensorLike or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        If an operator, :class:`~PauliWord` or :class:`~PauliSentence` is provided as input, the matrix is returned directly in the form of a tensor.
        Otherwise, the transformed circuit is returned as described in :func:`qml.transform <pennylane.transform>`.
        Executing this circuit will provide its matrix representation.

    **Example**

    Given an instantiated operator, ``qml.matrix`` returns the matrix representation:

    >>> op = qml.RX(0.54, wires=0)
    >>> qml.matrix(op)
    [[0.9637709+0.j         0.       -0.26673144j]
    [0.       -0.26673144j 0.9637709+0.j        ]]

    It can also be used in a functional form:

    >>> x = torch.tensor(0.6, requires_grad=True)
    >>> matrix_fn = qml.matrix(qml.RX)
    >>> matrix_fn(x, wires=0)
    tensor([[0.9553+0.0000j, 0.0000-0.2955j],
            [0.0000-0.2955j, 0.9553+0.0000j]], grad_fn=<AddBackward0>)

    In its functional form, it is fully differentiable with respect to gate arguments:

    >>> loss = torch.real(torch.trace(matrix_fn(x, wires=0)))
    >>> loss.backward()
    >>> x.grad
    tensor(-0.5910)

    This operator transform can also be applied to QNodes, tapes, and quantum functions
    that contain multiple operations; see Usage Details below for more details.

    .. details::
        :title: Usage Details

        ``qml.matrix`` can also be used with :class:`~PauliWord` and :class:`~PauliSentence` instances.
        Internally, we are using their ``to_mat()`` methods.

        >>> X0 = PauliWord({0:"X"})
        >>> np.allclose(qml.matrix(X0), X0.to_mat())
        True

        ``qml.matrix`` can also be used with QNodes, tapes, or quantum functions that
        contain multiple operations.

        Consider the following quantum function:

        .. code-block:: python3

            def circuit(theta):
                qml.RX(theta, wires=1)
                qml.Z(0)

        We can use ``qml.matrix`` to generate a new function that returns the unitary matrix
        corresponding to the function ``circuit``:

        >>> matrix_fn = qml.matrix(circuit)
        >>> theta = np.pi / 4
        >>> matrix_fn(theta)
        array([[ 0.92387953+0.j,  0.+0.j ,  0.-0.38268343j,  0.+0.j],
        [ 0.+0.j,  -0.92387953+0.j,  0.+0.j,  0. +0.38268343j],
        [ 0. -0.38268343j,  0.+0.j,  0.92387953+0.j,  0.+0.j],
        [ 0.+0.j,  0.+0.38268343j,  0.+0.j,  -0.92387953+0.j]])

        Note that since ``wire_order`` was not specified, the default order ``[1, 0]`` for ``circuit``
        was used, and the unitary matrix corresponds to the operation :math:`R_X(\theta)\otimes Z`. To
        obtain the matrix for :math:`Z\otimes R_X(\theta)`, specify ``wire_order=[0, 1]`` in the
        function call:

        >>> matrix = qml.matrix(circuit, wire_order=[0, 1])

        You can also get the unitary matrix for operations on a subspace of a larger Hilbert space. For
        example, with the same function ``circuit`` and ``wire_order=["a", 0, "b", 1]`` you obtain the
        :math:`16\times 16` matrix for the operation :math:`I\otimes Z\otimes I\otimes  R_X(\theta)`.

        This unitary matrix can also be used in differentiable calculations. For example, consider the
        following cost function:

        .. code-block:: python

            def circuit(theta):
                qml.RX(theta, wires=1) qml.Z(0)
                qml.CNOT(wires=[0, 1])

            def cost(theta):
                matrix = qml.matrix(circuit)(theta)
                return np.real(np.trace(matrix))

        Since this cost function returns a real scalar as a function of ``theta``, we can differentiate
        it:

        >>> theta = np.array(0.3, requires_grad=True)
        >>> cost(theta)
        1.9775421558720845
        >>> qml.grad(cost)(theta)
        -0.14943813247359922

        .. note::

            When using ``qml.matrix`` with a ``QNode``, unless specified, the device wire order will
            be used. If the device wires are not set, the wire order will be inferred
            from the quantum function used to create the ``QNode``. Consider the following example:

            .. code-block:: python

                def circuit():
                    qml.Hadamard(wires=1)
                    qml.CZ(wires=[0, 1])
                    qml.Hadamard(wires=1)
                    return qml.state()

                dev_with_wires = qml.device("default.qubit", wires=[0, 1])
                dev_without_wires = qml.device("default.qubit")

                qnode_with_wires = qml.QNode(circuit, dev_with_wires)
                qnode_without_wires = qml.QNode(circuit, dev_without_wires)

            >>> qml.matrix(qnode_with_wires)().round(2)
            array([[ 1.+0.j, -0.+0.j,  0.+0.j,  0.+0.j],
                   [-0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
                   [ 0.+0.j,  0.+0.j, -0.+0.j,  1.+0.j],
                   [ 0.+0.j,  0.+0.j,  1.+0.j, -0.+0.j]])
            >>> qml.matrix(qnode_without_wires)().round(2)
            array([[ 1.+0.j,  0.+0.j, -0.+0.j,  0.+0.j],
                   [ 0.+0.j, -0.+0.j,  0.+0.j,  1.+0.j],
                   [-0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
                   [ 0.+0.j,  1.+0.j,  0.+0.j, -0.+0.j]])

            The second matrix above uses wire order ``[1, 0]`` because the device does not have
            wires specified, and this is the order in which wires appear in ``circuit()``.

    """
    if not isinstance(op, Operator):
        if isinstance(op, (PauliWord, PauliSentence)):
            if wire_order is None and len(op.wires) > 1:
                warn(_wire_order_none_warning, qml.PennyLaneDeprecationWarning)
            return op.to_mat(wire_order=wire_order)

        if isinstance(op, qml.tape.QuantumScript):
            if wire_order is None and len(op.wires) > 1:
                warn(_wire_order_none_warning, qml.PennyLaneDeprecationWarning)
        elif isinstance(op, qml.QNode):
            if wire_order is None and op.device.wires is None:
                warn(_wire_order_none_warning, qml.PennyLaneDeprecationWarning)
        elif callable(op):
            if wire_order is None:
                warn(_wire_order_none_warning, qml.PennyLaneDeprecationWarning)
        else:
            raise OperationTransformError(
                "Input is not an Operator, tape, QNode, or quantum function"
            )
        return _matrix_transform(op, wire_order=wire_order)

    if isinstance(op, qml.operation.Tensor) and wire_order is not None:
        op = 1.0 * op  # convert to a Hamiltonian

    if isinstance(op, qml.Hamiltonian):
        return op.sparse_matrix(wire_order=wire_order).toarray()

    try:
        return op.matrix(wire_order=wire_order)
    except:  # pylint: disable=bare-except
        return matrix(op.expand(), wire_order=wire_order or op.wires)


@partial(transform, is_informative=True)
def _matrix_transform(
    tape: qml.tape.QuantumTape, wire_order=None, **kwargs
) -> (Sequence[qml.tape.QuantumTape], Callable):
    if not tape.wires:
        raise qml.operation.MatrixUndefinedError

    if wire_order and not set(tape.wires).issubset(wire_order):
        raise OperationTransformError(
            f"Wires in circuit {list(tape.wires)} are inconsistent with "
            f"those in wire_order {list(wire_order)}"
        )

    wires = kwargs.get("device_wires", None) or tape.wires
    wire_order = wire_order or wires

    def processing_fn(res):
        """Defines how matrix works if applied to a tape containing multiple operations."""

        params = res[0].get_parameters(trainable_only=False)
        interface = qml.math.get_interface(*params)

        # initialize the unitary matrix
        if len(res[0].operations) == 0:
            result = qml.math.eye(2 ** len(wire_order), like=interface)
        else:
            result = matrix(res[0].operations[0], wire_order=wire_order)

        for op in res[0].operations[1:]:
            U = matrix(op, wire_order=wire_order)
            # Coerce the matrices U and result and use matrix multiplication. Broadcasted axes
            # are handled correctly automatically by ``matmul`` (See e.g. NumPy documentation)
            result = qml.math.matmul(*qml.math.coerce([U, result], like=interface), like=interface)

        return result

    return [tape], processing_fn


@_matrix_transform.custom_qnode_transform
def _matrix_transform_qnode(self, qnode, targs, tkwargs):
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)
