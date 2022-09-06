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
# pylint: disable=protected-access
import pennylane as qml


@qml.op_transform
def matrix(op, *, wire_order=None):
    r"""The matrix representation of an operation or quantum circuit.

    Args:
        op (.Operator, pennylane.QNode, .QuantumTape, or Callable): An operator, quantum node, tape,
            or function that applies quantum operations.
        wire_order (Sequence[Any], optional): Order of the wires in the quantum circuit.
            Defaults to the order in which the wires appear in the quantum function.

    Returns:
        tensor_like or function: Function which accepts the same arguments as the QNode or quantum
        function. When called, this function will return the unitary matrix in the appropriate
        autodiff framework (Autograd, TensorFlow, PyTorch, JAX) given its parameters.

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

        ``qml.matrix`` can also be used with QNodes, tapes, or quantum functions that
        contain multiple operations.

        Consider the following quantum function:

        .. code-block:: python3

            def circuit(theta):
                qml.RX(theta, wires=1)
                qml.PauliZ(wires=0)

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
        was used, and the unitary matrix corresponds to the operation :math:`Z\otimes R_X(\theta)`. To
        obtain the matrix for :math:`R_X(\theta)\otimes Z`, specify ``wire_order=[0, 1]`` in the
        function call:

        >>> matrix = qml.matrix(circuit, wire_order=[0, 1])

        You can also get the unitary matrix for operations on a subspace of a larger Hilbert space. For
        example, with the same function ``circuit`` and ``wire_order=["a", 0, "b", 1]`` you obtain the
        :math:`16\times 16` matrix for the operation :math:`I\otimes Z\otimes I\otimes  R_X(\theta)`.

        This unitary matrix can also be used in differentiable calculations. For example, consider the
        following cost function:

        .. code-block:: python

            def circuit(theta):
                qml.RX(theta, wires=1) qml.PauliZ(wires=0)
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
    """
    if isinstance(op, qml.operation.Tensor) and wire_order is not None:
        op = 1.0 * op  # convert to a Hamiltonian

    if isinstance(op, qml.Hamiltonian):
        return qml.utils.sparse_hamiltonian(op, wires=wire_order).toarray()

    return op.matrix(wire_order=wire_order)


@matrix.tape_transform
def _matrix(tape, wire_order=None):
    """Defines how matrix works if applied to a tape containing multiple operations."""
    params = tape.get_parameters(trainable_only=False)
    interface = qml.math._multi_dispatch(params)

    wire_order = wire_order or tape.wires

    # initialize the unitary matrix
    result = qml.math.eye(2 ** len(wire_order), like=interface)

    result_is_broadcasted = False
    for op in tape.operations:
        U = matrix(op, wire_order=wire_order)
        U_is_broadcasted = qml.math.ndim(U) == 3
        if U_is_broadcasted and result_is_broadcasted:
            # If both, U and result are broadcasted, we need a special syntax
            result = qml.math.stack([qml.math.dot(u, _unitary) for u, _unitary in zip(U, result)])
        else:
            # This covers the cases where at most one of U and result is broadcasted
            result = qml.math.tensordot(U, result, axes=[[-1], [-2]])
            # If result already was broadcasted, we need to move the corresponding axis up front
            if result_is_broadcasted:
                result = qml.math.moveaxis(result, 1, 0)

        if U_is_broadcasted:
            result_is_broadcasted = True

    return result
