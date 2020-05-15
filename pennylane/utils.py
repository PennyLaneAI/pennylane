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
This module contains utilities and auxiliary functions which are shared
across the PennyLane submodules.
"""
# pylint: disable=protected-access
from collections.abc import Iterable
from collections import OrderedDict
import copy
import numbers
import functools
import inspect

import numpy as np

import pennylane as qml
from pennylane.variable import Variable


def _flatten(x):
    """Iterate recursively through an arbitrarily nested structure in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, Any): each element of an array or an Iterable may itself be any of these types

    Yields:
        Any: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from _flatten(x.flat)  # should we allow object arrays? or just "yield from x.flat"?
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from _flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`_flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        TypeError: if ``model`` contains an object of unsupported type

    Returns:
        Union[array, list, Any], array: first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    if isinstance(model, (numbers.Number, Variable, str)):
        return flat[0], flat[1:]

    if isinstance(model, np.ndarray):
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        return res, flat[idx:]

    if isinstance(model, Iterable):
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            res.append(val)
        return res, flat

    raise TypeError("Unsupported type in the model: {}".format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Raises:
        ValueError: if ``flat`` has more elements than ``model``
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError("Flattened iterable has more elements than the model.")
    return res


def _inv_dict(d):
    """Reverse a dictionary mapping.

    Returns multimap where the keys are the former values,
    and values are sets of the former keys.

    Args:
        d (dict[a->b]): mapping to reverse

    Returns:
        dict[b->set[a]]: reversed mapping
    """
    ret = {}
    for k, v in d.items():
        ret.setdefault(v, set()).add(k)
    return ret


def _get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (callable): a function

    Returns:
        dict[str, tuple]: mapping from argument name to (positional idx, default value)
    """
    signature = inspect.signature(func)
    return {
        k: (idx, v.default)
        for idx, (k, v) in enumerate(signature.parameters.items())
        if v.default is not inspect.Parameter.empty
    }


@functools.lru_cache()
def pauli_eigs(n):
    r"""Eigenvalues for :math:`A^{\otimes n}`, where :math:`A` is
    Pauli operator, or shares its eigenvalues.

    As an example if n==2, then the eigenvalues of a tensor product consisting
    of two matrices sharing the eigenvalues with Pauli matrices is returned.

    Args:
        n (int): the number of qubits the matrix acts on
    Returns:
        list: the eigenvalues of the specified observable
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


class OperationRecorder(qml.QueuingContext):
    """A template and quantum function inspector,
    allowing easy introspection of operators that have been
    applied without requiring a QNode.

    **Example**:

    The OperationRecorder is a context manager. Executing templates
    or quantum functions stores resulting applied operators in the
    recorder, which can then be printed.

    >>> weights = qml.init.strong_ent_layers_normal(n_layers=1, n_wires=2)
    >>>
    >>> with qml.utils.OperationRecorder() as rec:
    >>>    qml.templates.layers.StronglyEntanglingLayers(*weights, wires=[0, 1])
    >>>
    >>> print(rec)
    Operations
    ==========
    Rot(-0.10832656163640327, 0.14429091013664083, -0.010835826725765343, wires=[0])
    Rot(-0.11254523669444501, 0.0947222564914006, -0.09139600968423377, wires=[1])
    CNOT(wires=[0, 1])
    CNOT(wires=[1, 0])

    Alternatively, the :attr:`~.OperationRecorder.queue` attribute can be used
    to directly accessed the applied :class:`~.Operation` and :class:`~.Observable`
    objects.

    Attributes:
        queue (List[~.Operators]): list of operators applied within
            the OperatorRecorder context, includes operations and observables
        operations (List[~.Operations]): list of operations applied within
            the OperatorRecorder context
        observables (List[~.Observables]): list of observables applied within
            the OperatorRecorder context
    """

    def __init__(self):
        self.queue = []
        self.operations = None
        self.observables = None

    def _append_operator(self, operator):
        self.queue.append(operator)

    def _remove_operator(self, operator):
        self.queue.remove(operator)

    def __exit__(self, exception_type, exception_value, traceback):
        super().__exit__(exception_type, exception_value, traceback)

        # Remove duplicates that might have arisen from measurements
        self.queue = list(OrderedDict.fromkeys(self.queue))
        self.operations = list(
            filter(
                lambda op: not (
                    isinstance(op, qml.operation.Observable) and not op.return_type is None
                ),
                self.queue,
            )
        )
        self.observables = list(
            filter(
                lambda op: isinstance(op, qml.operation.Observable) and not op.return_type is None,
                self.queue,
            )
        )

    def __str__(self):
        output = ""
        output += "Operations\n"
        output += "==========\n"
        for op in self.operations:
            output += repr(op) + "\n"

        output += "\n"
        output += "Observables\n"
        output += "==========\n"
        for op in self.observables:
            output += repr(op) + "\n"

        return output


def inv(operation_list):
    """Invert a list of operations or a :doc:`template </introduction/templates>`.

    If the inversion happens inside a QNode, the operations are removed and requeued
    in the reversed order for proper inversion.

    **Example:**

    The following example illuminates the inversion of a template:

    .. code-block:: python3

        @qml.template
        def ansatz(weights, wires):
            for idx, wire in enumerate(wires):
                qml.RX(weights[idx], wires=[wire])

            for idx in range(len(wires) - 1):
                qml.CNOT(wires=[wires[idx], wires[idx + 1]])

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.inv(ansatz(weights, wires=[0, 1]))
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We may also invert an operation sequence:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit1():
            qml.T(wires=[0]).inv()
            qml.Hadamard(wires=[0]).inv()
            qml.S(wires=[0]).inv()
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev)
        def circuit2():
            qml.inv([qml.S(wires=[0]), qml.Hadamard(wires=[0]), qml.T(wires=[0])])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    Double checking that both circuits produce the same output:

    >>> ZZ1 = circuit1()
    >>> ZZ2 = circuit2()
    >>> assert ZZ1 == ZZ2
    True

    Args:
        operation_list (Iterable[~.Operation]): An iterable of operations

    Returns:
        List[~.Operation]: The inverted list of operations
    """
    if isinstance(operation_list, qml.operation.Operation):
        operation_list = [operation_list]
    elif operation_list is None:
        raise ValueError(
            "None was passed as an argument to inv. "
            "This could happen if inversion of a template without the template decorator is attempted."
        )
    elif callable(operation_list):
        raise ValueError(
            "A function was passed as an argument to inv. "
            "This could happen if inversion of a template function is attempted. "
            "Please use inv on the function including its arguments, as in inv(template(args))."
        )
    elif not isinstance(operation_list, Iterable):
        raise ValueError("The provided operation_list is not iterable.")

    non_ops = [
        (idx, op)
        for idx, op in enumerate(operation_list)
        if not isinstance(op, qml.operation.Operation)
    ]

    if non_ops:
        string_reps = [" operation_list[{}] = {}".format(idx, op) for idx, op in non_ops]
        raise ValueError(
            "The given operation_list does not only contain Operations."
            + "The following elements of the iterable were not Operations:"
            + ",".join(string_reps)
        )

    inv_ops = [op.inv() for op in reversed(copy.deepcopy(operation_list))]

    for op in operation_list:
        qml.QueuingContext.remove_operator(op)

    for inv_op in inv_ops:
        qml.QueuingContext.append_operator(inv_op)

    return inv_ops


def expand(matrix, original_wires, expanded_wires):
    r"""Expand a an operator matrix to more wires.

    Args:
        matrix (array): :math:`2^n \times 2^n` matrix where n = len(original_wires).
        original_wires (Sequence[int]): original wires of matrix
        expanded_wires (Union[Sequence[int], int]): expanded wires of matrix, can be shuffled.
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`2^m \times 2^m` matrix where m = len(expanded_wires).
    """
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))

    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")

    if matrix.shape != (2 ** N, 2 ** N):
        raise ValueError(
            "Matrix parameter must be of size (2**len(original_wires), 2**len(original_wires))"
        )

    dims = [2] * (2 * N)
    tensor = matrix.reshape(dims)

    if D > 0:
        extra_dims = [2] * (2 * D)
        identity = np.eye(2 ** D).reshape(extra_dims)
        expanded_tensor = np.tensordot(tensor, identity, axes=0)
        # Fix order of tensor factors
        expanded_tensor = np.moveaxis(expanded_tensor, range(2 * N, 2 * N + D), range(N, N + D))
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = np.moveaxis(expanded_tensor, original_indices, wire_indices)
    expanded_tensor = np.moveaxis(expanded_tensor, original_indices + M, wire_indices + M)

    return expanded_tensor.reshape((2 ** M, 2 ** M))


def expand_vector(vector, original_wires, expanded_wires):
    r"""Expand a vector to more wires.

    Args:
        vector (array): :math:`2^n` vector where n = len(original_wires).
        original_wires (Sequence[int]): original wires of vector
        expanded_wires (Union[Sequence[int], int]): expanded wires of vector, can be shuffled
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`2^m` vector where m = len(expanded_wires).
    """
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))

    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")

    if vector.shape != (2 ** N,):
        raise ValueError("Vector parameter must be of length 2**len(original_wires)")

    dims = [2] * N
    tensor = vector.reshape(dims)

    if D > 0:
        extra_dims = [2] * D
        ones = np.ones(2 ** D).reshape(extra_dims)
        expanded_tensor = np.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = np.moveaxis(expanded_tensor, original_indices, wire_indices)

    return expanded_tensor.reshape(2 ** M)
