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
This module contains utilities and auxiliary functions which are shared
across the PennyLane submodules.
"""
# pylint: disable=protected-access,too-many-branches
from collections.abc import Iterable
import functools
import inspect
import itertools
import numbers
from operator import matmul

import numpy as np
import scipy

import pennylane as qml


def decompose_hamiltonian(H, hide_identity=False, wire_order=None):
    r"""Decomposes a Hermitian matrix into a linear combination of Pauli operators.

    Args:
        H (array[complex]): a Hermitian matrix of dimension :math:`2^n\times 2^n`
        hide_identity (bool): does not include the :class:`~.Identity` observable within
            the tensor products of the decomposition if ``True``

    Returns:
        tuple[list[float], list[~.Observable]]: a list of coefficients and a list
        of corresponding tensor products of Pauli observables that decompose the Hamiltonian.

    **Example:**

    We can use this function to compute the Pauli operator decomposition of an arbitrary Hermitian
    matrix:

    >>> A = np.array(
    ... [[-2, -2+1j, -2, -2], [-2-1j,  0,  0, -1], [-2,  0, -2, -1], [-2, -1, -1,  0]])
    >>> coeffs, obs_list = decompose_hamiltonian(A)
    >>> coeffs
    [-1.0, -1.5, -0.5, -1.0, -1.5, -1.0, -0.5, 1.0, -0.5, -0.5]

    We can use the output coefficients and tensor Pauli terms to construct a :class:`~.Hamiltonian`:

    >>> H = qml.Hamiltonian(coeffs, obs_list)
    >>> print(H)
    (-1.0) [I0 I1]
    + (-1.5) [X1]
    + (-0.5) [Y1]
    + (-1.0) [Z1]
    + (-1.5) [X0]
    + (-1.0) [X0 X1]
    + (-0.5) [X0 Z1]
    + (1.0) [Y0 Y1]
    + (-0.5) [Z0 X1]
    + (-0.5) [Z0 Y1]

    This Hamiltonian can then be used in defining VQE problems using :class:`~.ExpvalCost`.
    """
    n = int(np.log2(len(H)))
    N = 2**n

    if wire_order is None:
        wire_order = range(n)

    if H.shape != (N, N):
        raise ValueError(
            "The Hamiltonian should have shape (2**n, 2**n), for any qubit number n>=1"
        )

    if not np.allclose(H, H.conj().T):
        raise ValueError("The Hamiltonian is not Hermitian")

    paulis = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
    obs = []
    coeffs = []

    for term in itertools.product(paulis, repeat=n):
        matrices = [i.compute_matrix() for i in term]
        coeff = np.trace(functools.reduce(np.kron, matrices) @ H) / N
        coeff = np.real_if_close(coeff).item()

        if not np.allclose(coeff, 0):
            coeffs.append(coeff)

            if not all(t is qml.Identity for t in term) and hide_identity:
                obs.append(
                    functools.reduce(
                        matmul,
                        [t(i) for i, t in zip(wire_order, term) if t is not qml.Identity],
                    )
                )
            else:
                obs.append(functools.reduce(matmul, [t(i) for i, t in enumerate(term)]))

    return coeffs, obs


def sparse_hamiltonian(H, wires=None):
    r"""Computes the sparse matrix representation a Hamiltonian in the computational basis.

    Args:
        H (~.Hamiltonian): Hamiltonian operator for which the matrix representation should be
         computed
        wires (Iterable): Wire labels that indicate the order of wires according to which the matrix
         is constructed. If not profided, ``H.wires`` is used.

    Returns:
        coo_matrix: a sparse matrix in scipy coordinate list (COO) format with dimension
        :math:`(2^n, 2^n)`, where :math:`n` is the number of wires

    **Example:**

    This function can be used by passing a `qml.Hamiltonian` object as:

    >>> coeffs = [1, -0.45]
    >>> obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> H_sparse = sparse_hamiltonian(H)
    >>> H_sparse
    <4x4 sparse matrix of type '<class 'numpy.complex128'>'
        with 2 stored elements in COOrdinate format>

    The resulting sparse matrix can be either used directly or transformed into a numpy array:

    >>> H_sparse.toarray()
    array([[ 1.+0.j  ,  0.+0.j  ,  0.+0.45j,  0.+0.j  ],
           [ 0.+0.j  , -1.+0.j  ,  0.+0.j  ,  0.-0.45j],
           [ 0.-0.45j,  0.+0.j  , -1.+0.j  ,  0.+0.j  ],
           [ 0.+0.j  ,  0.+0.45j,  0.+0.j  ,  1.+0.j  ]])
    """
    if not isinstance(H, qml.Hamiltonian):
        raise TypeError("Passed Hamiltonian must be of type `qml.Hamiltonian`")

    if wires is None:
        wires = H.wires
    else:
        wires = qml.wires.Wires(wires)

    n = len(wires)
    matrix = scipy.sparse.coo_matrix((2**n, 2**n), dtype="complex128")

    coeffs = qml.math.toarray(H.data)

    for coeff, op in zip(coeffs, H.ops):

        obs = []
        for o in qml.operation.Tensor(op).obs:
            if len(o.wires) > 1:
                # todo: deal with operations created from multi-qubit operations such as Hermitian
                raise ValueError(
                    f"Can only sparsify Hamiltonians whose constituent observables consist of "
                    f"(tensor products of) single-qubit operators; got {op}."
                )
            obs.append(scipy.sparse.coo_matrix(o.get_matrix()))

        mat = [scipy.sparse.eye(2, format="coo")] * n

        for i, wire in enumerate(op.wires):
            # find index of this wire in the ordering
            idx = wires.index(wire)
            mat[idx] = obs[i]

        matrix += functools.reduce(lambda i, j: scipy.sparse.kron(i, j, format="coo"), mat) * coeff

    return matrix.tocoo()


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
    elif isinstance(x, qml.wires.Wires):
        # Reursive calls to flatten `Wires` will cause infinite recursion (`Wires` atoms are `Wires`).
        # Since Wires are always flat, just yield.
        for item in x:
            yield item
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
    if isinstance(model, (numbers.Number, str)):
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

    raise TypeError(f"Unsupported type in the model: {type(model)}")


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


def expand(matrix, original_wires, expanded_wires):
    r"""Expand an operator matrix to more wires.

    .. note::

        This function has essentially the same behaviour as :func:`.operation.expand_matrix`, but is not
        fully differentiable.


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

    if qml.math.shape(matrix) != (2**N, 2**N):
        raise ValueError(
            "Matrix parameter must be of size (2**len(original_wires), 2**len(original_wires))"
        )

    dims = [2] * (2 * N)
    tensor = qml.math.reshape(matrix, dims)

    interface = qml.math.get_interface(tensor)

    if D > 0:
        extra_dims = [2] * (2 * D)
        identity = qml.math.reshape(qml.math.eye(2**D), extra_dims)

        if interface == "tensorflow":
            identity = qml.math.cast_like(identity, tensor)

        expanded_tensor = qml.math.tensordot(tensor, identity, axes=0)
        # Fix order of tensor factors
        expanded_tensor = qml.math.moveaxis(
            expanded_tensor, tuple(range(2 * N, 2 * N + D)), tuple(range(N, N + D))
        )
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = qml.math.moveaxis(
        expanded_tensor, tuple(original_indices), tuple(wire_indices)
    )
    expanded_tensor = qml.math.moveaxis(
        expanded_tensor, tuple(original_indices + M), tuple(wire_indices + M)
    )

    return qml.math.reshape(expanded_tensor, (2**M, 2**M))


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

    if qml.math.shape(vector) != (2**N,):
        raise ValueError("Vector parameter must be of length 2**len(original_wires)")

    dims = [2] * N
    tensor = qml.math.reshape(vector, dims)

    if D > 0:
        extra_dims = [2] * D
        ones = qml.math.ones(2**D).reshape(extra_dims)
        expanded_tensor = qml.math.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = qml.math.moveaxis(
        expanded_tensor, tuple(original_indices), tuple(wire_indices)
    )

    return qml.math.reshape(expanded_tensor, 2**M)
