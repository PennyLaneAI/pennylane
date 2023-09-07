# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the qml.eigvals function.
"""
from typing import Sequence, Callable
import warnings

# pylint: disable=protected-access
from functools import reduce, partial, singledispatch
import scipy

import pennylane as qml
from pennylane.transforms.core import transform
from pennylane.typing import TensorLike


@singledispatch
def eigvals(op: qml.operation.Operator, k=1, which="SA") -> TensorLike:
    r"""The eigenvalues of one or more operations.

    .. note::

        For a :class:`~.SparseHamiltonian` object, the eigenvalues are computed with the efficient
        ``scipy.sparse.linalg.eigsh`` method which returns :math:`k` eigenvalues. The default value
        of :math:`k` is :math:`1`. For an :math:`N \times N` sparse matrix, :math:`k` must be
        smaller than :math:`N - 1`, otherwise ``scipy.sparse.linalg.eigsh`` fails. If the requested
        :math:`k` is equal or larger than :math:`N - 1`, the regular ``qml.math.linalg.eigvalsh``
        is applied on the dense matrix. For more details see the ``scipy.sparse.linalg.eigsh``
        `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh>`_.

    Args:
        op (.Operator or .QuantumTape): A quantum operator or tape.
        k (int): The number of eigenvalues to be returned for a :class:`~.SparseHamiltonian`.
        which (str): Method for computing the eigenvalues of a :class:`~.SparseHamiltonian`. The
            possible methods are ``'LM'`` (largest in magnitude), ``'SM'`` (smallest in magnitude),
            ``'LA'`` (largest algebraic), ``'SA'`` (smallest algebraic) and ``'BE'`` (:math:`k/2`
            from each end of the spectrum).

    Returns:
        TensorLike or (Sequence[.QuantumTape], Callable): If an operator is provided as input, the eigenvalues
        are returned directly. If a quantum tape is provided as input, a list of transformed tapes and a post-processing
        function are returned. When called, this function will return the unitary matrix in the appropriate autodiff
        framework (Autograd, TensorFlow, PyTorch, JAX) given its parameters.

    **Example**

    Given an operation, ``qml.eigvals`` returns the eigenvalues:

    >>> op = qml.PauliZ(0) @ qml.PauliX(1) - 0.5 * qml.PauliY(1)
    >>> qml.eigvals(op)
    array([-1.11803399, -1.11803399,  1.11803399,  1.11803399])

    It can also be used in a functional form:

    >>> x = torch.tensor(0.6, requires_grad=True)
    >>> eigval_fn = qml.eigvals(qml.RX)
    >>> eigval_fn(x, wires=0)
    tensor([0.9553+0.2955j, 0.9553-0.2955j], grad_fn=<LinalgEigBackward>)

    In its functional form, it is fully differentiable with respect to gate arguments:

    >>> loss = torch.real(torch.sum(eigval_fn(x, wires=0)))
    >>> loss.backward()
    >>> x.grad
    tensor(-0.2955)

    This operator transform can also be applied to QNodes, tapes, and quantum functions
    that contain multiple operations; see Usage Details below for more details.

    .. details::
        :title: Usage Details

        ``qml.eigvals`` can also be used with QNodes, tapes, or quantum functions that
        contain multiple operations. However, in this situation, **eigenvalues may
        be computed numerically**. This can lead to a large computational overhead
        for a large number of wires.

        Consider the following quantum function:

        .. code-block:: python3

            def circuit(theta):
                qml.RX(theta, wires=1)
                qml.PauliZ(wires=0)

        We can use ``qml.eigvals`` to generate a new function that returns the eigenvalues
        corresponding to the function ``circuit``:

        >>> eigvals_fn = qml.eigvals(circuit)
        >>> theta = np.pi / 4
        >>> eigvals_fn(theta)
        array([ 0.92387953+0.38268343j,  0.92387953-0.38268343j,
               -0.92387953+0.38268343j, -0.92387953-0.38268343j])
    """
    if not isinstance(op, qml.operation.Operator):
        return _eigvals_tape(op)

    if isinstance(op, qml.Hamiltonian):
        warnings.warn(
            "For Hamiltonians, the eigenvalues will be computed numerically. "
            "This may be computationally intensive for a large number of wires. "
            "Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
            UserWarning,
        )
        return qml.math.linalg.eigvalsh(qml.matrix(op))

    if isinstance(op, qml.SparseHamiltonian):
        sparse_matrix = op.sparse_matrix()
        if k < sparse_matrix.shape[0] - 1:
            return scipy.sparse.linalg.eigsh(sparse_matrix, k=k, which=which)[0]
        return qml.math.linalg.eigvalsh(sparse_matrix.toarray())

    # TODO: make `eigvals` take a `wire_order` argument to mimic `matrix`
    return op.eigvals()


@partial(transform, is_informative=True)
@eigvals.register
def _eigvals_tape(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    def processing_fn(res):
        op_wires = [op.wires for op in res[0].operations]
        all_wires = qml.wires.Wires.all_wires(op_wires).tolist()
        unique_wires = qml.wires.Wires.unique_wires(op_wires).tolist()

        if len(all_wires) != len(unique_wires):
            warnings.warn(
                "For multiple operations, the eigenvalues will be computed numerically. "
                "This may be computationally intensive for a large number of wires.",
                UserWarning,
            )
            return qml.math.linalg.eigvals(qml.matrix(res[0]))

        # TODO: take into account wire ordering, by reordering eigenvalues
        # as per operator wires/wire ordering, and by inserting implicit identity
        # matrices (eigenvalues [1, 1]) at missing locations.

        ev = [eigvals(op) for op in res[0].operations]

        if len(ev) == 1:
            return ev[0]

        return reduce(qml.math.kron, ev)

    return [tape], processing_fn
