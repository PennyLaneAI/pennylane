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
import warnings
from functools import partial, reduce

import scipy

import pennylane as qml
from pennylane import transform
from pennylane.exceptions import TransformError
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, TensorLike


def eigvals(op: qml.operation.Operator, k=1, which="SA") -> TensorLike:
    r"""The eigenvalues of one or more operations.

    .. note::

        - For a :class:`~.SparseHamiltonian` object, the eigenvalues are computed with the efficient
          ``scipy.sparse.linalg.eigsh`` method which returns :math:`k` eigenvalues. The default value
          of :math:`k` is :math:`1`. For an :math:`N \times N` sparse matrix, :math:`k` must be
          smaller than :math:`N - 1`, otherwise ``scipy.sparse.linalg.eigsh`` fails. If the requested
          :math:`k` is equal or larger than :math:`N - 1`, the regular ``qml.math.linalg.eigvalsh``
          is applied on the dense matrix. For more details see the ``scipy.sparse.linalg.eigsh``
          `documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#scipy.sparse.linalg.eigsh>`_.
        - A second-quantized :mod:`molecular Hamiltonian <pennylane.qchem.molecular_hamiltonian>` is
          independent of the number of electrons and its eigenspectrum contains the energies of the
          neutral and charged molecules. Therefore, the `smallest` eigenvalue returned by ``qml.eigvals``
          for a molecular Hamiltonian might not always correspond to the neutral molecule.

    Args:
        op (Operator or QNode or QuantumTape or Callable): A quantum operator or quantum circuit.
        k (int): The number of eigenvalues to be returned for a :class:`~.SparseHamiltonian`.
        which (str): Method for computing the eigenvalues of a :class:`~.SparseHamiltonian`. The
            possible methods are ``'LM'`` (largest in magnitude), ``'SM'`` (smallest in magnitude),
            ``'LA'`` (largest algebraic), ``'SA'`` (smallest algebraic) and ``'BE'`` (:math:`k/2`
            from each end of the spectrum).

    Returns:
        TensorLike or qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        If an operator is provided as input, the eigenvalues are returned directly in the form of a tensor.
        Otherwise, the transformed circuit is returned as described in :func:`qml.transform <pennylane.transform>`.
        Executing this circuit will provide the eigenvalues as a tensor.

    **Example**

    Given an operation, ``qml.eigvals`` returns the eigenvalues:

    >>> op = qml.Z(0) @ qml.X(1) - 0.5 * qml.Y(1)
    >>> qml.eigvals(op)
    array([-1.11803399, -1.11803399,  1.11803399,  1.11803399])

    It can also be used in a functional form:

    >>> x = torch.tensor(0.6, requires_grad=True)
    >>> eigval_fn = qml.eigvals(qml.RX)
    >>> eigval_fn(x, wires=0)
    tensor([0.9553+0.2955j, 0.9553-0.2955j], grad_fn=<LinalgEigBackward0>)

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

        .. code-block:: python

            def circuit(theta):
                qml.RX(theta, wires=1)
                qml.Z(0)

        We can use ``qml.eigvals`` to generate a new function that returns the eigenvalues
        corresponding to the function ``circuit``:

        >>> eigvals_fn = qml.eigvals(circuit)
        >>> theta = np.pi / 4
        >>> eigvals_fn(theta)
        array([ 0.92387953+0.38268343j, -0.92387953-0.38268343j,
            0.92387953-0.38268343j, -0.92387953+0.38268343j])
    """
    if not isinstance(op, qml.operation.Operator):
        if not isinstance(op, (qml.tape.QuantumScript, qml.QNode)) and not callable(op):
            raise TransformError("Input is not an Operator, tape, QNode, or quantum function")
        return _eigvals_transform(op, k=k, which=which)

    if isinstance(op, qml.SparseHamiltonian):
        sparse_matrix = op.sparse_matrix()
        if k < sparse_matrix.shape[0] - 1:
            return scipy.sparse.linalg.eigsh(sparse_matrix, k=k, which=which)[0]
        return qml.math.linalg.eigvalsh(sparse_matrix.toarray())

    # TODO: make `eigvals` take a `wire_order` argument to mimic `matrix`
    try:
        return op.eigvals()
    except qml.operation.EigvalsUndefinedError:
        return eigvals(qml.tape.QuantumScript(op.decomposition()), k=k, which=which)


@partial(transform, is_informative=True)
def _eigvals_transform(
    tape: QuantumScript, k=1, which="SA"
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    def processing_fn(res):
        [qs] = res
        op_wires = [op.wires for op in qs.operations]
        all_wires = qml.wires.Wires.all_wires(op_wires).tolist()
        unique_wires = qml.wires.Wires.unique_wires(op_wires).tolist()

        if len(all_wires) != len(unique_wires):
            warnings.warn(
                "For multiple operations, the eigenvalues will be computed numerically. "
                "This may be computationally intensive for a large number of wires.",
                UserWarning,
            )
            matrix = qml.matrix(qs, wire_order=qs.wires)
            return qml.math.linalg.eigvals(matrix)

        # TODO: take into account wire ordering, by reordering eigenvalues
        # as per operator wires/wire ordering, and by inserting implicit identity
        # matrices (eigenvalues [1, 1]) at missing locations.

        ev = [eigvals(op, k=k, which=which) for op in qs.operations]

        if len(ev) == 1:
            return ev[0]

        return reduce(qml.math.kron, ev)

    return [tape], processing_fn
