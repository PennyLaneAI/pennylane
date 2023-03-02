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
This submodule contains the discrete-variable quantum operations that
accept a hermitian or an unitary matrix as a parameter.
"""
# pylint:disable=arguments-differ
import warnings

import numpy as np
from scipy.linalg import sqrtm, norm

import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
        unitary_check (bool): check for unitarity of the given matrix

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)
    >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.QubitUnitary(U, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(example_circuit())
    0.0
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(
        self, U, wires, do_queue=True, id=None, unitary_check=False
    ):  # pylint: disable=too-many-arguments
        # For pure QubitUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix

        wires = Wires(wires)

        U_shape = qml.math.shape(U)

        dim = 2 ** len(wires)

        if len(U_shape) not in {2, 3} or U_shape[-2:] != (dim, dim):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                f"to act on {len(wires)} wires."
            )

        # Check for unitarity; due to variable precision across the different ML frameworks,
        # here we issue a warning to check the operation, instead of raising an error outright.
        if unitary_check and not (
            qml.math.is_abstract(U)
            or qml.math.allclose(
                qml.math.einsum("...ij,...kj->...ik", U, qml.math.conj(U)),
                qml.math.eye(dim),
                atol=1e-6,
            )
        ):
            warnings.warn(
                f"Operator {U}\n may not be unitary."
                "Verify unitarity of operation, or use a datatype with increased precision.",
                UserWarning,
            )

        super().__init__(U, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(U):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.QubitUnitary.matrix`

        Args:
            U (tensor_like): unitary matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> U = np.array([[0.98877108+0.j, 0.-0.14943813j], [0.-0.14943813j, 0.98877108+0.j]])
        >>> qml.QubitUnitary.compute_matrix(U)
        [[0.98877108+0.j, 0.-0.14943813j],
        [0.-0.14943813j, 0.98877108+0.j]]
        """
        return U

    @staticmethod
    def compute_decomposition(U, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        A decomposition is only defined for matrices that act on either one or two wires. For more
        than two wires, this method raises a ``DecompositionUndefined``.

        See :func:`~.transforms.zyz_decomposition` and :func:`~.transforms.two_qubit_decomposition`
        for more information on how the decompositions are computed.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        >>> qml.QubitUnitary.compute_decomposition(U, 0)
        [Rot(tensor(3.14159265, requires_grad=True), tensor(1.57079633, requires_grad=True), tensor(0., requires_grad=True), wires=[0])]

        """
        # Decomposes arbitrary single-qubit unitaries as Rot gates (RZ - RY - RZ format),
        # or a single RZ for diagonal matrices.
        shape = qml.math.shape(U)

        is_batched = len(shape) == 3
        shape_without_batch_dim = shape[1:] if is_batched else shape

        if shape_without_batch_dim == (2, 2):
            return qml.transforms.decompositions.zyz_decomposition(U, Wires(wires)[0])

        if shape_without_batch_dim == (4, 4):
            # TODO[dwierichs]: Implement decomposition of broadcasted unitary
            if is_batched:
                raise DecompositionUndefinedError(
                    "The decomposition of a two-qubit QubitUnitary does not support broadcasting."
                )

            return qml.transforms.two_qubit_decomposition(U, Wires(wires))

        return super(QubitUnitary, QubitUnitary).compute_decomposition(U, wires=wires)

    def adjoint(self):
        U = self.matrix()
        return QubitUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)

    def pow(self, z):
        if isinstance(z, int):
            return [QubitUnitary(qml.math.linalg.matrix_power(self.matrix(), z), wires=self.wires)]
        return super().pow(z)

    def _controlled(self, wire):
        return qml.ControlledQubitUnitary(*self.parameters, control_wires=wire, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class DiagonalQubitUnitary(Operation):
    r"""DiagonalQubitUnitary(D, wires)
    Apply an arbitrary fixed diagonal unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    @staticmethod
    def compute_matrix(D):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.matrix`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_matrix(torch.tensor([1, -1]))
        tensor([[ 1,  0],
                [ 0, -1]])
        """
        D = qml.math.asarray(D)

        if not qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D)):
            raise ValueError("Operator must be unitary.")

        # The diagonal is supposed to have one-dimension. If it is broadcasted, it has two
        if qml.math.ndim(D) == 2:
            return qml.math.stack([qml.math.diag(_D) for _D in D])

        return qml.math.diag(D)

    @staticmethod
    def compute_eigvals(D):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.eigvals`

        Args:
            D (tensor_like): diagonal of the matrix

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.DiagonalQubitUnitary.compute_eigvals(torch.tensor([1, -1]))
        tensor([ 1, -1])
        """
        D = qml.math.asarray(D)

        if not (
            qml.math.is_abstract(D)
            or qml.math.allclose(D * qml.math.conj(D), qml.math.ones_like(D))
        ):
            raise ValueError("Operator must be unitary.")

        return D

    @staticmethod
    def compute_decomposition(D, wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        ``DiagonalQubitUnitary`` decomposes into :class:`~.QubitUnitary`, which has further
        decompositions for one and two qubit matrices.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.DiagonalQubitUnitary.compute_decomposition([1, 1], wires=0)
        [QubitUnitary(array([[1, 0], [0, 1]]), wires=[0])]

        """
        return [QubitUnitary(DiagonalQubitUnitary.compute_matrix(D), wires=wires)]

    def adjoint(self):
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def pow(self, z):
        cast_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(cast_data**z, wires=self.wires)]

    def _controlled(self, control):
        return DiagonalQubitUnitary(
            qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]),
            wires=control + self.wires,
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)

class BlockEncode(Operation):
    r"""BlockEncode(a, wires)
    Apply an arbitrary matrix, :math:`A`, encoded in the top left block of a unitary matrix.

    .. math::

        \begin{align}
             U(A) &=
             \begin{bmatrix}
                A & \sqrt{I-AA^\dagger} \\
                \sqrt{I-A^\dagger A} & -A^\dagger
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        a (array[complex]): general n-by-m matrix to be encoded
        wires (Sequence[int] or int): the wire(s) the operation acts on
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> A = [[0.1,0.2],[0.3,0.4]]
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BlockEncode(A, wires=range(2))
    ...     return qml.state()
    ...     print(qml.matrix(example_circuit)())
    [[ 0.1         0.2         0.97283788 -0.05988708]
    [ 0.3         0.4        -0.05988708  0.86395228]
    [ 0.94561648 -0.07621992 -0.1        -0.3       ]
    [-0.07621992  0.89117368 -0.2        -0.4       ]]
    """

    num_params = 1
    num_wires = AnyWires

    def __init__(self, a, wires, do_queue=True, id=None):
        a = np.atleast_2d(a)
        wires = Wires(wires)
        if np.sum(qml.math.shape(a)) <= 2:
            normalization = a if a > 1 else 1
            subspace = (1, 1, 2 ** len(wires))
        else:
            normalization = np.max(
                [norm(a @ np.conj(a).T, ord=np.inf), norm(np.conj(a).T @ a, ord=np.inf)]
            )
            subspace = (*qml.math.shape(a), 2 ** len(wires))

        a = a / normalization if normalization > 1 else a

        # if subspace[2] < (subspace[0] + subspace[1]):
        if subspace[2] < np.max(subspace[0:-1])*2:
            raise ValueError(
                f"Block encoding a {subspace[0]} x {subspace[1]} matrix"
                f" requires a hilbert space of size at least "
                # f"{subspace[0] + subspace[1]} x {subspace[0] + subspace[1]}."
                f"{np.max(subspace[0:-1])*2} x {np.max(subspace[0:-1])*2}."
                f" Cannot be embedded in a {len(wires)} qubit system."
            )

        super().__init__(a, wires=wires, do_queue=do_queue, id=id)
        self.hyperparameters["norm"] = normalization
        self.hyperparameters["subspace"] = subspace

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        """Get the matrix representation of block encoding unitary."""
        a = params[0]
        n, m, k = hyperparams["subspace"]

        if np.sum(qml.math.shape(qml.math.atleast_2d(a))) <= 2:
            u = np.block(
                [[a, np.sqrt(1 - a * np.conj(a))], [np.sqrt(1 - a * np.conj(a)), -np.conj(a)]]
            )
        else:
            d1, d2 = qml.math.shape(a)
            u = np.block(
                [
                    [a, sqrtm(np.eye(d1) - a @ np.conj(a).T)],
                    [sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T],
                ]
            )
        if n + m < k:
            r = k - (n + m)
            u = np.block([[u, np.zeros((n + m, r))], [np.zeros((r, n + m)), np.eye(r)]])
        return u

    # @staticmethod
    # def compute_matrix(a,norm,subspace):
    #     """Get the matrix representation of block encoding unitary."""
    #     n, m, k = subspace

    #     if np.sum(qml.math.shape(qml.math.atleast_2d(a))) <= 2:
    #         u = a * np.diag([1, 0]) -np.conj(a)*np.diag([0,1]) + np.sqrt(1 - a * np.conj(a)) * np.fliplr(np.diag([1,0]))+ np.sqrt(1 - a * np.conj(a)) * np.fliplr(np.diag([0,1]))
        
    #     else:
    #         d1, d2 = qml.math.shape(a)
    #         # top = qml.math.hstack([a,sqrtm(np.eye(d1) - a @ np.conj(a).T)]) # -> object arrays are not supported
    #         # u=qml.math.block_diag([a,-np.conj(np.transpose(a))]) # -> works
    #         # u = qml.math.stack([a, -np.conj(np.transpose(a))]) # -> works

    #         # bottom = qml.math.hstack([sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T])
    #         # u = qml.math.vstack([top,bottom])

    #         # u = qml.math.block(
    #         #     [
    #         #         [a, sqrtm(np.eye(d1) - a @ np.conj(a).T)],
    #         #         [sqrtm(np.eye(d2) - np.conj(a).T @ a), -np.conj(a).T],
    #         #     ]
    #         # )
    #         # if d1 < d2:
    #         #     qml.math.concatenate([a,np.zeros(d3))

    #     # if n + m < k:
    #     #     r = k - (n + m)
    #     #     u = qml.math.block([[u, np.zeros((n + m, r))], [np.zeros((r, n + m)), np.eye(r)]])
        
    #     return a
