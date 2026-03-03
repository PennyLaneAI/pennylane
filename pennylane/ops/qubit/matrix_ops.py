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
import scipy as sp
from scipy.linalg import fractional_matrix_power
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import math
from pennylane import numpy as pnp
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.decomposition.symbolic_decomposition import is_integer
from pennylane.exceptions import DecompositionUndefinedError
from pennylane.operation import FlatPytree, Operation
from pennylane.ops.op_math.decompositions.unitary_decompositions import (
    multi_qubit_decomp_rule,
    rot_decomp_rule,
    two_qubit_decomp_rule,
    xyx_decomp_rule,
    xzx_decomp_rule,
    zxz_decomp_rule,
    zyz_decomp_rule,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

_walsh_hadamard_matrix = np.array([[1, 1], [1, -1]]) / 2


def _walsh_hadamard_transform(D: TensorLike, n: int | None = None):
    r"""Compute the Walsh–Hadamard Transform of a one-dimensional array.

    Args:
        D (tensor_like): The array or tensor to be transformed. Must have a length that
            is a power of two.

    Returns:
        tensor_like: The transformed tensor with the same shape as the input ``D``.

    Due to the execution of the transform as a sequence of tensor multiplications
    with shapes ``(2, 2), (2, 2,... 2)->(2, 2,... 2)``, the theoretical scaling of this
    method is the same as the one for the
    `Fast Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Fast_Walsh-Hadamard_transform>`__:
    On ``n`` qubits, there are ``n`` calls to ``tensordot``, each multiplying a
    ``(2, 2)`` matrix to a ``(2,)*n`` vector, with a single axis being contracted. This means
    that there are ``n`` operations with a FLOP count of ``4 * 2**(n-1)``, where ``4`` is the cost
    of a single ``(2, 2) @ (2,)`` contraction and ``2**(n-1)`` is the number of copies due to the
    non-contracted ``n-1`` axes.
    Due to the large internal speedups of compiled matrix multiplication and compatibility
    with autodifferentiation frameworks, the approach taken here is favourable over a manual
    realization of the FWHT unless memory limitations restrict the creation of intermediate
    arrays.
    """
    orig_shape = qml.math.shape(D)
    n = n or int(qml.math.log2(orig_shape[-1]))
    # Reshape the array so that we may apply the Hadamard transform to each axis individually
    if broadcasted := len(orig_shape) > 1:
        new_shape = (orig_shape[0],) + (2,) * n
    else:
        new_shape = (2,) * n
    D = qml.math.reshape(D, new_shape)
    # Apply Hadamard transform to each axis, shifted by one for broadcasting
    for i in range(broadcasted, n + broadcasted):
        D = qml.math.tensordot(_walsh_hadamard_matrix, D, axes=[[1], [i]])
    # The axes are in reverted order after all matrix multiplications, so we need to transpose;
    # If D was broadcasted, this moves the broadcasting axis to first position as well.
    # Finally, reshape to original shape
    return qml.math.reshape(qml.math.transpose(D), orig_shape)


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary unitary matrix with a dimension that is a power of two.

    .. warning::

        The sparse matrix representation of QubitUnitary is still under development. Currently,
        we only support a limited set of interfaces that preserve the sparsity of the matrix,
        including :func:`~.adjoint`, :func:`~.pow`, and :meth:`~.QubitUnitary.compute_sparse_matrix`.
        Differentiability is not supported for sparse matrices.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (2,)
    * Gradient recipe: None

    Args:
        U (array[complex] or csr_matrix): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
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
    ...     return qml.expval(qml.Z(0))
    >>> print(example_circuit())
    0.0
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    resource_keys = {"num_wires"}

    grad_method = None
    """Gradient computation method."""

    def __init__(
        self,
        U: TensorLike | csr_matrix,
        wires: WiresLike,
        id: str | None = None,
        unitary_check: bool = False,
    ):
        wires = Wires(wires)
        U_shape = qml.math.shape(U)
        dim = 2 ** len(wires)

        # For pure QubitUnitary operations (not controlled), check that the number
        # of wires fits the dimensions of the matrix
        if len(U_shape) not in {2, 3} or U_shape[-2:] != (dim, dim):
            raise ValueError(
                f"Input unitary must be of shape {(dim, dim)} or (batch_size, {dim}, {dim}) "
                f"to act on {len(wires)} wires. Got shape {U_shape} instead."
            )

        # If the matrix is sparse, we need to convert it to a csr_matrix
        self._issparse = sp.sparse.issparse(U)
        if self._issparse:
            U = U.tocsr()

        # Check for unitarity; due to variable precision across the different ML frameworks,
        # here we issue a warning to check the operation, instead of raising an error outright.
        if unitary_check and not self._unitary_check(U, dim):
            warnings.warn(
                f"Operator {U}\n may not be unitary. "
                "Verify unitarity of operation, or use a datatype with increased precision.",
                UserWarning,
            )

        super().__init__(U, wires=wires, id=id)

    @staticmethod
    def _unitary_check(U, dim):
        if isinstance(U, csr_matrix):
            U_dagger = U.conjugate().transpose()
            identity = sp.sparse.eye(dim, format="csr")
            return sp.sparse.linalg.norm(U @ U_dagger - identity) < 1e-10
        return qml.math.allclose(
            qml.math.einsum("...ij,...kj->...ik", U, qml.math.conj(U)),
            qml.math.eye(dim),
            atol=1e-6,
        )

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @staticmethod
    def compute_matrix(U: TensorLike):  # pylint: disable=arguments-differ
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
         array([[0.988...+0.j        , 0.        -0.149...j],
                [0.        -0.149...j, 0.988...+0.j        ]])
        """
        if sp.sparse.issparse(U):
            raise qml.operation.MatrixUndefinedError(
                "U is sparse matrix. Use sparse_matrix method instead."
            )
        return U

    @staticmethod
    def compute_sparse_matrix(U: TensorLike, format="csr"):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a sparse matrix.

        Args:
            U (tensor_like): unitary matrix

        Returns:
            csr_matrix: sparse matrix representation

        **Example**

        >>> U = np.array([
        ...     [1, 0, 0, 0],
        ...     [0, 1, 0, 0],
        ...     [0, 0, 0, 1],
        ...     [0, 0, 1, 0]
        ... ])
        >>> U = sp.sparse.csr_matrix(U)
        >>> qml.QubitUnitary.compute_sparse_matrix(U)
        <Compressed Sparse Row sparse matrix of dtype 'int64'
            with 4 stored elements and shape (4, 4)>
        """
        if sp.sparse.issparse(U):
            return U.asformat(format)
        raise qml.operation.SparseMatrixUndefinedError(
            "U is a dense matrix. Use matrix method instead"
        )

    @staticmethod
    def compute_decomposition(U: TensorLike, wires: WiresLike):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        See :func:`~.ops.one_qubit_decomposition`, :func:`~.ops.two_qubit_decomposition`
        and :func:`~.ops.multi_qubit_decomposition` for more information on how the decompositions are computed.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            U (array[complex]): square unitary matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> U = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        >>> decomp = qml.QubitUnitary.compute_decomposition(U, 0)
        >>> from pprint import pprint
        >>> pprint(decomp)
        [RZ(np.float64(3.141...), wires=[0]),
        RY(np.float64(1.570...), wires=[0]),
        RZ(np.float64(0.0), wires=[0]),
        GlobalPhase(np.float64(-1.570...), wires=[])]

        """
        # Decomposes arbitrary single-qubit unitaries as Rot gates (RZ - RY - RZ format),
        # or a single RZ for diagonal matrices.
        shape = qml.math.shape(U)

        is_batched = len(shape) == 3
        shape_without_batch_dim = shape[1:] if is_batched else shape

        if shape_without_batch_dim == (2, 2):
            return qml.ops.one_qubit_decomposition(U, Wires(wires)[0], return_global_phase=True)

        if shape_without_batch_dim == (4, 4):
            # TODO[dwierichs]: Implement decomposition of broadcasted unitary
            if is_batched:
                raise DecompositionUndefinedError(
                    "The decomposition of a two-qubit QubitUnitary does not support broadcasting."
                )
            if sp.sparse.issparse(U):
                raise DecompositionUndefinedError(
                    "The decomposition of a two-qubit sparse QubitUnitary is undefined."
                )

            return qml.ops.two_qubit_decomposition(U, Wires(wires))

        return qml.ops.op_math.decompositions.multi_qubit_decomposition(U, Wires(wires))

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_sparse_matrix(self) -> bool:
        return self._issparse

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self) -> bool:
        return not self._issparse

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self) -> bool:
        # We are unable to decompose sparse matrices larger than 1 qubit.
        return self.has_matrix or len(self.wires) == 1

    def adjoint(self) -> "QubitUnitary":
        if self.has_matrix:
            U = self.matrix()
            return QubitUnitary(qml.math.moveaxis(qml.math.conj(U), -2, -1), wires=self.wires)
        U = self.sparse_matrix()
        adjoint_sp_mat = U.conjugate().transpose()
        # Note: it is necessary to explicitly cast back to csr, or it will become csc.
        return QubitUnitary(adjoint_sp_mat, wires=self.wires)

    def pow(self, z: int | float):
        if self.has_sparse_matrix:
            mat = self.sparse_matrix()
            pow_mat = sp.sparse.linalg.matrix_power(mat, z)
            return [QubitUnitary(pow_mat, wires=self.wires)]

        mat = self.matrix()
        if isinstance(z, int) and qml.math.get_deep_interface(mat) != "tensorflow":
            pow_mat = qml.math.linalg.matrix_power(mat, z)
        elif self.batch_size is not None or qml.math.shape(z) != ():
            return super().pow(z)
        else:
            pow_mat = qml.math.convert_like(fractional_matrix_power(mat, z), mat)
        return [QubitUnitary(pow_mat, wires=self.wires)]

    def _controlled(self, wire):
        return qml.ControlledQubitUnitary(*self.parameters, wires=wire + self.wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


add_decomps(
    QubitUnitary,
    zyz_decomp_rule,
    zxz_decomp_rule,
    xzx_decomp_rule,
    xyx_decomp_rule,
    rot_decomp_rule,
    two_qubit_decomp_rule,
    multi_qubit_decomp_rule,
)


def _qubit_unitary_resource(base_class, base_params, **_):
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_qubit_unitary_resource)
def _adjoint_qubit_unitary(U, wires, **_):
    U = (
        U.conjugate().transpose()
        if sp.sparse.issparse(U)
        else qml.math.moveaxis(qml.math.conj(U), -2, -1)
    )
    QubitUnitary(U, wires=wires)


add_decomps("Adjoint(QubitUnitary)", _adjoint_qubit_unitary)


def _matrix_pow(U, z):
    if sp.sparse.issparse(U):
        return sp.sparse.linalg.matrix_power(U, z)
    if is_integer(z) and qml.math.get_deep_interface(U) != "tensorflow":
        return qml.math.linalg.matrix_power(U, z)
    return qml.math.convert_like(fractional_matrix_power(U, z), U)


@register_resources(_qubit_unitary_resource)
def _pow_qubit_unitary(U, wires, z, **_):
    QubitUnitary(_matrix_pow(U, z), wires=wires)


add_decomps("Pow(QubitUnitary)", _pow_qubit_unitary)


# pylint: disable=unused-argument
def _controlled_qubit_unitary_resource(base_class, base_params, **kwargs):
    return {
        resource_rep(
            qml.ControlledQubitUnitary, num_target_wires=base_params["num_wires"], **kwargs
        ): 1,
    }


@register_resources(_controlled_qubit_unitary_resource)
def _controlled_qubit_unitary(U, wires, control_values, work_wires, work_wire_type, **__):
    qml.ControlledQubitUnitary(
        U,
        wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )


add_decomps("C(QubitUnitary)", _controlled_qubit_unitary)


class DiagonalQubitUnitary(Operation):
    r"""DiagonalQubitUnitary(D, wires)
    Apply an arbitrary diagonal unitary matrix with a dimension that is a power of two.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe: None

    Args:
        D (array[complex]): diagonal of unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    resource_keys = {"num_wires"}

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @staticmethod
    def compute_matrix(D: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
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

        if not qml.math.is_abstract(D) and not qml.math.allclose(
            D * qml.math.conj(D), qml.math.ones_like(D)
        ):
            raise ValueError("Operator must be unitary.")

        # The diagonal is supposed to have one-dimension. If it is broadcasted, it has two
        if qml.math.ndim(D) == 2:
            return qml.math.stack([qml.math.diag(_D) for _D in D])

        return qml.math.diag(D)

    @staticmethod
    def compute_eigvals(D: TensorLike) -> TensorLike:  # pylint: disable=arguments-differ
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
    def compute_decomposition(D: TensorLike, wires: WiresLike) -> list["qml.operation.Operator"]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        ``DiagonalQubitUnitary`` decomposes into :class:`~.DiagonalQubitUnitary`, :class:`~.SelectPauliRot`,
        :class:`~.RZ`, and/or :class:`~.GlobalPhase` depending on the number of wires.

        .. note::

            The parameters of the decomposed operations are cast to the ``complex128`` dtype
            as real dtypes can lead to ``NaN`` values in the decomposition.

        .. seealso:: :meth:`~.DiagonalQubitUnitary.decomposition`.

        Args:
            D (tensor_like): diagonal of the matrix
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        Implements Theorem 7 of `Shende et al. <https://arxiv.org/abs/quant-ph/0406176>`__.
        Decomposing a ``DiagonalQubitUnitary`` on :math:`n` wires (:math:`n>1`) yields a
        uniformly-controlled :math:`R_Z` gate, or :class:`~.SelectPauliRot` gate, as well as a
        ``DiagonalQubitUnitary`` on :math:`n-1` wires. For :math:`n=1` wires, the decomposition
        yields a :class:`~.RZ` gate and a :class:`~.GlobalPhase`.
        Resolving this recursion relationship, one would obtain :math:`n-1` ``SelectPauliRot``
        gates with :math:`n, n-1, \dots, 1` controls each, a single ``RZ`` gate, and
        a ``GlobalPhase``.

        **Example:**

        >>> diag = np.exp(1j * np.array([0.4, 2.1, 0.5, 1.8]))
        >>> qml.DiagonalQubitUnitary.compute_decomposition(diag, wires=[0, 1])
        [DiagonalQubitUnitary(array([0.31532236+0.94898462j, 0.40848744+0.91276394j]), wires=[0]),
        SelectPauliRot(array([1.7, 1.3]), wires=[0, 1])]

        .. details::
            :title: Finding the parameters

            Theorem 7 referenced above only tells us the structure of the circuit, but not the
            parameters for the ``SelectPauliRot`` and ``DiagonalQubitUnitary`` in the decomposition.
            In the following, we will only write out the diagonals of all gates.
            Consider a ``DiagonalQubitUnitary`` on :math:`n` qubits that we want to decompose:

            .. math::

                D(\theta) = (\exp(i\theta_0), \exp(i\theta_1), \dots,
                \exp(i\theta_{N-2}), \exp(i\theta_{N-1})).

            Here, :math:`N=2^n` is the Hilbert space dimension for :math:`n` qubits, which is
            the same as the number of parameters in :math:`D`.

            A ``SelectPauliRot`` gate using ``RZ`` rotations, or multiplexed ``RZ`` rotation, using
            the first :math:`n-1` qubits as controls and the last qubit as target, takes the form

            .. math::

                UCR_Z(\phi) = (\exp(-\frac{i}{2}\phi_0), \exp(\frac{i}{2}\phi_0), \dots,
                \exp(-\frac{i}{2}\phi_{N/2-1}), \exp(\frac{i}{2}\phi_{N/2-1})),

            i.e., it moves the phase of neighbouring pairs of computational basis states by
            the same amount, but in opposite direction. There are :math:`N/2` parameters
            in this gate.
            Similarly, a ``DiagonalQubitUnitary`` acting on the first :math:`n-1` qubits only (the
            ones that were controls for ``SelectPauliRot``) takes the form

            .. math::

                D'(\theta') = (\exp(i\theta'_0), \exp(i\theta'_0), \dots,
                \exp(i\theta'_{N/2-1}), \exp(i\theta'_{N/2-1})).

            That is, :math:`D'` moves the phase of neighbouring pairs of basis states by the same
            amount and in the same direction. It, too, has :math:`N/2` parameters.
            Now, we see that we can compute the rotation angles, or phases, :math:`\phi` and
            :math:`\theta'` quite easily from the original :math:`\theta`:

            .. math::

                (\exp(i\theta_{2i}), \exp(i\theta_{2i+1})) &=
                (\exp(-\frac{i}{2}\phi_i)\exp(i\theta'_i), \exp(\frac{i}{2}\phi_i)\exp(i\theta'_i))\\
                \Rightarrow \qquad \theta'_i &=\frac{1}{2}(\theta_{2i}+\theta_{2i+1})\\
                \phi_i &=\theta_{2i+1}-\theta_{2i}.

            So the phases for the new gates arise simply as difference and average of the
            odd-indexed and even-indexed phases.

        """
        angles = qml.math.angle(D)
        diff = angles[..., 1::2] - angles[..., ::2]
        mean = (angles[..., ::2] + angles[..., 1::2]) / 2
        if len(wires) == 1:
            return [  # Squeeze away non-broadcasting axis (there is just one angle for RZ/GPhase
                qml.GlobalPhase(-qml.math.squeeze(mean, axis=-1), wires=wires),
                qml.RZ(qml.math.squeeze(diff, axis=-1), wires=wires),
            ]
        return [  # Note that we use the first qubits as control, the reference uses the last qubits
            qml.DiagonalQubitUnitary(np.exp(1j * mean), wires=wires[:-1]),
            qml.SelectPauliRot(diff, control_wires=wires[:-1], target_wire=wires[-1]),
        ]

    def adjoint(self) -> "DiagonalQubitUnitary":
        return DiagonalQubitUnitary(qml.math.conj(self.parameters[0]), wires=self.wires)

    def pow(self, z) -> list["DiagonalQubitUnitary"]:
        cast_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(cast_data**z, wires=self.wires)]

    def _controlled(self, control: WiresLike):
        return DiagonalQubitUnitary(
            qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]),
            wires=control + self.wires,
        )

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


def _diagonal_qu_resource(num_wires):
    if num_wires == 1:
        return {qml.RZ: 1, qml.GlobalPhase: 1}
    return {
        resource_rep(DiagonalQubitUnitary, num_wires=num_wires - 1): 1,
        resource_rep(qml.SelectPauliRot, num_wires=num_wires, rot_axis="Z"): 1,
    }


@register_resources(_diagonal_qu_resource)
def _diagonal_qu_decomp(D, wires):
    angles = qml.math.angle(D)
    diff = angles[..., 1::2] - angles[..., ::2]
    mean = (angles[..., ::2] + angles[..., 1::2]) / 2
    if len(wires) == 1:
        qml.GlobalPhase(-qml.math.squeeze(mean, axis=-1), wires=wires)
        qml.RZ(qml.math.squeeze(diff, axis=-1), wires=wires)
    else:
        qml.DiagonalQubitUnitary(np.exp(1j * mean), wires=wires[:-1])
        qml.SelectPauliRot(diff, control_wires=wires[:-1], target_wire=wires[-1])


add_decomps(DiagonalQubitUnitary, _diagonal_qu_decomp)


def _diagonal_qubit_unitary_resource(base_class, base_params, **_):
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_diagonal_qubit_unitary_resource)
def _adjoint_diagonal_unitary(U, wires, **_):
    U = qml.math.conj(U)
    DiagonalQubitUnitary(U, wires=wires)


add_decomps("Adjoint(DiagonalQubitUnitary)", _adjoint_diagonal_unitary)


@register_resources(_diagonal_qubit_unitary_resource)
def _pow_diagonal_unitary(U, wires, z, **_):
    DiagonalQubitUnitary(qml.math.cast(U, np.complex128) ** z, wires=wires)


add_decomps("Pow(DiagonalQubitUnitary)", _pow_diagonal_unitary)


class BlockEncode(Operation):
    r"""BlockEncode(A, wires)
    Construct a unitary :math:`U(A)` such that an arbitrary matrix :math:`A`
    is encoded in the top-left block.

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
        A (tensor_like): a general :math:`(n \times m)` matrix to be encoded
        wires (Iterable[int, str], Wires): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: if the number of wires doesn't fit the dimensions of the matrix

    **Example**

    We can define a matrix and a block-encoding circuit as follows:

    >>> A = [[0.1,0.2],[0.3,0.4]]
    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.BlockEncode(A, wires=range(2))
    ...     return qml.state()

    We can see that :math:`A` has been block encoded in the matrix of the circuit:

    >>> print(qml.matrix(example_circuit)())
    [[ 0.1         0.2         0.97283788 -0.05988708]
     [ 0.3         0.4        -0.05988708  0.86395228]
     [ 0.94561648 -0.07621992 -0.1        -0.3       ]
     [-0.07621992  0.89117368 -0.2        -0.4       ]]

    We can also block-encode a non-square matrix and check the resulting unitary matrix:

    >>> A = [[0.2, 0, 0.2],[-0.2, 0.2, 0]]
    >>> op = qml.BlockEncode(A, wires=range(3))
    >>> print(np.round(qml.matrix(op), 2))
    [[ 0.2   0.    0.2   0.96  0.02  0.    0.    0.  ]
     [-0.2   0.2   0.    0.02  0.96  0.    0.    0.  ]
     [ 0.96  0.02 -0.02 -0.2   0.2   0.    0.    0.  ]
     [ 0.02  0.98  0.   -0.   -0.2   0.    0.    0.  ]
     [-0.02  0.    0.98 -0.2  -0.    0.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    1.    0.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    1.    0.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    1.  ]]

    .. note::
        If the operator norm of :math:`A`  is greater than 1, we normalize it to ensure
        :math:`U(A)` is unitary. The normalization constant can be
        accessed through :code:`op.hyperparameters["norm"]`.

        Specifically, the norm is computed as the maximum of
        :math:`\| AA^\dagger \|` and
        :math:`\| A^\dagger A \|`.
    """

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (2,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, A: TensorLike, wires: WiresLike, id: str | None = None):
        wires = Wires(wires)
        shape_a = qml.math.shape(A)
        if shape_a == () or all(x == 1 for x in shape_a):
            A = qml.math.reshape(A, [1, 1])
            normalization = qml.math.abs(A)
            subspace = (1, 1, 2 ** len(wires))

        else:
            if len(shape_a) == 1:
                A = qml.math.reshape(A, [1, len(A)])
                shape_a = qml.math.shape(A)

            normalization = qml.math.maximum(
                math.norm(A @ qml.math.transpose(qml.math.conj(A)), ord=pnp.inf),
                math.norm(qml.math.transpose(qml.math.conj(A)) @ A, ord=pnp.inf),
            )
            subspace = (*shape_a, 2 ** len(wires))

        # Clip the normalization to at least 1 (= normalize(A) if norm > 1 else A).
        A = qml.math.array(A) / qml.math.maximum(normalization, qml.math.ones_like(normalization))

        if subspace[2] < (subspace[0] + subspace[1]):
            raise ValueError(
                f"Block encoding a ({subspace[0]} x {subspace[1]}) matrix "
                f"requires a Hilbert space of size at least "
                f"({subspace[0] + subspace[1]} x {subspace[0] + subspace[1]})."
                f" Cannot be embedded in a {len(wires)} qubit system."
            )

        super().__init__(A, wires=wires, id=id)
        self.hyperparameters["norm"] = normalization
        self.hyperparameters["subspace"] = subspace

        self._issparse = sp.sparse.issparse(A)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_sparse_matrix(self) -> bool:
        """bool: Whether the operator has a sparse matrix representation."""
        return self._issparse

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self) -> bool:
        """bool: Whether the operator has a sparse matrix representation."""
        return not self._issparse

    def _flatten(self) -> FlatPytree:
        return self.data, (self.wires, ())

    @staticmethod
    def compute_matrix(*params, **hyperparams):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.BlockEncode.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute


        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[0.1,0.2],[0.3,0.4]])
        >>> A
        array([[0.1, 0.2],
            [0.3, 0.4]])
        >>> qml.BlockEncode.compute_matrix(A, subspace=[2,2,4])
        array([[ 0.1       ,  0.2       ,  0.97283788, -0.05988708],
               [ 0.3       ,  0.4       , -0.05988708,  0.86395228],
               [ 0.94561648, -0.07621992, -0.1       , -0.3       ],
               [-0.07621992,  0.89117368, -0.2       , -0.4       ]])
        """
        A = params[0]
        subspace = hyperparams["subspace"]
        if sp.sparse.issparse(A):
            raise qml.operation.MatrixUndefinedError(
                "The operator was initialized with a sparse matrix. Use sparse_matrix instead."
            )
        return _process_blockencode(A, subspace)

    @staticmethod
    def compute_sparse_matrix(*params, **hyperparams):
        A = params[0]
        subspace = hyperparams["subspace"]
        if sp.sparse.issparse(A):
            return _process_blockencode(A, subspace)
        raise qml.operation.SparseMatrixUndefinedError(
            "The operator is initialized with a dense matrix, use the matrix method instead."
        )

    def adjoint(self) -> "BlockEncode":
        A = self.parameters[0]
        return BlockEncode(qml.math.transpose(qml.math.conj(A)), wires=self.wires)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ):
        return super().label(decimals=decimals, base_label=base_label or "BlockEncode", cache=cache)


def _process_blockencode(A, subspace):
    """
    Process the BlockEncode operation.
    """
    n, m, k = subspace
    shape_a = qml.math.shape(A)

    sqrtm = math.sqrt_matrix_sparse if sp.sparse.issparse(A) else math.sqrt_matrix

    def _stack(lst, h=False, like=None):
        if (
            like == "tensorflow"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            axis = 1 if h else 0
            return qml.math.concat(lst, like=like, axis=axis)
        return qml.math.hstack(lst) if h else qml.math.vstack(lst)

    interface = qml.math.get_interface(A)

    if qml.math.sum(shape_a) <= 2:
        col1 = _stack([A, math.sqrt(1 - A * math.conj(A))], like=interface)
        col2 = _stack([math.sqrt(1 - A * math.conj(A)), -math.conj(A)], like=interface)
        u = _stack([col1, col2], h=True, like=interface)
    else:
        d1, d2 = shape_a
        col1 = _stack(
            [
                A,
                sqrtm(
                    math.cast(math.eye(d2, like=A), A.dtype) - qml.math.transpose(math.conj(A)) @ A
                ),
            ],
            like=interface,
        )
        col2 = _stack(
            [
                sqrtm(math.cast(math.eye(d1, like=A), A.dtype) - A @ math.transpose(math.conj(A))),
                -math.transpose(math.conj(A)),
            ],
            like=interface,
        )
        u = _stack([col1, col2], h=True, like=interface)

    if n + m < k:
        r = k - (n + m)
        col1 = _stack([u, math.zeros((r, n + m), like=A)], like=interface)
        col2 = _stack([math.zeros((n + m, r), like=A), math.eye(r, like=A)], like=interface)
        u = _stack([col1, col2], h=True, like=interface)

    return u
