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
from functools import lru_cache, reduce
from itertools import product
import warnings
import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, DecompositionUndefinedError, Operation
from pennylane.wires import Wires

_paulis = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
"""Single-qubit Paulis:    I                 X                   Y                  Z"""

_pauli_letters = ["I", "X", "Y", "Z"]
"""Single-qubit Pauli letters that make up Pauli words."""


@lru_cache
def pauli_basis(num_wires):
    r"""Compute all elements of the Pauli basis of the Lie algebra :math:`\mathfrak{su}(n)`.

    Args:
        num_wires (int): The number of wires on which the associated Pauli group acts.

    Returns:
        ndarray: All Pauli basis elements of :math:`\mathfrak{su}(n)`.

    The basis has :math:`d=4^n-1` elements for :math:`n` qubits, yielding an output tensor
    with shape ``(4**num_wires-1, 2**num_wires, 2**num_wires)`` and :math:`16^n-4^n` entries.
    The identity Pauli word :math:`I^{\otimes n}` does not belong to :math:`\mathfrak{su}(n)`
    and therefore is not included.

    The basis elements are ordered (choose the description that suits you most)

      - lexicographically.

      - such that the term acting on the last qubit changes fastest, the one acting on the first
        qubit changes slowest when iterating through the output.

      - such that the basis index, written in base :math:`4`, contains the indices for the list
        ``["I", "X", "Y", "Z"]``, in the order of the qubits

      - such that for three qubits, the first basis elements are the Pauli words
        ``"IIX", ""IIY", "IIZ", "IXI", "IXX", "IXY", "IXZ", "IYI"...``

    .. admonition::

        Note that this method internally handles a complex-valued tensor of size
        ``(4**num_wires, 2**num_wires, 2**num_wires)``, which requires at least
        ``4 ** (2 * num_wires - 13)`` GB of memory (at default precision).

    **Example**

    >>> pauli_basis(1)
    array([[[ 0.+0.j,  1.+0.j],
            [ 1.+0.j,  0.+0.j]],
           [[ 0.+0.j, -0.-1.j],
            [ 0.+1.j,  0.+0.j]],
           [[ 1.+0.j,  0.+0.j],
            [ 0.+0.j, -1.+0.j]]])

    >>> pauli_basis(3).shape
    (63, 8, 8)
    """
    return reduce(np.kron, (_paulis for _ in range(num_wires)))[1:]


@lru_cache
def pauli_words(num_wires):
    r"""Compute all :math:`n`-qubit Pauli words except ``"I"*num_wires``,
    corresponding to the Pauli basis of the Lie algebra :math:`\mathfrak{su}(n)`.

    Args:
        num_wires (int): The number of wires, or number of letters per word.

    Returns:
        list[str]: All Pauli words on ``num_wires`` qubits, except from the identity.

    There are :math:`d=4^n-1` Pauli words that are not the identity. They are ordered
    (choose the description that suits you most)

      - lexicographically.

      - such that the term acting on the last qubit changes fastest, the one acting on the first
        qubit changes slowest when iterating through the output.

      - such that the basis index, written in base :math:`4`, contains the indices for the list
        ``["I", "X", "Y", "Z"]``, in the order of the qubits

      - such that for three qubits, the first Pauli words are
        ``"IIX", ""IIY", "IIZ", "IXI", "IXX", "IXY", "IXZ", "IYI"...``

    **Example**

    >>> pauli_words(1)
    ['X', 'Y', 'Z']
    >>> len(pauli_words(3))
    63
    """
    return ["".join(letters) for letters in list(product(_pauli_letters, repeat=num_wires))[1:]]


def special_unitary_matrix(theta, num_wires):
    r"""Compute the matrix of an element in SU(N), given by the Pauli basis coordinated
    of the associated Lie algebra element.

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`.
        num_wires (int): The number of wires the matrix acts on.

    Returns:
        tensor_like: matrix of the special unitary corresponding to ``theta``. It
            has the shape ``(2**num_wires, 2**num_wires)``.

    The :math:`4^n-1` Pauli basis elements of the Lie algebra :math:`\mathfrak{su}(n)`
    for :math:`n` qubits are
    :math:`P_m\in\{I, X, Y, Z\}^{\otimes n}\setminus\{I^{\otimes n}\}`, and the special
    unitary matrix is computed as

    .. math::

        U(\theta) = \exp(i\sum_{m=1}^d \theta_m P_m)

    See :func:`~.ops.qubit.matrix_ops.pauli_basis` for the ordering of Pauli words.

    ..admonition::

        Note that this method internally handles a complex-valued tensor of size
        ``(4**num_wires, 2**num_wires, 2**num_wires)``, which requires at least
        ``4 ** (2 * num_wires - 13)`` GB of memory (at default precision).
    """
    A = qml.math.tensordot(theta, pauli_basis(num_wires), axes=[[-1], [0]])
    return qml.math.expm(1j * A)


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
        if isinstance(self.data[0], list):
            if isinstance(self.data[0][0], list):
                # Support broadcasted list
                new_data = [[(el + 0j) ** z for el in x] for x in self.data[0]]
            else:
                new_data = [(x + 0.0j) ** z for x in self.data[0]]
            return [DiagonalQubitUnitary(new_data, wires=self.wires)]
        casted_data = qml.math.cast(self.data[0], np.complex128)
        return [DiagonalQubitUnitary(casted_data**z, wires=self.wires)]

    def _controlled(self, control):
        return DiagonalQubitUnitary(
            qml.math.hstack([np.ones_like(self.parameters[0]), self.parameters[0]]),
            wires=control + self.wires,
        )

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "U", cache=cache)


class SpecialUnitary(Operation):
    r"""Gate from the group :math:`SU(N)` with :math:`N=2^n` for :math:`n` qubits.

    .. math::

        U(\theta) &= e^{A(\theta)}\\
        A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
        P_m &\in {I, X, Y, Z}^{\otimes n} \setminus \{I^{\otimes n}\}

    This means, :math:`U(\theta)` is the exponential of the skew-Hermitian operator
    :math:`A(\theta)`, which in turn is parametrized as a linear combination of
    Pauli words with coefficients :math:`\theta`.
    Note that this gate takes an exponential number :math:`d=4^n-1` of parameters.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe:

    .. math::

        \frac{\partial}{\partial\theta_\ell} f(U(\theta)) &= -i \sum_{m=1}^d \omega_{\ell m} \frac{\mathrm{d}}{\mathrm{d} x} f(e^{ixP_m} U(\theta))

      where :math:`f` is an expectation value depending on :math:`U(\theta)` and the derivative
      of the Pauli rotation gates can be computed with the two-term parameter-shift rule
      (also see: :class:`~.PauliRot`).

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`.
            See details below for the order of the Pauli words.
        wires (Sequence[int] or int): The wire(s) the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: If the shape of the input parameters does not match the Lie algebra
            dimension :math:`d=4*n-1` for :math:`n` qubits.

    **Examples**

    Simple examples of this operation are single-qubit Pauli rotation gates:

    >>> x = 0.412
    >>> theta = x * np.array([1, 0, 0]) # The first entry belongs to the Pauli word "X"
    >>> su = qml.SpecialUnitary(theta, wires=0)
    >>> prot = qml.PauliRot(-2 * x, "X", wires=0) # PauliRot introduces a prefactor -0.5
    >>> rx = qml.RX(-2 * x, 0) # RX introduces a prefactor -0.5
    >>> qml.math.allclose(su.matrix(), prot.matrix())
    True
    >>> qml.math.allclose(su.matrix(), rx.matrix())
    True

    More interestingly, multiple Pauli words can be activated simultaneously, giving
    access to more complex operations. For two qubits, this may look like this:

    >>> wires = [0, 1]
    >>> theta = 0.3 * np.array([0, 1, 2, 0, -1, 1, 0, 0, 0, 1, 1, 1, 0, 0, -1])
    >>> len(theta) == 4 ** len(wires) - 1 # theta contains one parameter per Pauli word
    True
    >>> su = qml.SpecialUnitary(theta, wires=wires)
    >>> su.matrix()
    array([[ 0.56397118+0.52139241j,  0.30652227+0.02438052j,
             0.13555302+0.22630716j,  0.0689876 -0.49110826j],
           [-0.15454843+0.00998377j,  0.88294943+0.01496327j,
            -0.25396275-0.10785888j, -0.26041566+0.22857073j],
           [-0.2876174 -0.2443733j ,  0.25423439+0.05896445j,
             0.71621665+0.50686226j,  0.1380692 +0.02252197j],
           [-0.34495668-0.35307844j,  0.10817019-0.21404059j,
            -0.29040522+0.00830631j,  0.15015337-0.76933485j]])
    """
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, theta, wires, do_queue=True, id=None):
        num_wires = 1 if isinstance(wires, int) else len(wires)
        self.hyperparameters["num_wires"] = num_wires
        theta_shape = qml.math.shape(theta)
        expected_dim = 4**num_wires - 1

        if len(theta_shape) not in {1, 2}:
            raise ValueError(
                "Expected the parameters to have one or two dimensions without or with "
                f"broadcasting, respectively. The parameters have the shape {theta_shape}"
            )

        if theta_shape[-1] != expected_dim:
            raise ValueError(
                f"Expected the parameters to have shape ({expected_dim},) or (batch_size, "
                f"{expected_dim}). The parameters have the shape {theta_shape}"
            )

        super().__init__(theta, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(theta, num_wires):
        r"""Representation of the operator as a canonical matrix in the computational basis
        (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SpecialUnitary.matrix`

        Args:
            theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`
            num_wires (int): The number of wires

        Returns:
            tensor_like: canonical matrix

        ..admonition::

            Note that this method internally handles a complex-valued tensor of size
            ``(4**num_wires, 2**num_wires, 2**num_wires)``, which requires at least
            ``4 ** (2 * num_wires - 13)`` GB of memory (at default precision).

        **Example**

        >>> theta = np.array([0.5, 0.1, -0.3])
        >>> qml.SpecialUnitary.compute_matrix(theta, num_wires=1)
        array([[ 0.83004499-0.28280371j,  0.0942679 +0.47133952j],
               [-0.0942679 +0.47133952j,  0.83004499+0.28280371j]])
        """
        return special_unitary_matrix(theta, num_wires)

    @staticmethod
    def compute_decomposition(theta, wires, num_wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        This ``Operation`` is decomposed into the corresponding ``QubitUnitary``.

        .. seealso:: :meth:`~.QubitUnitary.decomposition`.

        Args:
            theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`
            wires (Iterable[Any] or Wires): the wire(s) the operation acts on
            num_wires (int): The number of wires

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> theta = np.array([0.5, 0.1, -0.3])
        >>> qml.SpecialUnitary.compute_decomposition(theta, 0, num_wires=1)
        [QubitUnitary(array([[ 0.83004499-0.28280371j,  0.0942679 +0.47133952j],
            [-0.0942679 +0.47133952j,  0.83004499+0.28280371j]]), wires=[0])]
        """
        return [QubitUnitary(special_unitary_matrix(theta, num_wires), wires=wires)]

    def adjoint(self):
        return SpecialUnitary(-self.data[0], wires=self.wires)
