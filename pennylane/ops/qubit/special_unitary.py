# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule contains the operation SpecialUnitary and
its utility functions.
"""
# pylint: disable=arguments-differ
from functools import lru_cache, reduce
from itertools import product

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation

_pauli_matrices = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
)
"""Single-qubit Paulis:    I                 X                   Y                  Z"""

_pauli_letters = "IXYZ"
"""Single-qubit Pauli letters that make up Pauli words."""


@lru_cache
def pauli_basis(num_wires):
    r"""Compute all elements of the Pauli basis of the Lie algebra :math:`\mathfrak{su}(N)`.

    Args:
        num_wires (int): The number of wires on which the associated Pauli group acts.

    Returns:
        ndarray: All Pauli basis elements of :math:`\mathfrak{su}(N)`.

    The basis has :math:`d=4^n-1` elements for :math:`n` qubits, yielding an output tensor
    with shape ``(4**num_wires-1, 2**num_wires, 2**num_wires)`` and :math:`16^n-4^n` entries.
    The identity Pauli word :math:`I^{\otimes n}` does not belong to :math:`\mathfrak{su}(N)`
    and therefore is not included.

    The basis elements are ordered (choose the description that suits you most)

      - lexicographically.

      - such that the term acting on the last qubit changes fastest, the one acting on the first
        qubit changes slowest when iterating through the output.

      - such that the basis index, written in base :math:`4`, contains the indices for the list
        ``["I", "X", "Y", "Z"]``, in the order of the qubits

      - such that for three qubits, the first basis elements are the Pauli words
        ``"IIX", ""IIY", "IIZ", "IXI", "IXX", "IXY", "IXZ", "IYI"...``

    .. note::

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
    if num_wires < 1:
        raise ValueError(f"Require at least one wire, got {num_wires}.")
    if num_wires > 7:
        raise ValueError(
            f"Creating the Pauli basis tensor for more than 7 wires (got {num_wires}) is deactivated."
        )
    return reduce(np.kron, (_pauli_matrices for _ in range(num_wires)))[1:]


@lru_cache
def pauli_words(num_wires):
    r"""Compute all :math:`n`-qubit Pauli words except ``"I"*num_wires``,
    corresponding to the Pauli basis of the Lie algebra :math:`\mathfrak{su}(N)`.

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


letter_counter = {"I": "0", "X": "1", "Y": "2", "Z": "3"}


def _params_to_pauli_basis(params, words):
    """Create a new zeros tensor and set the entries corresponding
    to the given Pauli words in the Pauli basis to the values given in ``params``.
    The Pauli basis is ordered lexicographically to determine the indices."""
    if isinstance(words, str):
        words = [words]
    num_words = len(words)
    par_shape = qml.math.shape(params)
    if not 0 != num_words == len(set(words)) == par_shape[-1]:
        raise ValueError(
            f"Expected as many unique Pauli words as parameters, but at least one. Got {par_shape[-1]} parameters and words {words}"
        )
    n = len(words[0])
    if "I" * n in words:
        raise ValueError(
            "The identity is not a valid Pauli basis element for the Lie algebra su(n)."
        )
    ids = tuple(int("".join(letter_counter[letter] for letter in w), base=4) - 1 for w in words)

    output_shape = (
        4**n - 1
        if len(par_shape) == 1
        else (
            4**n - 1,
            par_shape[0],
        )
    )
    zeros = qml.math.cast_like(qml.math.convert_like(np.zeros(output_shape), params), params)
    return qml.math.T(qml.math.scatter_element_add(zeros, [ids], qml.math.T(params)))


def special_unitary_matrix(theta, num_wires):
    r"""Compute the matrix of an element in SU(N), given by the Pauli basis coordinates
    of the associated Lie algebra element.

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`.
        num_wires (int): The number of wires the matrix acts on.

    Returns:
        tensor_like: matrix of the special unitary corresponding to ``theta``. It
            has the shape ``(2**num_wires, 2**num_wires)``.

    The :math:`4^n-1` Pauli basis elements of the Lie algebra :math:`\mathfrak{su}(N)`
    for :math:`n` qubits are
    :math:`P_m\in\{I, X, Y, Z\}^{\otimes n}\setminus\{I^{\otimes n}\}`, and the special
    unitary matrix is computed as

    .. math::

        U(\theta) = \exp(i\sum_{m=1}^d \theta_m P_m)

    See :func:`~.ops.qubit.matrix_ops.pauli_words` for the ordering of Pauli words.

    .. note::

        Note that this method internally handles a complex-valued tensor of size
        ``(4**num_wires, 2**num_wires, 2**num_wires)``, which requires at least
        ``4 ** (2 * num_wires - 13)`` GB of memory (at default precision).
    """
    if num_wires > 5:
        theta = qml.math.hstack([[0], theta])
        A = sum(
            t * reduce(np.kron, letters)
            for t, letters in zip(theta, product(_pauli_matrices, repeat=num_wires))
        )
    else:
        A = qml.math.tensordot(theta, pauli_basis(num_wires), axes=[[-1], [0]])
    return qml.math.expm(1j * A)


class SpecialUnitary(Operation):
    r"""Gate from the group :math:`SU(N)` with :math:`N=2^n` for :math:`n` qubits.

    We use the following parametrization of a special unitary operator:

    .. math::

        U(\theta) &= e^{A(\theta)}\\
        A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
        P_m &\in {I, X, Y, Z}^{\otimes n} \setminus \{I^{\otimes n}\}

    This means, :math:`U(\theta)` is the exponential of operator :math:`A(\theta)`,
    which in turn is a linear combination of Pauli words with coefficients :math:`i\theta`
    and satisfies :math:`A(\theta)^\dagger=-A(\theta)` (it is *skew-Hermitian*).
    Note that this gate takes an exponential number :math:`d=4^n-1` of parameters.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe

    .. math::

        \frac{\partial}{\partial\theta_\ell} f(U(\theta)) &= -i \sum_{m=1}^d \omega_{\ell m}
        \frac{\mathrm{d}}{\mathrm{d} x} f(e^{ixP_m} U(\theta))

      where :math:`f` is an expectation value depending on :math:`U(\theta)` and the derivative
      of the Pauli rotation gates :math:`e^{ixP_m}` can be computed with the two-term
      parameter-shift rule (also see: :class:`~.PauliRot`).

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`.
            See details below for the order of the Pauli words.
        wires (Sequence[int] or int): The wire(s) the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: If the shape of the input parameter does not match the Lie algebra
            dimension :math:`d=4*n-1` for :math:`n` wires.

    The parameter ``theta`` refers to all Pauli words (except for the identity) in
    lexicographical order, which looks like the following for one and two qubits:

    >>> qml.ops.qubit.matrix_ops.pauli_words(1) # 4**1-1 = 3 Pauli words
    ['X', 'Y', 'Z']
    >>> qml.ops.qubit.matrix_ops.pauli_words(2) # 4**2-1 = 15 Pauli words
    ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

    **Examples**

    Simple examples of this operation are single-qubit Pauli rotation gates, which can be
    created by setting all but one parameter :math:`\theta_m` to zero:

    >>> x = 0.412
    >>> theta = x * np.array([1, 0, 0]) # The first entry belongs to the Pauli word "X"
    >>> su = qml.SpecialUnitary(theta, wires=0)
    >>> prot = qml.PauliRot(-2 * x, "X", wires=0) # PauliRot introduces a prefactor -0.5
    >>> rx = qml.RX(-2 * x, 0) # RX introduces a prefactor -0.5
    >>> qml.math.allclose(su.matrix(), prot.matrix())
    True
    >>> qml.math.allclose(su.matrix(), rx.matrix())
    True

    However, ``SpecialUnitary`` gates go beyong such rotations: Multiple Pauli words
    can be activated simultaneously, giving access to more complex operations.
    For two qubits, this could look like this:

    >>> wires = [0, 1]
    # Activating the Pauli words ["IY", "IZ", "XX", "XY", "YY", "YZ", "ZY", "ZZ"]
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

    Alternatively, it is possible to pass the parameters for specific Pauli words and
    indicate the selection and order of words with the optional argument ``words``:

    >>> x = [0.2, 0.4]
    >>> words = ["IX", "YY"] # The first parameter will be used for IX, the second for YY
    >>> qml.SpecialUnitary(x, wires=[0, 1], words=words)
    SpecialUnitary(array([0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.4, 0. , 0. , 0. ,
           0. , 0. ]), wires=[0, 1])

    """
    # pylint: disable=too-many-arguments
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, theta, wires, words=None, do_queue=True, id=None):
        num_wires = 1 if isinstance(wires, int) else len(wires)
        self.hyperparameters["num_wires"] = num_wires
        if words is not None:
            theta = _params_to_pauli_basis(theta, words)
        theta_shape = qml.math.shape(theta)
        expected_dim = 4**num_wires - 1

        if len(theta_shape) not in {1, 2}:
            raise ValueError(
                "Expected the parameters to have one or two dimensions without or with "
                f"broadcasting, respectively. The parameters have shape {theta_shape}"
            )

        if theta_shape[-1] != expected_dim:
            raise ValueError(
                f"Expected the parameters to have shape ({expected_dim},) or (batch_size, "
                f"{expected_dim}). The parameters have shape {theta_shape}"
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

        .. note::

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

        .. seealso:: :meth:`~.SpecialUnitary.decomposition`.

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
        return [qml.QubitUnitary(special_unitary_matrix(theta, num_wires), wires=wires)]

    def adjoint(self):
        return SpecialUnitary(-self.data[0], wires=self.wires)
