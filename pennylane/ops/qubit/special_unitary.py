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
# pylint: disable=arguments-differ, import-outside-toplevel
from functools import lru_cache, reduce
from itertools import product

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot

_pauli_matrices = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
)
"""Single-qubit Paulis:    I                 X                   Y                  Z"""

_pauli_letters = "IXYZ"
"""Single-qubit Pauli letters that make up Pauli words."""


@lru_cache
def pauli_basis_matrices(num_wires):
    r"""Compute all elements of the Pauli basis of the Lie algebra :math:`\mathfrak{su}(N)`
    as a single, dense tensor.

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

    >>> pauli_basis_matrices(1)
    array([[[ 0.+0.j,  1.+0.j],
            [ 1.+0.j,  0.+0.j]],
           [[ 0.+0.j, -0.-1.j],
            [ 0.+1.j,  0.+0.j]],
           [[ 1.+0.j,  0.+0.j],
            [ 0.+0.j, -1.+0.j]]])

    >>> pauli_basis_matrices(3).shape
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
def pauli_basis_strings(num_wires):
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

    >>> pauli_basis_strings(1)
    ['X', 'Y', 'Z']
    >>> len(pauli_basis_strings(3))
    63
    """
    return ["".join(letters) for letters in product(_pauli_letters, repeat=num_wires)][1:]


def _pauli_decompose(matrix, num_wires):
    r"""Compute the coefficients of a matrix or a batch of matrices (batch dimension(s) in the
    leading axes) in the Pauli basis.

    Args:
        matrix (tensor_like): Matrix or batch of matrices to decompose into the Pauli basis.
        num_wires (int): Number of wires the matrices act on.

    Returns:
        tensor_like: Coefficients of the input ``matrix`` in the Pauli basis.

    For a matrix :math:`M`, these coefficients are defined via
    :math:`M = \sum_\ell c_\ell P_\ell` and they can be computed using the (Frobenius) inner
    product of :math:`M` with the corresponding Pauli word :math:`P_\ell`:
    :math:`c_\ell = \frac{1}{2^N}\operatorname{Tr}\left\{P_\ell M\right\}` where the prefactor
    is the normalization that makes the standard Pauli basis orthonormal, and :math:`N`
    is the number of qubits.
    That is, the normalization is such that a single Pauli word :math:`P_k` has
    coefficients ``c_\ell = \delta_{k\ell}``.

    Note that this implementation takes :math:`\mathcal{O}(16^N)` operations per input
    matrix but there is a more efficient method taking only :math:`\mathcal{O}(N4^N)`
    operations per matrix.
    """
    # Create the dense Pauli basis tensor (shape (d^2-1, d, d)) with d = 2**num_wires
    basis = pauli_basis_matrices(num_wires)
    # Contract the Pauli basis tensor with the input, which has shape (*batch_dims, d, d)
    # We are interested in the traces of the matrix products, for each Pauli basis. Both
    # contractions (mult and trace) can be executed by providing all axes of size d to
    # ``tensordot``, which gives us a vectorized way to compute the coefficients.
    coefficients = qml.math.tensordot(basis, matrix, axes=[[1, 2], [-1, -2]])
    # Finally, cast to the original data type and renormalize
    return qml.math.cast(coefficients, matrix.dtype) / 2**num_wires


class SpecialUnitary(Operation):
    r"""Gate from the group :math:`SU(N)` with :math:`N=2^n` for :math:`n` qubits.

    We use the following parametrization of a special unitary operator:

    .. math::

        U(\bm{\theta}) &= \exp\{A(\bm{\theta})\}\\
        A(\bm{\theta}) &= \sum_{m=1}^d i \theta_m P_m\\
        P_m &\in \{I, X, Y, Z\}^{\otimes n} \setminus \{I^{\otimes n}\}

    This means, :math:`U(\bm{\theta})` is the exponential of the operator :math:`A(\bm{\theta})`,
    which in turn is a linear combination of Pauli words with coefficients :math:`i\bm{\theta}`
    and satisfies :math:`A(\bm{\theta})^\dagger=-A(\bm{\theta})` (it is *skew-Hermitian*).
    Note that this gate takes an exponential number :math:`d=4^n-1` of parameters.
    See below for more theoretical background and details regarding differentiability.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Number of dimensions per parameter: (1,)
    * Gradient recipe:

    .. math::

        \frac{\partial}{\partial\theta_\ell} f(U(\bm{\theta})) &= \sum_{m=1}^d 2i\omega_{\ell m}
        \frac{\mathrm{d}}{\mathrm{d} t}
        f\left(\exp\left\{-i\frac{t}{2}G_m\right\} U(\bm{\theta})\right)\\
        &= \sum_{m=1}^d i\omega_{\ell m}
        \left[
        f\left(\exp\left\{-i\frac{\pi}{4}G_m\right\} U(\bm{\theta})\right)
        -f\left(\exp\left\{i\frac{\pi}{4}G_m\right\} U(\bm{\theta})\right)
        \right]

    where :math:`f` is an expectation value depending on :math:`U(\bm{\theta})` and the derivative
    of the Pauli rotation gates :math:`\exp\left\{-i\frac{t}{2}G_m\right\}` follows from
    the two-term parameter-shift rule (also see: :class:`~.PauliRot`). For details on
    the gradient recipe, also consider the theoretical background section below.

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\bm{\theta})`.
            See details below for the order of the Pauli words.
        wires (Sequence[int] or int): The wire(s) the operation acts on
        id (str or None): String representing the operation (optional)

    Raises:
        ValueError: If the shape of the input does not match the Lie algebra
            dimension :math:`d=4^n-1` for :math:`n` wires.

    .. note::

        This operation should only be used to parametrize special unitaries that
        can not easily be represented by other operations, as it incurs
        computational cost that scale exponentially with the wires it acts on.

    The parameter ``theta`` refers to all Pauli words (except for the identity) in
    lexicographical order, which looks like the following for one and two qubits:

    >>> qml.ops.qubit.special_unitary.pauli_basis_strings(1) # 4**1-1 = 3 Pauli words
    ['X', 'Y', 'Z']
    >>> qml.ops.qubit.special_unitary.pauli_basis_strings(2) # 4**2-1 = 15 Pauli words
    ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']

    .. seealso::

        For more details on using this operator in applications, see the
        :doc:`SU(N) gate demo <demos/tutorial_here_comes_the_sun>`.

    .. warning::

        This operation only is differentiable when using the JAX, Torch or TensorFlow
        interfaces, even when using hardware-compatible differentiation techniques like
        the parameter-shift rule.


    .. warning::

        This operation supports broadcasting and hardware-compatible differentiation
        techniques like the parameter-shift rule. However, the two features can not be
        used simultaneously.

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

    Note that for specific operations like the ``RX`` rotation gate above, it is
    strongly recommended to use the specialized implementation ``qml.RX`` rather
    than ``PauliRot`` or ``SpecialUnitary``.
    However, ``SpecialUnitary`` gates go beyond such rotations: Multiple Pauli words
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

    The ``SpecialUnitary`` operation also can be differentiated with hardware-compatible
    differentiation techniques if the JAX, Torch or TensorFlow interface is used.
    See the theoretical background section below for details.

    .. details::
        :title: Theoretical background
        :href: theory

        We consider special unitaries parametrized in the following way:

        .. math::

            U(\bm{\theta}) &= \exp\{A(\bm{\theta})\}\\
            A(\bm{\theta}) &= \sum_{m=1}^d i \theta_m G_m\\
            G_m &\in \mathcal{P^{(n)}}=\{I, X, Y, Z\}^{\otimes n} \setminus \{I^{\otimes n}\}

        where :math:`n` is the number of qubits and :math:`\theta_m` are :math:`d=4^n-1`
        real-valued parameters. This parametrization allows us to express any special unitary
        for the given number of qubits.

        Note that this differs from a sequence of all possible Pauli rotation gates because
        for non-commuting Pauli words :math:`G_1, G_2` we have
        :math:`\exp\{i\theta_1G_1\}\exp\{i\theta_2G_2\}\neq \exp\{i(\theta_1G_1+\theta_2G_2)\}`.

        **Differentiation**

        The partial derivatives of :math:`U(\bm{\theta})` above can be expressed as

        .. math::

            \frac{\partial}{\partial \theta_\ell} U(\bm{\theta}) &= U(\bm{\theta})
            \frac{\mathrm{d}}{\mathrm{d}t}\exp\left(t\Omega_\ell(\bm{\theta})\right)\large|_{t=0}\\
            &=U(\bm{\theta})\Omega_\ell(\bm{\theta})

        where :math:`\Omega_\ell(\bm{\theta})` is the *effective generator* belonging to the partial
        derivative :math:`\partial_\ell U(\bm{\theta})` at the parameters :math:`\bm{\theta}`.
        It can be computed via

        .. math::

            \Omega_\ell(\bm{\theta}) = U(\bm{\theta})^\dagger
            \left(\frac{\partial}{\partial \theta_\ell}\mathfrak{Re}[U(\bm{\theta})]
            +i\frac{\partial}{\partial \theta_\ell}\mathfrak{Im}[U(\bm{\theta})]\right)

        where we may compute the derivatives of the real and imaginary parts of :math:`U(\bm{\theta})`
        using auto-differentiation.

        Each effective generator :math:`\Omega_\ell(\bm{\theta})` that reproduces a partial derivative
        can be decomposed in the Pauli basis of :math:`\mathfrak{su}(N)` via

        .. math::

            \Omega_\ell(\bm{\theta}) = \sum_{m=1}^d \omega_{\ell m}(\bm{\theta}) G_m.

        As the Pauli words are orthonormal with respect to the
        `trace, or Frobenius, inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`__
        (rescaled by :math:`2^{-n}`), we can compute the coefficients using this
        inner product (:math:`G_m` is Hermitian, so we skip the adjoint :math:`{}^\dagger`):

        .. math::

            \omega_{\ell m}(\bm{\theta}) = \frac{1}{2^n}\operatorname{tr}
            \left\{G_m \Omega_\ell(\bm{\theta}) \right\}

        The coefficients satisfy :math:`\omega_{\ell m}^\ast=-\omega_{\ell m}` because
        :math:`\Omega_\ell(\bm{\theta})` is skew-Hermitian. Therefore they are purely imaginary.

        Now we turn to the derivative of an expectation value-based function which uses a circuit
        with a ``SpecialUnitary`` operation. Absorbing the remaining circuit into the
        quantum state :math:`\rho` and the observable :math:`H`, this can be written as

        .. math::

            f(U(\bm{\theta})) &= \operatorname{Tr}\left\{H U(\bm{\theta})\rho
            U^\dagger(\bm{\theta})\right\}\\
            \partial_\ell f(U(\bm{\theta})) &= \operatorname{Tr}\left\{H U(\bm{\theta})
            [\Omega_\ell(\bm{\theta}),\rho] U^\dagger(\bm{\theta})\right\}

        Inserting the decomposition for the effective generator from above, we may rewrite this as
        a combination of derivatives of Pauli rotation gates:

        .. math::

            \partial_\ell f(U(\bm{\theta}))
            &= \operatorname{Tr}\left\{H U(\bm{\theta})
            \left[\sum_{m=1}^d \omega_{\ell m}(\bm{\theta}) G_m,\rho\right]
            U^\dagger(\bm{\theta})\right\}\\
            &= \sum_{m=1}^d 2i\omega_{\ell m}(\bm{\theta})
            \frac{\mathrm{d}}{\mathrm{d}t}f\left(R_{G_m}(t)U(\bm{\theta})\right)\bigg|_{t=0}.

        Here we abbreviated a Pauli rotation gate as :math:`R_{G_m}(t) = \exp\{-i\frac{t}{2} G_m\}`.
        As all partial derivatives are combinations of these Pauli rotation derivatives, we may
        write the gradient of :math:`f` as

        .. math::

            \nabla f(U(\bm{\theta})) = \overline{\omega}(\bm{\theta}) \cdot \bm{\mathrm{d}f}

        where we defined the matrix :math:`\overline{\omega_{\ell m}}=2i\omega_{\ell m}` and
        the vector of Pauli rotation derivatives
        :math:`\mathrm{d}f_m=\frac{\mathrm{d}}{\mathrm{d}t}
        f\left(R_{G_m}(t)U(\bm{\theta})\right)\bigg|_{t=0}`.
        These derivatives can be computed with the
        standard parameter-shift rule because Pauli words satisfy the condition :math:`G_m^2=1`
        (see e.g. `Mitarai et al. (2018) <https://arxiv.org/abs/1803.00745>`_).
        Provided that we can implement the ``SpecialUnitary`` gate
        itself, this allows us to compute :math:`\bm{\mathrm{d}f}` in a hardware-compatible
        manner using :math:`2d` quantum circuits.

        Note that for ``SpecialUnitary`` we frequently handle objects that have one or
        multiple dimensions of exponentially large size :math:`d=4^n-1`, and that the number
        of quantum circuits for the differentiation scales accordingly. This strongly affects
        the number of qubits to which we can apply a ``SpecialUnitary`` gate in practice.

    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, theta, wires, id=None):
        num_wires = 1 if isinstance(wires, int) else len(wires)
        self.hyperparameters["num_wires"] = num_wires
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

        super().__init__(theta, wires=wires, id=id)

    def _flatten(self):
        return self.data, (self.wires, ())

    @staticmethod
    def compute_matrix(theta, num_wires):
        r"""Representation of the operator as a canonical matrix in the computational basis
        (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SpecialUnitary.matrix`

        Args:
            theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`.
            num_wires (int): The number of wires the matrix acts on.

        Returns:
            tensor_like: canonical matrix of the special unitary corresponding to ``theta``. It
                has the shape ``(2**num_wires, 2**num_wires)``.

        Compute the matrix of an element in SU(N), given by the Pauli basis coordinated
        of the associated Lie algebra element.
        The :math:`4^n-1` Pauli basis elements of the Lie algebra :math:`\mathfrak{su}(N)`
        for :math:`n` qubits are
        :math:`P_m\in\{I, X, Y, Z\}^{\otimes n}\setminus\{I^{\otimes n}\}`, and the special
        unitary matrix is computed as

        .. math::

            U(\theta) = \exp(i\sum_{m=1}^d \theta_m P_m)

        See the main class documentation above for the ordering of Pauli words.

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
        interface = qml.math.get_interface(theta)
        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
        if num_wires > 5:
            matrices = product(_pauli_matrices, repeat=num_wires)
            # Drop the identity from the generator of matrices
            _ = next(matrices)
            A = sum(t * reduce(np.kron, letters) for t, letters in zip(theta, matrices))
        else:
            A = qml.math.tensordot(theta, pauli_basis_matrices(num_wires), axes=[[-1], [0]])
        if interface == "jax" and qml.math.ndim(theta) > 1:
            # jax.numpy.expm does not support broadcasting
            return qml.math.stack([qml.math.expm(1j * _A) for _A in A])
        return qml.math.expm(1j * A)

    def get_one_parameter_generators(self, interface=None):
        r"""Compute the generators of one-parameter groups that reproduce
        the partial derivatives of a special unitary gate.

        Args:
            interface (str): The auto-differentiation framework to be used for the
                computation. Has to be one of ``["jax", "tensorflow", "tf", "torch"]``.

        Raises:
            NotImplementedError: If the chosen interface is ``"autograd"``. Autograd
                does not support differentiation of ``linalg.expm``.
            ValueError: If the chosen interface is not supported.

        Returns:
            tensor_like: The generators for one-parameter groups that reproduce the
            partial derivatives of the special unitary gate.
            There are :math:`d=4^n-1` generators for :math:`n` qubits, so that the
            output shape is ``(4**num_wires-1, 2**num_wires, 2**num_wires)``.

        Consider a special unitary gate parametrized in the following way:

        .. math::

            U(\theta) &= \exp\{A(\theta)\}\\
            A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
            P_m &\in \{I, X, Y, Z\}^{\otimes n} \setminus \{I^{\otimes n}\}

        Then the partial derivatives of the gate can be shown to be given by

        .. math::

            \frac{\partial}{\partial \theta_\ell} U(\theta) = U(\theta)
            \frac{\mathrm{d}}{\mathrm{d}t}\exp\left(t\Omega_\ell(\theta)\right)\large|_{t=0}

        where :math:`\Omega_\ell(\theta)` is the one-parameter generator belonging to the partial
        derivative :math:`\partial_\ell U(\theta)` at the parameters :math:`\theta`.
        It can be computed via

        .. math::

            \Omega_\ell(\theta) = U(\theta)^\dagger
            \left(\frac{\partial}{\partial \theta_\ell}\mathfrak{Re}[U(\theta)]
            +i\frac{\partial}{\partial \theta_\ell}\mathfrak{Im}[U(\theta)]\right)

        where we may compute the derivatives :math:`\frac{\partial}{\partial \theta_\ell} U(\theta)` using auto-differentiation.

        .. warning::

            An auto-differentiation framework is required for this function.
            The matrix exponential is not differentiable in Autograd. Therefore this function
            only supports JAX, Torch and TensorFlow.

        """
        theta = self.data[0]
        if len(qml.math.shape(theta)) > 1:
            raise ValueError("Broadcasting is not supported.")

        num_wires = self.hyperparameters["num_wires"]

        def split_matrix(theta):
            """Compute the real and imaginary parts of the special unitary matrix."""
            mat = self.compute_matrix(theta, num_wires)
            return qml.math.real(mat), qml.math.imag(mat)

        if interface == "jax":
            import jax

            theta = qml.math.cast_like(theta, 1j)
            # These lines compute the Jacobian of compute_matrix every time -> to be optimized
            jac = jax.jacobian(self.compute_matrix, argnums=0, holomorphic=True)(theta, num_wires)

        elif interface == "torch":
            import torch

            rjac, ijac = torch.autograd.functional.jacobian(split_matrix, theta)
            jac = rjac + 1j * ijac

        elif interface in ("tensorflow", "tf"):
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                mats = qml.math.stack(split_matrix(theta))

            rjac, ijac = tape.jacobian(mats, theta)
            jac = qml.math.cast_like(rjac, 1j) + 1j * qml.math.cast_like(ijac, 1j)

        elif interface == "autograd":
            # TODO check whether we can add support for Autograd using eigenvalue decomposition
            raise NotImplementedError(
                "The matrix exponential expm is not differentiable in Autograd."
            )

        else:
            raise ValueError(f"The interface {interface} is not supported.")

        # Compute the Omegas from the Jacobian. The adjoint of U(theta) is realized via -theta
        U_dagger = self.compute_matrix(-qml.math.detach(theta), num_wires)
        # After contracting, move the parameter derivative axis to the first position
        return qml.math.transpose(qml.math.tensordot(U_dagger, jac, axes=[[1], [0]]), [2, 0, 1])

    def get_one_parameter_coeffs(self, interface):
        r"""Compute the Pauli basis coefficients of the generators of one-parameter groups
        that reproduce the partial derivatives of a special unitary gate.

        Args:
            interface (str): The auto-differentiation framework to be used for the
                computation.

        Returns:
            tensor_like: The Pauli basis coefficients of the effective generators
            that reproduce the partial derivatives of the special unitary gate
            defined by ``theta``. There are :math:`d=4^n-1` generators for
            :math:`n` qubits and :math:`d` Pauli coefficients per generator, so that the
            output shape is ``(4**num_wires-1, 4**num_wires-1)``.

        Given a generator :math:`\Omega` of a one-parameter group that
        reproduces a partial derivative of a special unitary gate, it can be decomposed in
        the Pauli basis of :math:`\mathfrak{su}(N)` via

        .. math::

            \Omega = \sum_{m=1}^d \omega_m P_m

        where :math:`d=4^n-1` is the size of the basis for :math:`n` qubits and :math:`P_m` are the
        Pauli words making up the basis. As the Pauli words are orthonormal with respect to the
        `trace or Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`__
        (rescaled by :math:`2^n`), we can compute the coefficients using this
        inner product (:math:`P_m` is Hermitian, so we skip the adjoint :math:`{}^\dagger`):

        .. math::

            \omega_m = \frac{1}{2^n}\operatorname{tr}\left[P_m \Omega \right]

        The coefficients satisfy :math:`\omega_m^\ast=-\omega_m` because :math:`\Omega` is
        skew-Hermitian. Therefore they are purely imaginary.

        .. warning::

            An auto-differentiation framework is required by this function.
            The matrix exponential is not differentiable in Autograd. Therefore this function
            only supports JAX, Torch and Tensorflow.

        .. seealso:: :meth:`~.SpecialUnitary.get_one_parameter_generators`

        """
        num_wires = self.hyperparameters["num_wires"]
        generators = self.get_one_parameter_generators(interface)
        return _pauli_decompose(generators, num_wires)

    def decomposition(self):
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        This ``Operation`` is decomposed into the corresponding ``QubitUnitary``.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> theta = np.array([0.5, 0.1, -0.3])
        >>> qml.SpecialUnitary(theta, wires=[0]).decomposition()
        [QubitUnitary(array([[ 0.83004499-0.28280371j,  0.0942679 +0.47133952j],
            [-0.0942679 +0.47133952j,  0.83004499+0.28280371j]]), wires=[0])]
        """
        theta = self.data[0]
        if qml.math.requires_grad(theta):
            interface = qml.math.get_interface(theta)
            # Get all Pauli words for the basis of the Lie algebra for this gate
            words = pauli_basis_strings(self.hyperparameters["num_wires"])

            # Compute the linear map that transforms between the Pauli basis and effective generators
            # Consider the mathematical derivation for the prefactor 2j
            omega = qml.math.real(2j * self.get_one_parameter_coeffs(interface))

            # Create zero parameters for each Pauli rotation gate that take over the trace of theta
            detached_theta = qml.math.detach(theta)
            zeros = theta - detached_theta
            # Apply the linear map omega to the zeros to create the correct preprocessing Jacobian
            zeros = qml.math.tensordot(omega, zeros, axes=[[1], [0]])

            # Apply Pauli rotations that yield the Pauli basis derivatives
            paulirots = [
                TmpPauliRot(zero, word, wires=self.wires, id="SU(N) byproduct")
                for zero, word in zip(zeros, words)
            ]
            return paulirots + [SpecialUnitary(detached_theta, wires=self.wires)]

        return [qml.QubitUnitary(self.matrix(), wires=self.wires)]

    def adjoint(self):
        return SpecialUnitary(-self.data[0], wires=self.wires)


class TmpPauliRot(PauliRot):
    r"""A custom version of ``PauliRot`` that is inserted with rotation angle zero when
    decomposing ``SpecialUnitary``. The differentiation logic makes use of the gradient
    recipe of ``PauliRot``, but deactivates the matrix property so that a decomposition
    of differentiated tapes is forced. During this decomposition, this private operation
    removes itself if its angle remained at zero.

    For details see :class:`~.PauliRot`.

    .. warning::

        Do not add this operation to the supported operations of any device.
        Wrong results and/or severe performance degradations may result.
    """

    # Deactivate the matrix property of qml.PauliRot in order to force decomposition
    has_matrix = False

    @staticmethod
    def compute_decomposition(theta, wires, pauli_word):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.TmpPauliRot.decomposition`.

        Args:
            theta (float): rotation angle :math:`\theta`
            wires (Iterable, Wires): the wires the operation acts on
            pauli_word (string): the Pauli word defining the rotation

        Returns:
            list[Operator]: decomposition into an empty list of operations for
            vanishing ``theta``, or into a list containing a single :class:`~.PauliRot`
            for non-zero ``theta``.

        .. note::

            This operation is used in a differentiation pipeline of :class:`~.SpecialUnitary`
            and most likely should not be created manually by users.
        """
        if qml.math.isclose(theta, theta * 0) and not qml.math.requires_grad(theta):
            return []
        return [PauliRot(theta, pauli_word, wires)]

    def __repr__(self):
        return f"TmpPauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})"
