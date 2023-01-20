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
"""Code for a quantum function transform that insert Pauli rotation
gates for hardware-based differentiation of SpecialUnitary."""
import pennylane as qml
from pennylane.transforms.qfunc_transforms import qfunc_transform
from pennylane.ops.qubit.matrix_ops import special_unitary_matrix, pauli_basis, pauli_words
# pylint: disable=import-outside-toplevel


def _trainable_zeros_like(tensor):
    return tensor - qml.math.convert_like(qml.math.to_numpy(tensor), tensor)


def _trainable_zeros_like(tensor):
    return qml.math.where(tensor == 0, tensor, 0)


def get_one_parameter_generators(theta, num_wires, interface="jax"):
    r"""Compute the generators of one-parameter groups that reproduce
    the partial derivatives of a special unitary gate.

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`
            of the special unitary gate.
        num_wires (int): The number of wires the special unitary gate acts on.
        interface (str): The auto-differentiation framework to be used for the
            computation.

    Returns:
        tensor_like: The generators for one-parameter groups that reproduce the
        partial derivatives of the special unitary gate defined by ``theta``.
        There are :math:`d=4^n-1` generators for :math:`n` qubits, so that the
        output shape is ``(4**num_wires-1, 2**num_wires, 2**num_wires)``.

    Consider a special unitary gate parametrized in the following way:

    .. math::

        U(\theta) &= e^{A(\theta)}\\
        A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
        P_m &\in {I, X, Y, Z}^{\otimes n} \setminus \{I^{\otimes n}\}

    Then the partial derivatives of the gate can be shown to be given by

    .. math::

        \frac{\partial}{\partial_\ell} U(\theta) &= U(\theta)
        \frac{\mathrm{d}}{\mathrm{d}t}\exp\left(t\Omega_\ell(\theta)\right)\large|_{t=0}

    where :math:`\Omega_\ell(\theta)` is the one-parameter generator belonging to the partial
    derivative :math:`\partial_\ell U(\theta)` at the parameters :math:`\theta`.
    It can be computed via

    .. math::

        \Omega_\ell(\theta) &= U(\theta)^\dagger
        \left(\frac{\partial}{\partial \theta_\ell}\mathfrak{Re}[U(\theta)]
        +i\frac{\partial}{\partial \theta_\ell}\mathfrak{Im}[U(\theta)]\right)

    where we may compute the derivatives on the left-hand side using auto-differentiation.

    .. warning::

        An auto-differentiation framework is required by this function.
        The matrix exponential is not differentiable in Autograd. Therefore this function
        only supports JAX, Torch and Tensorflow.

    ..seealso:: `~.SpecialUnitary`
    ..seealso:: `~.transforms.insert_paulirot`

    """

    if interface == "jax":
        import jax

        theta = qml.math.cast_like(theta, 1j)
        jac = jax.jacobian(special_unitary_matrix, argnums=0, holomorphic=True)(theta, num_wires)

    elif interface in ["torch", "autograd"]:
        # TODO check whether we can add support for Autograd using eigenvalue decomposition
        if interface == "autograd":
            raise NotImplementedError(
                "The matrix exponential expm is not differentiable in Autograd."
            )

        def real_matrix(theta):
            return qml.math.real(special_unitary_matrix(theta, num_wires))

        def imag_matrix(theta):
            return qml.math.imag(special_unitary_matrix(theta, num_wires))

        import torch

        # These lines compute the Jacobian of compute_matrix every time -> to be optimized
        real_jac = torch.autograd.functional.jacobian(real_matrix, theta)
        imag_jac = torch.autograd.functional.jacobian(imag_matrix, theta)
        jac = real_jac + 1j * imag_jac

    elif interface in ("tensorflow", "tf"):
        import tensorflow as tf

        with tf.GradientTape(persistent=True) as tape:
            mat = special_unitary_matrix(theta, num_wires)
            rmat = tf.math.real(mat)
            imat = tf.math.imag(mat)

        rjac = tape.jacobian(rmat, theta)
        ijac = tape.jacobian(imat, theta)
        jac = qml.math.cast_like(rjac, 1j) + 1j * qml.math.cast_like(ijac, 1j)

    else:
        raise NotImplementedError(f"The interface {interface} is not supported.")

    # Move parameter axis to first place
    jac = qml.math.transpose(jac, [2, 0, 1])
    # Compute the Omegas from the Jacobian. The adjoint of U(theta) is realized via -theta
    return qml.math.tensordot(jac, special_unitary_matrix(-theta, num_wires), axes=[[1], [1]])


def get_one_parameter_coeffs(theta, num_wires, interface="jax"):
    r"""Compute the Pauli basis coefficients of the generators of one-parameter groups
    that reproduce the partial derivatives of a special unitary gate.

    Args:
        theta (tensor_like): Pauli coordinates of the exponent :math:`A(\theta)`
            of the special unitary gate.
        num_wires (int): The number of wires the special unitary gate acts on.
        interface (str): The auto-differentiation framework to be used for the
            computation.

    Returns:
        tensor_like: The generators for one-parameter groups that reproduce the
        partial derivatives of the special unitary gate defined by ``theta``.
        There are :math:`d=4^n-1` generators for :math:`n` qubits, so that the
        output shape is ``(4**num_wires-1, 2**num_wires, 2**num_wires)``.

    Given a generator :math:`\Omega` of a one-parameter group that reproduces a partial derivative
    of a special unitary gate, it can be decomposed in the Pauli basis of :math:`\mathfrak{su}(N)`
    via

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

    ..seealso:: `~.transforms.insert_paulirot.get_one_parameter_generators`

    """
    basis = pauli_basis(num_wires)
    generators = get_one_parameter_generators(theta, num_wires, interface)
    return qml.math.tensordot(basis, generators, axes=[[1, 2], [2, 1]]) / 2**num_wires


@qfunc_transform
def insert_paulirot(tape):
    r"""Insert Pauli rotations in front of ``qml.SpecialUnitary`` gates to allow for
    their differentiation with parameter shifts.

    Args:
        qfunc (function): A quantum function. If it does not contain any ``qml.SpecialUnitary``
            operations, the transform has no effect.

    Returns:
        function: the transformed quantum function

    For each :class:`~.SpecialUnitary` in the quantum function, a series of
    :class:`~.PauliRot` operations with individual rotation angles zero is inserted, such that
    differentiating the quantum function with a :func:`~.gradients.gradient_transform` will
    yield the result of differentiating the ``SpecialUnitary`` gate itself.

    .. warning::

        If you are using this transform manually, make sure to remove ``qml.PauliRot`` operations
        with rotation angle zero from the circuits produced by the gradient transform.
        Without doing so, the result will be correct, but removing the redundant rotations will
        increase the performance significantly.

    .. warning::

        Note that this transform requires the ``SpecialUnitary`` gates of interest to
        use parameters from an auto-differentiation framework other than Autograd,
        even when the differentiation of the quantum function is performed without automatic
        differentiation later on.

    **Example**

    We will show an example for manually calling this transform. Usually, this will
    happen internally whenever ``SpecialUnitary`` gates need to be composed.
    Consider the following minimalistic circuit:

    .. code-block:: python

        import jax
        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2, shots=100)

        @qml.qnode(dev, interface="jax")
        def circuit(theta):
            qml.SpecialUnitary(theta, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        theta = jax.numpy.linspace(0.3, 0.9, 15)

    In order to differentiate it, we could use auto-differentiation. However, if we want to
    use a sampling-based device like here, or in particular a quantum device, this is not an option.
    Instead, we may apply the ``insert_paulirot`` transform to the quantum function and
    construct a new ``QNode``, which we then can differentiate:

    >>> # jax.jacobian(circuit)(theta) # This would raise an Error
    >>> new_qfunc = qml.transforms.insert_paulirot(circuit.func)
    >>> new_circuit = qml.QNode(new_qfunc, dev, interface="jax")
    >>> jax.jacobian(new_circuit)(theta) # This works!
    Array([-0.68, -0.38, -0.28,  0.7 , -0.96, -0.06, -0.56, -0.02, -0.6 ,
           -0.84, -0.16, -0.52,  0.14, -0.8 , -0.9 ], dtype=float64)

    .. details::
        :title: Mathematical background
        :href: mathematical-background

        A ``SpecialUnitary`` gate :math:`U(\theta)` is given by the parameters of its corresponding
        Lie algebra element :math:`A(\theta)` in the Pauli basis:

        .. math::

            U(\theta) &= \exp(A(\theta))\\
            A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
            P_m &\in {I, X, Y, Z}^{\otimes n} \setminus \{I^{\otimes n}\}

        The partial derivative of the gate can be written as

        .. math::

            \frac{\partial}{\partial_\ell} U(\theta) &= U(\theta)
            \frac{\mathrm{d}}{\mathrm{d}t}\exp\left(t\Omega_\ell(\theta)\right)\large|_{t=0}\\
            &= U(\theta)\Omega_\ell(\theta)

        where :math:`\Omega_\ell(\theta)` is the "effective" generator associated to :math:`U`
        and the parameter :math:`\theta_\ell` at :math:`\theta`. This means that the
        one-parameter group :math:`\exp(t\Omega_\ell)`
        generated by :math:`\Omega_\ell` and shifted by :math:`U(\theta)^\dagger`
        has the tangent vector :math:`\partial_\ell U(\theta)` at :math:`t=0`.

        Each effective generator can in turn be represented in the Pauli basis of
        :math:`\mathfrak{su}(N)`, which consists of the Pauli words :math:`\{P_m\}_{m=1}^d`
        from above:

        .. math::

            \Omega_\ell = \sum_{m=1}^d \omega_{\ell m} P_m.

        This gives a linear map between the effective generators and all Pauli words.
        Therefore we may compute the partial derivatives of :math:`U(\theta)`
        above as well by applying the linear map :math:`\omega` to the derivatives of rotations
        generated by Pauli words:

        .. math::

            \frac{\partial}{\partial_\ell} U(\theta) &=
            U(\theta) \sum_{m=1}^d \omega_{\ell m} P_m\\
            &=\sum_{m=1}^d 2i\omega_{\ell m} U(\theta) \frac{\mathrm{d}}{\mathrm{d}t}
            \exp\left(-i\frac{t}{2} P_m\right)\large|_{t=0},

        where the additional prefactor compensates for the notation convention for Pauli rotations.

        Computing the derivative based on parameter shifts, as in :func:`~.gradients.param_shift`,
        is "linear" in the generator. This allows us to
        combine the parameter-shift derivatives of an expectation value-based function :math:`f`
        with respect to the inserted Pauli rotations (at :math:`t=0`) into the derivatives of
        :math:`f` with respect to the original parameters of ``SpecialUnitary``. This is again
        done with :math:`\omega`:

        .. math::

            \frac{\partial}{\partial \theta_\ell} f(U(\theta))
            &= \sum_{m=1}^d 2i\omega_{\ell m} f(U(\theta)\exp\left(-i\frac{t/2}P_m\right))\large|_{t=0}
    """

    for op in tape:
        # All SpecialUnitary instances will be expanded, independent of their trainability
        if not isinstance(op, qml.SpecialUnitary):
            qml.apply(op)
            continue

        theta = op.data[0]
        # Determine the interface of the SpecialUnitary gate
        interface = qml.math.get_interface(theta)
        if interface == "numpy":
            # For numpy interface  we can not compute the linear map omega, which requires autodiff
            qml.apply(op)
            continue

        num_wires = len(op.wires)
        # Get all Pauli words for the basis of the Lie algebra for this gate
        words = pauli_words(num_wires)

        # Compute the linear map that transforms between the Pauli basis and effective generators
        # Consider the mathematical derivation for the prefactor 2j
        omega = qml.math.real(2j * get_one_parameter_coeffs(theta, num_wires, interface))

        # Create zero parameters for each Pauli rotation gate that take over the trace of theta
        zeros = _trainable_zeros_like(theta)
        # Apply the linear map omega to the zeros to create the correct preprocessing Jacobian
        zeros = qml.math.tensordot(omega, zeros, axes=[[1], [0]])

        # Apply Pauli rotations that yield the Pauli basis derivatives
        _ = [qml.PauliRot(zero, word, wires=op.wires) for zero, word in zip(zeros, words)]

        # Apply the original operation, which is no longer trainable
        qml.SpecialUnitary(qml.math.to_numpy(theta), wires=op.wires, id=op.id)
