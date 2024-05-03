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
Contains the implementation of quantum fidelity.

Note: care needs to be taken to make it fully differentiable. An explanation can
be found in pennylane/math/fidelity_gradient.md
"""
from functools import lru_cache

import autograd
import autoray as ar
import pennylane as qml

from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector


def fidelity_statevector(state0, state1, check_state=False, c_dtype="complex128"):
    r"""Compute the fidelity for two states (given as state vectors) acting on quantum
    systems with the same size.

    The fidelity for two pure states given by state vectors :math:`\ket{\psi}` and :math:`\ket{\phi}`
    is defined as

    .. math::
        F( \ket{\psi} , \ket{\phi}) = \left|\braket{\psi, \phi}\right|^2

    This is faster than calling :func:`pennylane.math.fidelity` on the density matrix
    representation of pure states.

    .. note::
        It supports all interfaces (NumPy, Autograd, Torch, TensorFlow and Jax). The second state is coerced
        to the type and dtype of the first state. The fidelity is returned in the type of the interface of the
        first state.

    Args:
        state0 (tensor_like): ``(2**N)`` or ``(batch_dim, 2**N)`` state vector.
        state1 (tensor_like): ``(2**N)`` or ``(batch_dim, 2**N)`` state vector.
        check_state (bool): If True, the function will check the validity of both states; that is,
            the shape and the norm
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Fidelity between the two quantum states.

    **Example**

    Two state vectors can be used as arguments and the fidelity (overlap) is returned, e.g.:

    >>> state0 = [0.98753537-0.14925137j, 0.00746879-0.04941796j]
    >>> state1 = [0.99500417+0.j, 0.09983342+0.j]
    >>> qml.math.fidelity_statevector(state0, state1)
    0.9905158135644924

    .. seealso:: :func:`pennylane.math.fidelity` and :func:`pennylane.qinfo.transforms.fidelity`

    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)
    state1 = cast(state1, dtype=c_dtype)

    if check_state:
        _check_state_vector(state0)
        _check_state_vector(state1)

    if qml.math.shape(state0)[-1] != qml.math.shape(state1)[-1]:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    batched0 = len(qml.math.shape(state0)) > 1
    batched1 = len(qml.math.shape(state1)) > 1

    # Two pure states, squared overlap
    indices0 = "ab" if batched0 else "b"
    indices1 = "ab" if batched1 else "b"
    target = "a" if batched0 or batched1 else ""
    overlap = qml.math.einsum(
        f"{indices0},{indices1}->{target}", state0, qml.math.conj(state1), optimize="greedy"
    )

    overlap = qml.math.abs(overlap) ** 2
    return overlap


def fidelity(state0, state1, check_state=False, c_dtype="complex128"):
    r"""Compute the fidelity for two states (given as density matrices) acting on quantum
    systems with the same size.

    The fidelity for two mixed states given by density matrices :math:`\rho` and :math:`\sigma`
    is defined as

    .. math::
        F( \rho , \sigma ) = \text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    .. note::
        It supports all interfaces (NumPy, Autograd, Torch, TensorFlow and Jax). The second state is coerced
        to the type and dtype of the first state. The fidelity is returned in the type of the interface of the
        first state.

    Args:
        state0 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        state1 (tensor_like): ``(2**N, 2**N)`` or ``(batch_dim, 2**N, 2**N)`` density matrix.
        check_state (bool): If True, the function will check the validity of both states; that is,
            (shape, trace, positive-definitiveness) for density matrices.
        c_dtype (str): Complex floating point precision type.

    Returns:
        float: Fidelity between the two quantum states.

    **Example**

    To find the fidelity between two state vectors, call :func:`~.math.dm_from_state_vector` on the
    inputs first, e.g.:

    >>> state0 = qml.math.dm_from_state_vector([0.98753537-0.14925137j, 0.00746879-0.04941796j])
    >>> state1 = qml.math.dm_from_state_vector([0.99500417+0.j, 0.09983342+0.j])
    >>> qml.math.fidelity(state0, state1)
    0.9905158135644924

    To find the fidelity between two density matrices, they can be passed directly:

    >>> state0 = [[1, 0], [0, 0]]
    >>> state1 = [[0, 0], [0, 1]]
    >>> qml.math.fidelity(state0, state1)
    0.0

    .. seealso:: :func:`pennylane.math.fidelity_statevector` and :func:`pennylane.qinfo.transforms.fidelity`

    """
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)
    state1 = cast(state1, dtype=c_dtype)

    if check_state:
        _check_density_matrix(state0)
        _check_density_matrix(state1)

    if qml.math.shape(state0)[-1] != qml.math.shape(state1)[-1]:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    batch_size0 = qml.math.shape(state0)[0] if qml.math.ndim(state0) > 2 else None
    batch_size1 = qml.math.shape(state1)[0] if qml.math.ndim(state1) > 2 else None

    if qml.math.get_interface(state0) == "jax" or qml.math.get_interface(state1) == "jax":
        if batch_size0 and not batch_size1:
            state1 = qml.math.broadcast_to(state1, (batch_size0, *qml.math.shape(state1)))
        elif not batch_size0 and batch_size1:
            state0 = qml.math.broadcast_to(state0, (batch_size1, *qml.math.shape(state0)))

    # Two mixed states
    _register_vjp(state0, state1)
    fid = qml.math.compute_fidelity(state0, state1)
    return fid


def _register_vjp(state0, state1):
    """
    Register the interface-specific custom VJP based on the interfaces of the given states

    This function is needed because we don't want to register the custom
    VJPs at PennyLane import time.
    """
    interface = qml.math.get_interface(state0, state1)
    if interface == "jax":
        _register_jax_vjp()
    elif interface == "torch":
        _register_torch_vjp()
    elif interface == "tensorflow":
        _register_tf_vjp()


def _compute_fidelity_vanilla(density_matrix0, density_matrix1):
    r"""Compute the fidelity for two density matrices with the same number of wires.

    .. math::
            F( \rho , \sigma ) = -\text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2
    """
    # Implementation in single dispatches (sqrt(rho))
    sqrt_mat = qml.math.sqrt_matrix(density_matrix0)

    # sqrt(rho) * sigma * sqrt(rho)
    sqrt_mat_sqrt = sqrt_mat @ density_matrix1 @ sqrt_mat

    # extract eigenvalues
    evs = qml.math.eigvalsh(sqrt_mat_sqrt)
    evs = qml.math.real(evs)
    evs = qml.math.where(evs > 0.0, evs, 0)

    trace = (qml.math.sum(qml.math.sqrt(evs), -1)) ** 2

    return trace


def _compute_fidelity_vjp0(dm0, dm1, grad_out):
    """
    Compute the VJP of fidelity with respect to the first density matrix
    """
    # sqrt of sigma
    sqrt_dm1 = qml.math.sqrt_matrix(dm1)

    # eigendecomposition of sqrt(sigma) * rho * sqrt(sigma)
    evs0, u0 = qml.math.linalg.eigh(sqrt_dm1 @ dm0 @ sqrt_dm1)
    evs0 = qml.math.real(evs0)
    evs0 = qml.math.where(evs0 > 1e-15, evs0, 1e-15)
    evs0 = qml.math.cast_like(evs0, sqrt_dm1)

    if len(qml.math.shape(dm0)) == 2 and len(qml.math.shape(dm1)) == 2:
        u0_dag = qml.math.transpose(qml.math.conj(u0))
        grad_dm0 = sqrt_dm1 @ u0 @ (1 / qml.math.sqrt(evs0)[..., None] * u0_dag) @ sqrt_dm1

        # torch and tensorflow use the Wirtinger derivative which is a different convention
        # than the one autograd and jax use for complex differentiation
        if qml.math.get_interface(dm0) in ["torch", "tensorflow"]:
            grad_dm0 = qml.math.sum(qml.math.sqrt(evs0), -1) * grad_dm0
        else:
            grad_dm0 = qml.math.sum(qml.math.sqrt(evs0), -1) * qml.math.transpose(grad_dm0)

        res = grad_dm0 * qml.math.cast_like(grad_out, grad_dm0)
        return res

    # broadcasting case
    u0_dag = qml.math.transpose(qml.math.conj(u0), (0, 2, 1))
    grad_dm0 = sqrt_dm1 @ u0 @ (1 / qml.math.sqrt(evs0)[..., None] * u0_dag) @ sqrt_dm1

    # torch and tensorflow use the Wirtinger derivative which is a different convention
    # than the one autograd and jax use for complex differentiation
    if qml.math.get_interface(dm0) in ["torch", "tensorflow"]:
        grad_dm0 = qml.math.sum(qml.math.sqrt(evs0), -1)[:, None, None] * grad_dm0
    else:
        grad_dm0 = qml.math.sum(qml.math.sqrt(evs0), -1)[:, None, None] * qml.math.transpose(
            grad_dm0, (0, 2, 1)
        )

    return grad_dm0 * qml.math.cast_like(grad_out, grad_dm0)[:, None, None]


def _compute_fidelity_vjp1(dm0, dm1, grad_out):
    """
    Compute the VJP of fidelity with respect to the second density matrix
    """
    # pylint: disable=arguments-out-of-order
    return _compute_fidelity_vjp0(dm1, dm0, grad_out)


def _compute_fidelity_grad(dm0, dm1, grad_out):
    return _compute_fidelity_vjp0(dm0, dm1, grad_out), _compute_fidelity_vjp1(dm0, dm1, grad_out)


################################ numpy ###################################

ar.register_function("numpy", "compute_fidelity", _compute_fidelity_vanilla)

################################ autograd ################################


@autograd.extend.primitive
def _compute_fidelity_autograd(dm0, dm1):
    return _compute_fidelity_vanilla(dm0, dm1)


def _compute_fidelity_autograd_vjp0(_, dm0, dm1):
    def vjp(grad_out):
        return _compute_fidelity_vjp0(dm0, dm1, grad_out)

    return vjp


def _compute_fidelity_autograd_vjp1(_, dm0, dm1):
    def vjp(grad_out):
        return _compute_fidelity_vjp1(dm0, dm1, grad_out)

    return vjp


autograd.extend.defvjp(
    _compute_fidelity_autograd, _compute_fidelity_autograd_vjp0, _compute_fidelity_autograd_vjp1
)
ar.register_function("autograd", "compute_fidelity", _compute_fidelity_autograd)

################################# jax #####################################


@lru_cache(maxsize=None)
def _register_jax_vjp():
    """
    Register the custom VJP for JAX
    """
    # pylint: disable=import-outside-toplevel
    import jax

    @jax.custom_vjp
    def _compute_fidelity_jax(dm0, dm1):
        return _compute_fidelity_vanilla(dm0, dm1)

    def _compute_fidelity_jax_fwd(dm0, dm1):
        fid = _compute_fidelity_jax(dm0, dm1)
        return fid, (dm0, dm1)

    def _compute_fidelity_jax_bwd(res, grad_out):
        dm0, dm1 = res
        return _compute_fidelity_grad(dm0, dm1, grad_out)

    _compute_fidelity_jax.defvjp(_compute_fidelity_jax_fwd, _compute_fidelity_jax_bwd)
    ar.register_function("jax", "compute_fidelity", _compute_fidelity_jax)


################################ torch ###################################


@lru_cache(maxsize=None)
def _register_torch_vjp():
    """
    Register the custom VJP for torch
    """
    # pylint: disable=import-outside-toplevel,abstract-method,arguments-differ
    import torch

    class _TorchFidelity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dm0, dm1):
            """Forward pass for _compute_fidelity"""
            fid = _compute_fidelity_vanilla(dm0, dm1)
            ctx.save_for_backward(dm0, dm1)
            return fid

        @staticmethod
        def backward(ctx, grad_out):
            """Backward pass for _compute_fidelity"""
            dm0, dm1 = ctx.saved_tensors
            return _compute_fidelity_grad(dm0, dm1, grad_out)

    ar.register_function("torch", "compute_fidelity", _TorchFidelity.apply)


############################### tensorflow ################################


@lru_cache(maxsize=None)
def _register_tf_vjp():
    """
    Register the custom VJP for tensorflow
    """
    # pylint: disable=import-outside-toplevel
    import tensorflow as tf

    @tf.custom_gradient
    def _compute_fidelity_tf(dm0, dm1):
        fid = _compute_fidelity_vanilla(dm0, dm1)

        def vjp(grad_out):
            return _compute_fidelity_grad(dm0, dm1, grad_out)

        return fid, vjp

    ar.register_function("tensorflow", "compute_fidelity", _compute_fidelity_tf)
