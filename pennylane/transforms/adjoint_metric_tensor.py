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
Contains the adjoint_metric_tensor.
"""
import warnings
from itertools import chain
from pennylane import numpy as np

import pennylane as qml

# pylint: disable=protected-access
from pennylane.transforms.metric_tensor import _contract_metric_tensor_with_cjac


def _apply_operations(state, op, device, invert=False):
    """Wrapper that allows to apply a variety of operations---or groups
    of operations---to a state or to prepare a new state.
    If ``invert=True``, this function makes sure not to alter the operations.
    The state of the device, however may be altered, depending on the
    device and performed operation(s).
    """
    # pylint: disable=protected-access
    if isinstance(op, (list, np.ndarray)):
        if invert:
            op = op[::-1]
        for _op in op:
            state = _apply_operations(state, _op, device, invert)
        return state

    if isinstance(op, qml.QubitStateVector):
        if invert:
            raise ValueError("Can't invert state preparation.")
        device._apply_state_vector(op.parameters[0], op.wires)
        return device._state

    if isinstance(op, qml.BasisState):
        if invert:
            raise ValueError("Can't invert state preparation.")
        device._apply_basis_state(op.parameters[0], op.wires)
        return device._state

    if invert:
        op.inv()
    state = device._apply_operation(state, op)
    if invert:
        op.inv()

    return state


def _group_operations(tape):
    """Divide all operations of a tape into trainable operations and blocks
    of untrainable operations after each trainable one."""

    # Extract tape operations list
    ops = tape.operations
    # Find the indices of trainable operations in the tape operations list
    trainables = np.where([qml.operation.is_trainable(op) for op in ops])[0]
    # Add the indices incremented by one to the trainable indices
    split_ids = list(chain.from_iterable([idx, idx + 1] for idx in trainables))

    # Split at trainable and incremented indices to get groups after trainable
    # operations and single trainable operations (in alternating order)
    all_groups = np.split(ops, split_ids)

    # Collect trainable operations and groups after trainable operations
    # the first set of non-trainable ops are the ops "after the -1st" trainable op
    group_after_trainable_op = dict(enumerate(all_groups[::2], start=-1))
    trainable_operations = list(chain.from_iterable(all_groups[1::2]))

    return trainable_operations, group_after_trainable_op


def adjoint_metric_tensor(circuit, device=None, hybrid=True):
    r"""Implements the adjoint method outlined in
    `Jones <https://arxiv.org/abs/2011.02991>`__ to compute the metric tensor.

    A forward pass followed by intermediate partial backwards passes are
    used to evaluate the metric tensor in :math:`\mathcal{O}(p^2)` operations,
    where :math:`p` is the number of trainable operations, using 4 state
    vectors.

    .. note::
        The adjoint metric tensor method has the following restrictions:

        * Currently only ``"default.qubit"`` with ``shots=None`` is supported.

        * We assume the circuit to be composed of unitary gates only and rely
          on the ``generator`` property of the gates to be implemented.
          Note also that this makes the metric tensor strictly real-valued.

    Args:
        circuit (.QuantumTape or .QNode): Circuit to compute the metric tensor of
        device (.Device): Device to use for the adjoint method
        hybrid (bool): Whether to take classical preprocessing into account. Ignored if
            ``circuit`` is a tape.

    Returns:
        array: the metric tensor of the tape with respect to its trainable parameters.
        Dimensions are ``(tape.num_params, tape.num_params)``.

    .. seealso:: :func:`~.metric_tensor` for hardware-compatible metric tensor computations.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights):
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(weights[2], wires=1)
            qml.RZ(weights[3], wires=0)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(1))

    We can use the ``adjoint_metric_tensor`` transform to generate a new function
    that returns the metric tensor of this QNode:

    >>> mt_fn = qml.adjoint_metric_tensor(circuit)
    >>> weights = np.array([0.1, 0.2, 0.4, 0.5], requires_grad=True)
    >>> mt_fn(weights)
    tensor([[ 0.25  ,  0.    , -0.0497, -0.0497],
            [ 0.    ,  0.2475,  0.0243,  0.0243],
            [-0.0497,  0.0243,  0.0123,  0.0123],
            [-0.0497,  0.0243,  0.0123,  0.0123]], requires_grad=True)

    This approach has the benefit of being significantly faster than the hardware-ready
    ``metric_tensor`` function:

    >>> import time
    >>> start_time = time.process_time()
    >>> mt = mt_fn(weights)
    >>> time.process_time() - start_time
    0.019
    >>> mt_fn_2 = qml.metric_tensor(circuit)
    >>> start_time = time.process_time()
    >>> mt = mt_fn_2(weights)
    >>> time.process_time() - start_time
    0.025

    This speedup becomes more drastic for larger circuits.
    The drawback of the adjoint method is that it is only available on simulators and without
    shot simulations.
    """
    if isinstance(circuit, qml.tape.QuantumTape):
        return _adjoint_metric_tensor_tape(circuit, device)
    if isinstance(circuit, (qml.QNode, qml.ExpvalCost)):
        return _adjoint_metric_tensor_qnode(circuit, device, hybrid)

    raise qml.QuantumFunctionError("The passed object is not a QuantumTape or QNode.")


def _adjoint_metric_tensor_tape(tape, device):
    """Computes the metric tensor of a tape using the adjoint method and a given device."""
    # pylint: disable=protected-access
    if device.shots is not None:
        raise ValueError(
            "The adjoint method for the metric tensor is only implemented for shots=None"
        )
    tape = qml.transforms.expand_trainable_multipar(tape)

    # Divide all operations of a tape into trainable operations and blocks
    # of untrainable operations after each trainable one.
    trainable_operations, group_after_trainable_op = _group_operations(tape)

    dim = 2**device.num_wires
    # generate and extract initial state
    psi = device._create_basis_state(0)

    # initialize metric tensor components (which all will be real-valued)
    like_real = qml.math.real(psi[0])
    L = qml.math.convert_like(qml.math.zeros((tape.num_params, tape.num_params)), like_real)
    T = qml.math.convert_like(qml.math.zeros((tape.num_params,)), like_real)

    psi = _apply_operations(psi, group_after_trainable_op[-1], device)

    for j, outer_op in enumerate(trainable_operations):
        generator_1, prefactor_1 = qml.utils.get_generator(outer_op, return_matrix=True)

        # the state vector phi is missing a factor of 1j * prefactor_1
        phi = device._apply_unitary(
            psi, qml.math.convert_like(generator_1, like_real), outer_op.wires
        )

        phi_real = qml.math.reshape(qml.math.real(phi), (dim,))
        phi_imag = qml.math.reshape(qml.math.imag(phi), (dim,))
        diag_value = prefactor_1**2 * (
            qml.math.dot(phi_real, phi_real) + qml.math.dot(phi_imag, phi_imag)
        )
        L = qml.math.scatter_element_add(L, (j, j), diag_value)

        lam = psi * 1.0
        lam_real = qml.math.reshape(qml.math.real(lam), (dim,))
        lam_imag = qml.math.reshape(qml.math.imag(lam), (dim,))

        # this entry is missing a factor of 1j
        value = prefactor_1 * (qml.math.dot(lam_real, phi_real) + qml.math.dot(lam_imag, phi_imag))
        T = qml.math.scatter_element_add(T, (j,), value)

        for i in range(j - 1, -1, -1):
            # after first iteration of inner loop: apply U_{i+1}^\dagger
            if i < j - 1:
                phi = _apply_operations(phi, trainable_operations[i + 1], device, invert=True)
            # apply V_{i}^\dagger
            phi = _apply_operations(phi, group_after_trainable_op[i], device, invert=True)
            lam = _apply_operations(lam, group_after_trainable_op[i], device, invert=True)
            inner_op = trainable_operations[i]
            # extract and apply G_i
            generator_2, prefactor_2 = qml.utils.get_generator(inner_op, return_matrix=True)
            # this state vector is missing a factor of 1j * prefactor_2
            mu = device._apply_unitary(lam, qml.math.convert_like(generator_2, lam), inner_op.wires)
            phi_real = qml.math.reshape(qml.math.real(phi), (dim,))
            phi_imag = qml.math.reshape(qml.math.imag(phi), (dim,))
            mu_real = qml.math.reshape(qml.math.real(mu), (dim,))
            mu_imag = qml.math.reshape(qml.math.imag(mu), (dim,))
            # this entry is missing a factor of 1j * (-1j) = 1, i.e. none
            value = (
                prefactor_1
                * prefactor_2
                * (qml.math.dot(mu_real, phi_real) + qml.math.dot(mu_imag, phi_imag))
            )
            L = qml.math.scatter_element_add(
                L, [(i, j), (j, i)], value * qml.math.convert_like(qml.math.ones((2,)), value)
            )
            # apply U_i^\dagger
            lam = _apply_operations(lam, inner_op, device, invert=True)

        # apply U_j and V_j
        psi = _apply_operations(psi, [outer_op, *group_after_trainable_op[j]], device)

    # postprocessing: combine L and T into the metric tensor.
    # We require outer(conj(T), T) here, but as we skipped the factor 1j above,
    # the stored T is real-valued. Thus we have -1j*1j*outer(T, T) = outer(T, T)
    metric_tensor = L - qml.math.tensordot(T, T, 0)

    return metric_tensor


def _adjoint_metric_tensor_qnode(qnode, device, hybrid):
    """Computes the metric tensor of a qnode using the adjoint method and its device.
    For ``hybrid==True`` this wrapper accounts for classical preprocessing within the
    QNode.
    """
    if device is None:
        if isinstance(qnode, qml.ExpvalCost):
            if qnode._multiple_devices:  # pylint: disable=protected-access
                warnings.warn(
                    "ExpvalCost was instantiated with multiple devices. Only the first device "
                    "will be used to evaluate the metric tensor with the adjoint method.",
                    UserWarning,
                )
            qnode = qnode.qnodes.qnodes[0]
        device = qnode.device

    cjac_fn = qml.transforms.classical_jacobian(
        qnode, expand_fn=qml.transforms.expand_trainable_multipar
    )

    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        mt = _adjoint_metric_tensor_tape(qnode.qtape, device)

        if not hybrid:
            return mt

        cjac = cjac_fn(*args, **kwargs)

        return _contract_metric_tensor_with_cjac(mt, cjac)

    return wrapper
