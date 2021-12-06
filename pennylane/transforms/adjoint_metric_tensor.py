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
from pennylane import numpy as np

import pennylane as qml


def _get_generator(op):
    """Reads out the generator and prefactor of an operation and converts
    to matrix if necessary.

    Args:
        op (:class:`~.Operation`): Operation to obtain the generator of.
    Returns:
        array[float]: Generator matrix
        float: Prefactor of the generator
    """

    generator, prefactor = op.generator
    if not isinstance(generator, np.ndarray):
        generator = generator.matrix
    if op.inverse:
        generator = generator.conj().T
        prefactor *= -1

    return generator, prefactor


def _apply_any_operation(state, op, device, invert=False):
    """Wrapper that allows to apply a variety of operations---or groups
    of operations---to a state or to prepare a new state.
    If ``invert=True``, this function makes sure not to alter the operations.
    The state of the device, however may be altered, depending on the
    device and performed operation(s).
    """
    # pylint: disable=protected-access
    if isinstance(op, list):
        if invert:
            op = op[::-1]
        for _op in op:
            state = _apply_any_operation(state, _op, device, invert)
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
    trainable_operations = []
    group_after_trainable_op = {}
    # the first set of non-trainable ops are the ops "after the -1st" trainable op
    trainable_idx = -1
    group_after = []
    for op in tape.operations:
        if qml.operation.is_trainable(op):
            trainable_operations.append(op)
            group_after_trainable_op[trainable_idx] = group_after
            trainable_idx += 1
            group_after = []
        else:
            group_after.append(op)
    # store operations after last trainable op
    group_after_trainable_op[trainable_idx] = group_after

    return trainable_operations, group_after_trainable_op


def adjoint_metric_tensor(circuit, device=None, hybrid=True):
    """Implements the adjoint method outlined in
    `Jones <https://arxiv.org/abs/2011.02991>`__ to compute the metric tensor.

    A mixture of a main forward pass and intermediate partial backwards passes is
    used to evaluate the metric tensor in O(num_params^2) operations, using 4 state
    vectors.

    .. note::
        The adjoint metric tensor method has the following restrictions:

        * As it requires knowledge of the statevector, only statevector simulator
          devices can be used.

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
        Dimensions are ``(len(trainable_params), len(trainable_params))``.

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
    # original_parameters = _prepare_tape_params(tape, device, interface)
    tape = qml.transforms.expand_trainable_multipar(tape)

    # Divide all operations of a tape into trainable operations and blocks
    # of untrainable operations after each trainable one.
    trainable_operations, group_after_trainable_op = _group_operations(tape)

    # generate and extract initial state
    psi = device._create_basis_state(0)  # pylint: disable=protected-access
    dim = 2 ** device.num_wires

    num_params = len(tape.trainable_params)
    # initialize metric tensor components (which all will be real-valued)
    _like_real = qml.math.real(psi[0])
    L = qml.math.convert_like(qml.math.zeros((num_params, num_params)), _like_real)
    T = qml.math.convert_like(qml.math.zeros((num_params,)), _like_real)

    psi = _apply_any_operation(psi, group_after_trainable_op[-1], device)

    for j, outer_op in enumerate(trainable_operations):
        # sub_L = []
        generator_1, prefactor_1 = _get_generator(outer_op)

        # the state vector phi is missing a factor of 1j * prefactor_1
        phi = device._apply_unitary(psi, qml.math.convert_like(generator_1, psi), outer_op.wires)

        phi_real = qml.math.reshape(qml.math.real(phi), (dim,))
        phi_imag = qml.math.reshape(qml.math.imag(phi), (dim,))
        diag_value = prefactor_1 ** 2 * (
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
                trainable_operations[i + 1].inv()
                phi = device._apply_operation(phi, trainable_operations[i + 1])
                trainable_operations[i + 1].inv()
            # apply V_{i}^\dagger
            phi = _apply_any_operation(phi, group_after_trainable_op[i], device, invert=True)
            lam = _apply_any_operation(lam, group_after_trainable_op[i], device, invert=True)
            inner_op = trainable_operations[i]
            # extract and apply G_i
            generator_2, prefactor_2 = _get_generator(inner_op)
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
            inner_op.inv()
            lam = device._apply_operation(lam, inner_op)
            inner_op.inv()

        # apply U_j
        psi = device._apply_operation(psi, outer_op)
        # apply V_j
        psi = _apply_any_operation(psi, group_after_trainable_op[j], device)

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

        if isinstance(cjac, tuple):
            if len(cjac) == 1:
                cjac = cjac[0]
            else:
                # Classical processing of multiple arguments is present. Return cjac.T @ mt @ cjac.
                metric_tensors = []

                for c in cjac:
                    if c is not None:
                        _mt = qml.math.tensordot(mt, c, axes=[[-1], [0]])
                        _mt = qml.math.tensordot(c, _mt, axes=[[0], [0]])
                        metric_tensors.append(_mt)

                return tuple(metric_tensors)

        is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])

        if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
            # Classical Jacobian is the identity. No classical processing
            # is present inside the QNode.
            return mt

        # Classical processing of a single argument is present. Return mt @ cjac.
        mt = qml.math.tensordot(mt, cjac, [[-1], [0]])
        mt = qml.math.tensordot(cjac, mt, [[0], [0]])
        return mt

    return wrapper
