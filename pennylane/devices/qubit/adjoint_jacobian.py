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
"""Functions to apply adjoint jacobian differentiation"""
from numbers import Number
from typing import Tuple
import numpy as np

import pennylane as qml

from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape

from .apply_operation import apply_operation
from .simulate import get_final_state
from .initialize_state import create_initial_state

# pylint: disable=protected-access, too-many-branches


def _dot_product_real(bra, ket, num_wires):
    """Helper for calculating the inner product for adjoint differentiation."""
    # broadcasted inner product not summing over first dimension of the bra tensor
    sum_axes = tuple(range(1, num_wires + 1))
    return qml.math.real(qml.math.sum(qml.math.conj(bra) * ket, axis=sum_axes))


def _adjoint_jacobian_state(tape: QuantumTape):
    """Calculate the full jacobian for a circuit that returns the state.

    Args:
        tape (QuantumTape): the circuit we wish to differentiate

    Returns:
        TensorLike: the full jacobian.

    See ``adjoint_jacobian.md`` for details on the algorithm.
    """
    jacobian = []

    has_state_prep = isinstance(tape[0], qml.operation.StatePrepBase)
    state = create_initial_state(tape.wires, tape[0] if has_state_prep else None)

    param_idx = has_state_prep
    for op in tape.operations[has_state_prep:]:
        jacobian = [apply_operation(op, jac) for jac in jacobian]

        if op.num_params == 1:
            if param_idx in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                jacobian.append(
                    apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), state)
                )

            param_idx += 1
        state = apply_operation(op, state)

    return tuple(jac.flatten() for jac in jacobian)


def adjoint_jacobian(tape: QuantumTape, state=None):
    """Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Cannot differentiate with respect to observables.

        * Cannot differentiate with respect to state-prep operations.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.

    """
    # Map wires if custom wire labels used
    tape = tape.map_to_standard_wires()

    if isinstance(tape.measurements[0], qml.measurements.StateMP):
        return _adjoint_jacobian_state(tape)

    ket = state if state is not None else get_final_state(tape)[0]

    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * len(tape.wires), dtype=np.complex128)
    for kk, obs in enumerate(tape.observables):
        bras[kk, ...] = 2 * apply_operation(obs, ket)

    jac = np.zeros((len(tape.observables), len(tape.trainable_params)))

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1
    for op in reversed(tape.operations[tape.num_preps :]):
        if isinstance(op, qml.Snapshot):
            continue
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.num_params == 1:
            if param_number in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)
                jac[:, trainable_param_number] = _dot_product_real(bras, ket_temp, len(tape.wires))

                trainable_param_number -= 1
            param_number -= 1

        for kk in range(n_obs):
            bras[kk, ...] = apply_operation(adj_op, bras[kk, ...])

    # Post-process the Jacobian matrix for the new return
    jac = np.squeeze(jac)

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)


def adjoint_jvp(tape: QuantumTape, tangents: Tuple[Number], state=None):
    """The jacobian vector product used in forward mode calculation of derivatives.

    Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        tangents (Tuple[Number]): gradient vector for input parameters.
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for output parameters
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tapes, fn = qml.map_wires(tape, wire_map)
        tape = fn(tapes)

    ket = state if state is not None else get_final_state(tape)[0]

    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * len(tape.wires), dtype=np.complex128)
    for i, obs in enumerate(tape.observables):
        bras[i] = apply_operation(obs, ket)

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1

    tangents_out = np.zeros(n_obs)

    for op in reversed(tape.operations[tape.num_preps :]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.num_params == 1:
            if param_number in tape.trainable_params:
                # don't do anything if the tangent is 0
                if not np.allclose(tangents[trainable_param_number], 0):
                    d_op_matrix = operation_derivative(op)
                    ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)

                    tangents_out += (
                        2
                        * _dot_product_real(bras, ket_temp, len(tape.wires))
                        * tangents[trainable_param_number]
                    )

                trainable_param_number -= 1
            param_number -= 1

        for i in range(n_obs):
            bras[i] = apply_operation(adj_op, bras[i])

    if n_obs == 1:
        return np.array(tangents_out[0])

    return tuple(np.array(t) for t in tangents_out)


def _get_vjp_bras(tape, cotangents, ket):
    """Helper function for getting the bras for adjoint vjp, the batch size of the
    cotangents, as well as a list of indices for which the cotangents are zero.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of.
        tangents (Tuple[Number]): gradient vector for input parameters.
        ket (TensorLike): the final state of the circuit.

    Returns:
        Tuple[TensorLike, int, List]: The return contains the following:
            * Final bra for batch size ``None``, else array of bras
            * Batch size. None if cotangents are not batched
            * List containing batch indices that are zero. Empty for unbatched
              cotangents
    """

    if isinstance(tape.measurements[0], qml.measurements.StateMP):
        batched_cotangents = np.ndim(cotangents) == 2
        batch_size = np.shape(cotangents)[0] if batched_cotangents else None
        bras = np.conj(cotangents.reshape(-1, *ket.shape))
        bras = bras if batched_cotangents else np.squeeze(bras)
        return bras, batch_size, []

    # If not state measurement, measurements are guaranteed to be expectation values

    single_cotangent = len(tape.measurements) == 1

    if not single_cotangent:
        # Pad cotangents if shape is inhomogenous
        # inner_shape will only be None if cotangents is a vector. We assume that for
        # inhomogenous cotangents, all non-scalar values have the same shape.
        inner_shape = next((np.shape(cot) for cot in cotangents if np.shape(cot) != ()), None)
        if inner_shape is not None:
            # Batched cotangents. Find scalar zeros and pad to make the shape of cotangents
            # homogenous
            new_cotangents = []

            for i, c in enumerate(cotangents):
                if np.shape(c) == () and np.allclose(c, 0.0):
                    new_cotangents.append(np.zeros(inner_shape))
                else:
                    new_cotangents.append(c)

            cotangents = new_cotangents

    cotangents = np.array(cotangents)
    if single_cotangent:
        # Expand dimensions for cases when there is a single broadcasted cotangent
        # so that the cotangent has 2 dimensions, which is expected in the rest of the
        # function for batched cotangents. For unbatched cases, this will make a scalar
        # cotangent a one-item array
        cotangents = np.expand_dims(cotangents, 0)

    # One dimension for number of expectation values, one dimension for batch size.
    batched_cotangents = np.ndim(cotangents) == 2
    batch_size = cotangents.shape[1] if batched_cotangents else None
    if np.allclose(cotangents, 0.0):
        return None, batch_size, []

    new_obs, null_batch_indices = [], []

    # Collect list of observables to use for the adjoint algorithm. These will be used
    # to construct the initial bras
    if batched_cotangents:
        for i, cots in enumerate(cotangents.T):
            new_cs, new_os = [], []
            for c, o in zip(cots, tape.observables):
                if not np.allclose(c, 0.0):
                    new_cs.append(c)
                    new_os.append(o)
            if len(new_cs) == 0:
                null_batch_indices.append(i)
            else:
                new_obs.append(qml.dot(new_cs, new_os))

    else:
        new_cs, new_os = [], []
        for c, o in zip(cotangents, tape.observables):
            if not np.allclose(c, 0.0):
                new_cs.append(c)
                new_os.append(o)

        new_obs.append(qml.dot(new_cs, new_os))

    # Create bra(s) by taking product of observable(s) with the final state
    bras = np.empty((len(new_obs), *ket.shape), dtype=ket.dtype)

    for kk, obs in enumerate(new_obs):
        if obs.pauli_rep is not None:
            flat_bra = obs.pauli_rep.dot(ket.flatten(), wire_order=list(range(tape.num_wires)))
            bras[kk] = 2 * flat_bra.reshape(ket.shape)
        else:
            bras[kk] = 2 * apply_operation(obs, ket)

    bras = bras if batched_cotangents else np.squeeze(bras)

    return bras, batch_size, null_batch_indices


def adjoint_vjp(tape: QuantumTape, cotangents: Tuple[Number], state=None):
    """The vector jacobian product used in reverse-mode differentiation.

    Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        cotangents (Tuple[Number]): gradient vector for output parameters. For computing
            the full Jacobian, the cotangents can be batched to vectorize the computation.
            In this case, the cotangents can have the following shapes. ``batch_size``
            below refers to the number of entries in the Jacobian:

            * For a state measurement, cotangents must have shape ``(batch_size, 2 ** n_wires)``.
            * For ``n`` expectation values, the cotangents must have shape ``(n, batch_size)``.
              If ``n = 1``, then the shape must be ``(batch_size,)``.

        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for input parameters
    """
    # See ``adjoint_jacobian.md`` to more information on the algorithm.

    # Map wires if custom wire labels used)
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tapes, fn = qml.map_wires(tape, wire_map)
        tape = fn(tapes)

    ket = state if state is not None else get_final_state(tape)[0]

    bras, batch_size, null_batch_indices = _get_vjp_bras(tape, cotangents, ket)
    if bras is None:
        # Cotangents are zeros
        if batch_size is None:
            return tuple(0.0 for _ in tape.trainable_params)
        return tuple(np.zeros((len(tape.trainable_params), batch_size)))

    if isinstance(tape.measurements[0], qml.measurements.StateMP):

        def real_if_expval(val):
            return val

    else:

        def real_if_expval(val):
            return np.real(val)

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1

    res_shape = (
        (len(tape.trainable_params),)
        if batch_size is None
        else (len(tape.trainable_params), batch_size)
    )
    cotangents_in = np.empty(res_shape, dtype=tape.measurements[0].numeric_type)
    summing_axis = None if batch_size is None else tuple(range(1, np.ndim(bras)))

    for op in reversed(tape.operations[tape.num_preps :]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.num_params == 1:
            if param_number in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)

                # Pad cotangent in with zeros for batch number with zero cotangents
                cot_in = real_if_expval(np.sum(np.conj(bras) * ket_temp, axis=summing_axis))
                for i in null_batch_indices:
                    cot_in = np.insert(cot_in, i, 0.0)
                cotangents_in[trainable_param_number] = cot_in

                trainable_param_number -= 1
            param_number -= 1

        bras = apply_operation(adj_op, bras, is_state_batched=bool(batch_size))

    return tuple(cotangents_in)
