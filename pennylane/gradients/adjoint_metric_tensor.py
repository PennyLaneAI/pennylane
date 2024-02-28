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
from typing import Sequence, Callable
from itertools import chain
from functools import partial

import numpy as np
import pennylane as qml

# pylint: disable=too-many-statements,unused-argument
from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac
from pennylane.transforms import transform


def _reshape_real_imag(state, dim):
    state = qml.math.reshape(state, (dim,))
    return qml.math.real(state), qml.math.imag(state)


def _group_operations(tape):
    """Divide all operations of a tape into trainable operations and blocks
    of untrainable operations after each trainable one."""

    # Extract tape operations list
    ops = tape.operations
    # Find the indices of trainable operations in the tape operations list
    # pylint: disable=protected-access
    trainable_par_info = [tape._par_info[i] for i in tape.trainable_params]
    trainables = [info["op_idx"] for info in trainable_par_info]
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


def _expand_trainable_multipar(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Expand trainable multi-parameter operations in a quantum tape."""

    interface = qml.math.get_interface(*tape.get_parameters())
    use_tape_argnum = interface == "jax"
    expand_fn = qml.transforms.create_expand_trainable_multipar(
        tape, use_tape_argnum=use_tape_argnum
    )
    return [expand_fn(tape)], lambda x: x[0]


@partial(
    transform,
    expand_transform=_expand_trainable_multipar,
    classical_cotransform=_contract_metric_tensor_with_cjac,
    is_informative=True,
    use_argnum_in_expand=True,
)
def adjoint_metric_tensor(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
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
        tape (QNode or QuantumTape): Circuit to compute the metric tensor of

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the metric tensor in the form of a tensor. Dimensions are ``(tape.num_params, tape.num_params)``.

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
            return qml.expval(qml.Z(0) @ qml.Z(1)), qml.expval(qml.Y(1))

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

    def processing_fn(tapes):
        tape = tapes[0]
        if tape.shots:
            raise ValueError(
                "The adjoint method for the metric tensor is only implemented for shots=None"
            )
        if set(tape.wires) != set(range(tape.num_wires)):
            wire_map = {w: i for i, w in enumerate(tape.wires)}
            tapes, fn = qml.map_wires(tape, wire_map)
            tape = fn(tapes)

        # Divide all operations of a tape into trainable operations and blocks
        # of untrainable operations after each trainable one.

        trainable_operations, group_after_trainable_op = _group_operations(tape)

        dim = 2**tape.num_wires
        # generate and extract initial state
        prep = tape[0] if len(tape) > 0 and isinstance(tape[0], qml.operation.StatePrep) else None

        interface = qml.math.get_interface(*tape.get_parameters(trainable_only=False))
        psi = qml.devices.qubit.create_initial_state(tape.wires, prep, like=interface)

        # initialize metric tensor components (which all will be real-valued)
        like_real = qml.math.real(psi[0])
        L = qml.math.convert_like(qml.math.zeros((tape.num_params, tape.num_params)), like_real)
        T = qml.math.convert_like(qml.math.zeros((tape.num_params,)), like_real)

        for op in group_after_trainable_op[-1]:
            psi = qml.devices.qubit.apply_operation(op, psi)

        for j, outer_op in enumerate(trainable_operations):
            generator_1, prefactor_1 = qml.generator(outer_op)

            # the state vector phi is missing a factor of 1j * prefactor_1
            phi = qml.devices.qubit.apply_operation(generator_1, psi)

            phi_real, phi_imag = _reshape_real_imag(phi, dim)
            diag_value = prefactor_1**2 * (
                qml.math.dot(phi_real, phi_real) + qml.math.dot(phi_imag, phi_imag)
            )
            L = qml.math.scatter_element_add(L, (j, j), diag_value)

            lam = psi * 1.0
            lam_real, lam_imag = _reshape_real_imag(lam, dim)

            # this entry is missing a factor of 1j
            value = prefactor_1 * (
                qml.math.dot(lam_real, phi_real) + qml.math.dot(lam_imag, phi_imag)
            )
            T = qml.math.scatter_element_add(T, (j,), value)

            for i in range(j - 1, -1, -1):
                # after first iteration of inner loop: apply U_{i+1}^\dagger
                if i < j - 1:
                    phi = qml.devices.qubit.apply_operation(
                        qml.adjoint(trainable_operations[i + 1], lazy=False), phi
                    )
                # apply V_{i}^\dagger
                for op in reversed(group_after_trainable_op[i]):
                    adj_op = qml.adjoint(op, lazy=False)
                    phi = qml.devices.qubit.apply_operation(adj_op, phi)
                    lam = qml.devices.qubit.apply_operation(adj_op, lam)

                inner_op = trainable_operations[i]
                # extract and apply G_i
                generator_2, prefactor_2 = qml.generator(inner_op)
                # this state vector is missing a factor of 1j * prefactor_2
                mu = qml.devices.qubit.apply_operation(generator_2, lam)

                phi_real, phi_imag = _reshape_real_imag(phi, dim)
                mu_real, mu_imag = _reshape_real_imag(mu, dim)
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
                lam = qml.devices.qubit.apply_operation(qml.adjoint(inner_op, lazy=False), lam)

            # apply U_j and V_j
            psi = qml.devices.qubit.apply_operation(outer_op, psi)
            for op in group_after_trainable_op[j]:
                psi = qml.devices.qubit.apply_operation(op, psi)

        # postprocessing: combine L and T into the metric tensor.
        # We require outer(conj(T), T) here, but as we skipped the factor 1j above,
        # the stored T is real-valued. Thus we have -1j*1j*outer(T, T) = outer(T, T)
        metric_tensor = L - qml.math.tensordot(T, T, 0)

        return metric_tensor

    return [tape], processing_fn
