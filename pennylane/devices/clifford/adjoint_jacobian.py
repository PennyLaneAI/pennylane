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

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape


def adjoint_jacobian(tape: QuantumTape, **kwargs):
    """For Clifford simulation this should return trivial gradients without any calculation.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.
    """
    # Map wires if custom wire labels used
    tape, _ = tape.map_to_standard_wires(), kwargs

    jac = np.squeeze(np.zeros((len(tape.observables), len(tape.trainable_params))))

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)


def adjoint_jvp(tape: QuantumTape, **kwargs):
    """The jacobian vector product used in forward mode calculation of derivatives.

    For Clifford simulation this should return trivial gradients without any calculation.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of

    Returns:
        Tuple[Number]: gradient vector for output parameters
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    tangents_out, _ = np.zeros(len(tape.observables)), kwargs

    if len(tape.observables) == 1:
        return np.array(tangents_out[0])

    return tuple(np.array(t) for t in tangents_out)


def adjoint_vjp(tape: QuantumTape, **kwargs):
    """The vector jacobian product used in reverse-mode differentiation.

    For Clifford simulation this should return trivial gradients without any calculation.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of

    Returns:
        Tuple[Number]: gradient vector for input parameters
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    cotangents_in, _ = np.empty(len(tape.trainable_params)), kwargs

    if len(tape.trainable_params) == 1:
        return np.array(cotangents_in[0])

    return tuple(np.array(t) for t in cotangents_in)
