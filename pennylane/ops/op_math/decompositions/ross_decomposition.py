# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module contains the implementation of the Ross decomposition algorithm.
"""
import jax.numpy as jnp

import pennylane as qml


# TODO: Remove or modified this
def get_clifford_ops():
    """
    Get the clifford operators.

    Returns:
        list[qml.Operation]: A list of clifford operators.
    """
    I, X, Y, Z = qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)
    H, S, Sdg = qml.H(0), qml.S(0), qml.adjoint(qml.S(0))
    return [
        I,
        H,
        S,
        X,
        Y,
        Z,
        Sdg,
        H @ S,
        H @ Z,
        H @ Sdg,
        S @ H,
        S @ X,
        S @ Y,
        Z @ H,
        Sdg @ H,
        S @ H @ S,
        S @ H @ Z,
        S @ H @ Sdg,
        Z @ H @ S,
        Z @ H @ Z,
        Z @ H @ Sdg,
        Sdg @ H @ S,
        Sdg @ H @ Z,
        Sdg @ H @ Sdg,
    ]


# TODO: remove this function
# This is a mock functions
def ross(op):
    """
    Test the Ross decomposition algorithm.
    Assumes the return from the ross function.
    The decomposition is done in the following way:
    1. The leading T gate is applied if has_leading_t is True.
    2. The syllable sequence is applied.
    3. The clifford operator is applied.

    Args:
        op (qml.Operation): The operation to decompose.

    Returns:
        tuple: A tuple containing the has_leading_t, syllable_sequence, and clifford_op_idx.
    """
    has_leading_t = True
    syllable_sequence = jnp.array([0, 1, 0, 1, 1, 0, 0])
    clifford_op_idx = 10
    return (has_leading_t, syllable_sequence, clifford_op_idx)


def ross_decomposition(op, epsilon=1e-4, **method_kwargs):
    """
    TBD...
    TODO: epsilon is expected to be there in clifford_t_transform, may not used in ross_decomposition.

    Args:
        op (~pennylane.operation.Operation): A single-qubit gate operation.
        epsilon (float): The maximum permissible error.

    Returns:
        list[~pennylane.operation.Operation]: A list of gates in the Clifford+T basis set that approximates the given
        operation along with a final global phase operation. The operations are in the circuit-order.

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with rotation angle :math:`\phi = \pi/3`:

    .. code-block:: python

        from pennylane import clifford_t_decomposition

        @qml.qnode(qml.device("null.qubit", wires=3))
        def circuit():
            qml.RZ(0.3, wires=1)
            return qml.expval(qml.PauliZ(0))

        qnode_cir = clifford_t_decomposition(circuit, method="ross")

        decomposed_circuit = qml.qjit(cir)

        result = decomposed_circuit()
        print(result)


    """
    ops = []
    wire = op.wires[0]

    # Matsumoto-Amano normal form: (T|ε)(HT|SHT)*C
    # (T|ε): Optional leading T gate
    # (HT|SHT): Middle sequence of HT or SHT syllables
    # C: Right most Clifford operator

    has_leading_t, syllable_sequence, clifford_op_idx = ross(op)

    if qml.compiler.active_compiler() == "catalyst":

        # Optional leading T gate
        leading_t_cond = qml.cond(has_leading_t, qml.X)
        leading_t_cond(wire)
        ops.append(leading_t_cond.operation)

        # Middle sequence of HT or SHT syllables.
        @qml.for_loop(start=0, stop=syllable_sequence.shape[0])
        def syllable_sequence_loop(i):
            is_HT = syllable_sequence[i]

            def compose_HT():
                qml.H(wire)
                qml.T(wire)

            def compose_SHT():
                qml.S(wire)
                qml.H(wire)
                qml.T(wire)

            qml.cond(is_HT.astype(bool), true_fn=compose_HT, false_fn=compose_SHT)()

        syllable_sequence_loop()
        ops.append(syllable_sequence_loop.operation)

    else:  # Without QJIT
        if has_leading_t:
            ops.append(qml.T(wire))

        for i in syllable_sequence.shape[0]:
            is_HT = syllable_sequence[i]

            if is_HT:
                ops += [qml.H(wire), qml.T(wire)]
            else:
                ops += [qml.S(wire), qml.H(wire), qml.T(wire)]

    # Rightmost Clifford operator
    ops.append(get_clifford_ops()[clifford_op_idx])

    phase = 0.1  # TODO: remove this
    global_phase = qml.GlobalPhase(phase)
    return ops + [global_phase]
