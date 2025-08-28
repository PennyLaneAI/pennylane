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
"""Utility tools for checking (non-unitary) circuit identities"""
# pylint: disable=too-many-function-args,not-callable
import numpy as np

import pennylane as qml


def state2bin(state):
    r"""
    Helper function for displaying computational basis states

    Args:
        state (tensor_like[complex]): Input state vector. Needs to contain only one non-zero entry

    Returns:
        str: String representation of basis vector including a potential phase


    """
    n = int(np.log(len(state)) / np.log(2))
    integer = np.where(state > 1e-10)[0]
    assert len(integer) == 1, f"not a binary vector {state}"
    integer = integer[0]
    phase = state[integer]

    return str(phase) + " * |" + np.binary_repr(integer, n) + ">"


def check_circuit_id(target_U, circuit, wires, aux_wires, aux_state_io, **kwargs):
    r"""
    Check (non-unitary) circuit identities

    Args:
        target_U (tensor_like[complex]): target unitary
        circuit (callable): undecorated ``qfunc`` that is supposed to realize the ``target_U``. Currently only supports one mid-circuit-measurement
        wires (qml.wires.Wires): the wires of the circuit
        aux_wires (qml.wires.Wires): the auxiliary wires
        aux_state_io (list): a ``list`` containing the expected input and output states on the ``"aux"`` wire. Defaults to :math:`|0\rangle` in both cases.
        kwargs: additional ``kwargs`` for ``circuit``

    Returns:
        bool: whether or not the circuit faithfully realizes the ``target_U``
    """

    n = int(np.log(len(target_U)) / np.log(2))
    n_aux = int(np.log(len(aux_state_io[0])) / np.log(2))
    assert n_aux == len(aux_wires)
    n_tot = n + n_aux

    @qml.set_shots(10)
    @qml.qnode(qml.device("default.qubit"), mcm_method="tree-traversal")
    def qnode(in_state, **kwargs):
        qml.StatePrep(in_state, wires=wires + aux_wires)
        circuit(**kwargs)
        qml.Snapshot(measurement=qml.state(), tag="state")

        return qml.expval(qml.Z(0))  # dummy outout, needed for workaround

    in_states = np.eye(2**n)  # all 3 qubit input states
    in_states_zero = np.array(
        [np.kron(in_state, aux_state_io[0]) for in_state in in_states]
    )  # inputs |ψ> ⊗ |0> with zero on aux wire

    out_states_expected = np.eye(2**n)
    out_states_expected = np.array(
        [np.kron(target_U @ in_state, aux_state_io[0]) for in_state in in_states]
    )

    success = True
    for in_state_zero, out_state_expected in zip(in_states_zero, out_states_expected):
        out_states = qml.snapshots(qnode)(in_state_zero, **kwargs)[
            "state"
        ]  # when there is a measurement, there are multiple states

        # check if there are multiple out states
        if len(out_states) > 1 and not len(out_states) == 2 ** (n_tot):
            # check that all out_states are the same and circuit decomp is deterministic
            if not all(np.allclose(out_state_i, out_states[0]) for out_state_i in out_states):
                print(
                    f"not all out_states the same: {[state2bin(out_state) for out_state in out_states]}"
                )
                success = False
        else:
            out_states = [out_states]

        if not np.allclose(out_states[0], out_state_expected):
            print(
                f"missmatch {state2bin(out_states[0], n_tot)} and {state2bin(out_state_expected, n_tot)}"
            )
            success = False

    return success
