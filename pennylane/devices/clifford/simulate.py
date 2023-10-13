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
"""Simulate a quantum script."""
# pylint: disable=protected-access
from numpy.random import default_rng

import pennylane as qml
from pennylane.typing import Result
from pennylane.measurements.expval import ExpectationMP

# from .measure import measure
# from .sampling import measure_with_samples

from pennylane.devices.qubit.initialize_state import create_initial_state
from pennylane.devices.qubit.simulate import INTERFACE_TO_LIKE

_GATE_OPERATIONS = {
    "Identity": "I",
    "Snapshot": None,
    "BasisState": None,
    "StatePrep": None,
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Hadamard": "H",
    "S": "S",
    "Adjoint(S)": "S_DAG",
    "SX": "SX",
    "Adjoint(SX)": "SX_DAG",
    "CNOT": "CNOT",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
    "Adjoint(ISWAP)": "ISWAP_DAG",
    "CY": "CY",
    "CZ": "CZ",
    "GlobalPhase": None,
}


def _import_stim():
    """Import stim."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import, multiple-imports
        import stim
    except ImportError as Error:
        raise ImportError(
            "This feature requires stim, a fast stabilizer circuit simulator."
            "It can be installed with: pip install stim."
        ) from Error
    return stim


# def measure_final_state(circuit, state, is_state_batched, rng=None, prng_key=None) -> Result:
#     """
#     Perform the measurements required by the circuit on the provided state.

#     This is an internal function that will be called by the successor to ``default.qubit``.

#     Args:
#         circuit (.QuantumScript): The single circuit to simulate
#         state (TensorLike): The state to perform measurement on
#         is_state_batched (bool): Whether the state has a batch dimension or not.
#         rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
#             seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
#             If no value is provided, a default RNG will be used.
#         prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
#             the key to the JAX pseudo random number generator. Only for simulation using JAX.
#             If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
#             will be for sampling.

#     Returns:
#         Tuple[TensorLike]: The measurement results
#     """
#     circuit = circuit.map_to_standard_wires()

#     if not circuit.shots:
#         # analytic case

#         if len(circuit.measurements) == 1:
#             return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

#         return tuple(
#             measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
#         )

#     # finite-shot case

#     rng = default_rng(rng)
#     results = measure_with_samples(
#         circuit.measurements,
#         state,
#         shots=circuit.shots,
#         is_state_batched=is_state_batched,
#         rng=rng,
#         prng_key=prng_key,
#     )

#     if len(circuit.measurements) == 1:
#         if circuit.shots.has_partitioned_shots:
#             return tuple(res[0] for res in results)

#         return results[0]


#     return results
def get_final_state(c, debugger=None):
    pass


def _stabilizer_vn_entropy(stabilizer, wires, log_base=None):
    r"""Computing entanglement entropy for a subsytem described by wires A -
    :math:`S_A = \text{rank}(\text{projection}_A {G}) - N_{A}`, where G is the set
    of stabilizers
    """

    num_qubits = qml.math.shape(stabilizer)[0]

    if num_qubits == len(wires):
        return 0.0

    pauli_dict = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    terms = [
        qml.pauli.PauliWord({idx: pauli_dict[ele] for idx, ele in enumerate(row)})
        for row in stabilizer
    ]

    mat = qml.pauli.utils._binary_matrix_from_pws(terms, num_qubits)

    cut_mat = qml.math.hstack(
        (
            mat[:, num_qubits:][:, wires],
            mat[:, :num_qubits][:, wires],
        )
    )
    rank = qml.math.sum(qml.math.any(qml.qchem.tapering._reduced_row_echelon(cut_mat), axis=1))

    entropy = qml.math.log(2) * (rank - len(wires))

    if log_base is None:
        return entropy

    return entropy / qml.math.log(log_base)


# pylint: disable=unidiomatic-typecheck, unused-argument
def simulate(
    circuit: qml.tape.QuantumScript, rng=None, prng_key=None, debugger=None, interface=None
) -> Result:
    """Simulate a single quantum script.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (QuantumTape): The single circuit to simulate
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. If None, a random key will be
            generated. Only for simulation using JAX.
        debugger (_Debugger): The debugger to use
        interface (str): The machine learning interface to create the initial state with

    Returns:
        tuple(TensorLike): The results of the simulation

    Note that this function can return measurements for non-commuting observables simultaneously.

    This function assumes that all operations provide matrices.

    >>> qs = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))])
    >>> simulate(qs)
    (0.36235775447667357,
    tensor([0.68117888, 0.        , 0.31882112, 0.        ], requires_grad=True))

    """
    stim = _import_stim()

    circuit = circuit.map_to_standard_wires()

    if circuit.shots:
        raise NotImplementedError(
            "default.clifford currently doesn't support computation with shots."
        )

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    if is_state_batched:
        raise NotImplementedError("Clifford simulator doesn't support batching.")

    stim_ct = stim.Circuit()
    initial_state = create_initial_state(circuit.op_wires, prep, like=INTERFACE_TO_LIKE[interface])
    initial_tableau = stim.Tableau.from_state_vector(
        qml.math.reshape(initial_state, (1, -1))[0], endian="big"
    )

    global_phase_ops = []
    for op in circuit.operations[bool(prep) :]:
        gate, wires = _GATE_OPERATIONS[op.name], op.wires
        if gate is not None:
            stim_ct.append(gate, wires)
        else:
            if op.name == "GlobalPhase":
                global_phase_ops.append(op)
            elif op.name == "Snapshot":
                state = stim.Tableau.from_circuit(stim_ct).to_state_vector()
                if debugger is not None and debugger.active:
                    flat_state = qml.math.flatten(state)
                if op.tag:
                    debugger.snapshots[op.tag] = flat_state
                else:
                    debugger.snapshots[len(debugger.snapshots)] = flat_state
            else:
                pass

    tableau_simulator = stim.TableauSimulator()
    if prep:
        tableau_simulator.do_tableau(initial_tableau, circuit.wires)
    tableau_simulator.do_circuit(stim_ct)

    res = []
    for meas in circuit.measurements:
        # Computing statevector via tableaus
        if type(meas) is qml.measurements.StateMP:
            state_vector = qml.math.array(
                tableau_simulator.state_vector(endian="big"), like=INTERFACE_TO_LIKE[interface]
            )
            res.append(state_vector)

        # Computing density matrix via tableaus
        elif type(meas) is qml.measurements.DensityMatrixMP:
            state_vector = qml.math.array(
                tableau_simulator.state_vector(endian="big"), like=INTERFACE_TO_LIKE[interface]
            )
            density_matrix = qml.math.einsum("i, j->ij", state_vector, state_vector)
            res.append(density_matrix)

        # Computing purity via tableaus
        elif type(meas) is qml.measurements.PurityMP:
            res.append(1.0)

        # Computing entropy via tableaus
        elif type(meas) is qml.measurements.VnEntropyMP:
            tableau = tableau_simulator.current_inverse_tableau() ** -1
            zs = qml.math.array([tableau.z_output(k) for k in range(len(circuit.wires))])
            res.append(_stabilizer_vn_entropy(zs, list(meas.wires)))

        # Computing entropy via tableaus
        elif type(meas) is qml.measurements.MutualInfoMP:
            tableau = tableau_simulator.current_inverse_tableau() ** -1
            zs = qml.math.array([tableau.z_output(k) for k in range(len(circuit.wires))])
            indices0, indices1 = list(meas._wires[0]), list(meas._wires[1])
            res.append(_stabilizer_vn_entropy(zs, indices0) + _stabilizer_vn_entropy(zs, indices1))

        # Computing expectation values via measurement
        elif isinstance(meas, ExpectationMP):
            # Case for simple Pauli terms
            if (
                isinstance(meas.obs, qml.PauliZ)
                or isinstance(meas.obs, qml.PauliX)
                or isinstance(meas.obs, qml.PauliY)
            ):
                pauli = stim._stim_polyfill.PauliString(_GATE_OPERATIONS[meas.obs.name])
                res.append(tableau_simulator.peek_observable_expectation(pauli))

            # Case for simple Pauli tensor
            elif isinstance(meas.obs, qml.operation.Tensor):
                expec = "".join([_GATE_OPERATIONS[name] for name in meas.obs.name])
                pauli = stim._stim_polyfill.PauliString(expec)
                res.append(tableau_simulator.peek_observable_expectation(pauli))

            # Case for a Hamiltonian
            elif isinstance(meas.obs, qml.Hamiltonian):
                coeffs, obs = meas.obs.terms()
                expecs = qml.math.zeros_like(coeffs)
                for idx, ob in enumerate(obs):
                    expec = "".join([_GATE_OPERATIONS[name] for name in ob.name])
                    pauli = stim._stim_polyfill.PauliString(expec)
                    expecs[idx] = tableau_simulator.peek_observable_expectation(pauli)
                res.append(qml.math.sum(coeffs * expecs))

            # Add support for more case when the time is right
            else:
                raise NotImplementedError(f"default.clifford doesn't support {meas} at the moment.")

    # state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
    return tuple(res)
