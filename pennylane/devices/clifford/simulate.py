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
from pennylane.wires import Wires
from pennylane.measurements.expval import ExpectationMP

from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples

from .qubit.simulate import INTERFACE_TO_LIKE

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
    "SX": "SX",
    "CNOT": "CNOT",
    "SWAP": "SWAP",
    "ISWAP": "ISWAP",
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

def measure_final_state(circuit, state, is_state_batched, rng=None, prng_key=None) -> Result:
    """
    Perform the measurements required by the circuit on the provided state.

    This is an internal function that will be called by the successor to ``default.qubit``.

    Args:
        circuit (.QuantumScript): The single circuit to simulate
        state (TensorLike): The state to perform measurement on
        is_state_batched (bool): Whether the state has a batch dimension or not.
        rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
            If no value is provided, a default RNG will be used.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is
            the key to the JAX pseudo random number generator. Only for simulation using JAX.
            If None, the default ``sample_state`` function and a ``numpy.random.default_rng``
            will be for sampling.

    Returns:
        Tuple[TensorLike]: The measurement results
    """
    circuit = circuit.map_to_standard_wires()

    if not circuit.shots:
        # analytic case

        if len(circuit.measurements) == 1:
            return measure(circuit.measurements[0], state, is_state_batched=is_state_batched)

        return tuple(
            measure(mp, state, is_state_batched=is_state_batched) for mp in circuit.measurements
        )

    # finite-shot case

    rng = default_rng(rng)
    results = measure_with_samples(
        circuit.measurements,
        state,
        shots=circuit.shots,
        is_state_batched=is_state_batched,
        rng=rng,
        prng_key=prng_key,
    )

    if len(circuit.measurements) == 1:
        if circuit.shots.has_partitioned_shots:
            return tuple(res[0] for res in results)

        return results[0]

    return results


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
    stim_ct = stim.Circuit()

    prep = None
    if len(circuit) > 0 and isinstance(circuit[0], qml.operation.StatePrepBase):
        prep = circuit[0]

    initial_state = create_initial_state(circuit.op_wires, prep, like=INTERFACE_TO_LIKE[interface])
    initial_tableau = stim.Tableau.from_state_vector(initial_state)

    # initial state is batched only if the state preparation (if it exists) is batched
    is_state_batched = bool(prep and prep.batch_size is not None)
    for op in circuit.operations[bool(prep) :]:
        gate, wires = _GATE_OPERATIONS[op.name], op.wires
        if gate is not None:
            stim_ct.append(gate, *wires)
        else:
            if op.name == "GlobalPhase":
                pass
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

    circ_meas = []
    for meas in circuit.measurements:
        if isinstance(meas, ExpectationMP):
            if isinstance(meas.obs, qml.operation.Tensor):
                expec = ''.join([_GATE_OPERATIONS[name] for name in meas.obs.name])
                pauli = stim._stim_polyfill.PauliString(expec)
                circ_meas.append(pauli)
        else:
            pass

    tableau_simulator = stim.TableauSimulator()
    if prep:
        tableau_simulator.do_tableau(initial_tableau)
    tableau_simulator.do_circuit(stim_ct)

    if not circuit.shots:
        res = tuple(tableau_simulator.measure_observable(meas) for meas in circ_meas)
    else:
        res = tuple()

    #state, is_state_batched = get_final_state(circuit, debugger=debugger, interface=interface)
    return res
