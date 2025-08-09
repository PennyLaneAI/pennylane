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
Unit tests for the ``AdaptiveOptimizer``.
"""
import copy

import pytest

import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H", "H"]
geometry = np.array(
    [[0.01076341, 0.04449877, 0.0], [0.98729513, 1.63059094, 0.0], [1.87262415, -0.00815842, 0.0]],
    requires_grad=False,
)
H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=1)

hf_state = np.array([1, 1, 0, 0, 0, 0])
dev = qml.device("default.qubit", wires=qubits)

energy_h3p_hf = -1.2465499384199534
energy_first_step = -1.2613740231522113
energy_h3p = -1.274397672040264


@qml.qnode(dev)
def initial_circuit():
    qml.BasisState(hf_state, wires=range(qubits))
    return qml.expval(H)


@pytest.mark.parametrize(
    "circuit, params, gates, energy_ref",
    [
        (
            initial_circuit,
            np.array([0.0]),
            [qml.DoubleExcitation(0.0, [0, 1, 2, 3])],
            energy_h3p_hf,
        ),
    ],
)
def test_private_circuit(circuit, params, gates, energy_ref):
    r"""Test that _circuit returns the correct output."""
    # pylint: disable=protected-access
    qnode = qml.QNode(qml.AdaptiveOptimizer._circuit, dev)
    energy = qnode(params, gates, circuit.func)
    assert np.allclose(energy, energy_ref)


pool_exc = [
    qml.DoubleExcitation(np.array(0.0), wires=[0, 1, 2, 3]),
    qml.DoubleExcitation(np.array(0.0), wires=[0, 1, 2, 5]),
    qml.DoubleExcitation(np.array(0.0), wires=[0, 1, 3, 4]),
    qml.DoubleExcitation(np.array(0.0), wires=[0, 1, 4, 5]),
    qml.SingleExcitation(np.array(0.0), wires=[0, 2]),
    qml.SingleExcitation(np.array(0.0), wires=[0, 4]),
    qml.SingleExcitation(np.array(0.0), wires=[1, 3]),
    qml.SingleExcitation(np.array(0.0), wires=[1, 5]),
]


@pytest.mark.parametrize(
    "circuit, energy_ref, pool",
    [
        (initial_circuit, energy_first_step, pool_exc),
    ],
)
def test_step(circuit, energy_ref, pool):
    """Test that step function returns the correct cost."""
    opt = qml.AdaptiveOptimizer()
    circuit = opt.step(circuit, pool)
    energy = circuit()
    assert np.allclose(energy, energy_ref)


@pytest.mark.slow
@pytest.mark.parametrize(
    "circuit, energy_ref, pool",
    [
        (initial_circuit, energy_h3p, pool_exc),
    ],
)
def test_step_and_cost_drain(circuit, energy_ref, pool):
    """Test that step_and_cost function returns the correct results when drain_pool is True."""
    opt = qml.AdaptiveOptimizer()
    for _ in range(4):
        circuit, energy, _ = opt.step_and_cost(circuit, copy.copy(pool), drain_pool=True)

    tape = qml.workflow.construct_tape(circuit)()
    selected_excitations = [op.wires for op in tape.operations[1:]]

    assert np.allclose(energy, energy_ref)
    # assert that the operator pool is drained, no repeated gates in the circuit
    assert len(set(selected_excitations)) == len(selected_excitations)


@pytest.mark.slow
@pytest.mark.parametrize(
    "circuit, energy_ref, pool",
    [
        (initial_circuit, energy_h3p, pool_exc),
    ],
)
def test_step_and_cost_nodrain(circuit, energy_ref, pool):
    """Test that step_and_cost function returns the correct results when drain_pool is False."""
    opt = qml.AdaptiveOptimizer()
    for _ in range(4):
        circuit, energy, _ = opt.step_and_cost(circuit, pool, drain_pool=False)

    tape = qml.workflow.construct_tape(circuit)()
    selected_excitations = [op.wires for op in tape.operations[1:]]

    assert np.allclose(energy, energy_ref, rtol=1e-4)
    # assert that the operator pool is not drained, there are repeated gates in the circuit
    assert len(set(selected_excitations)) < len(selected_excitations)


@pytest.mark.parametrize("circuit", [initial_circuit])
def test_largest_gradient(circuit):
    """Test that step function selects the gate with the largest gradient."""

    opt = qml.AdaptiveOptimizer()
    circuit = opt.step(circuit, pool_exc)
    tape = qml.workflow.construct_tape(circuit)()
    selected_gate = tape.operations[-1]

    #  the reference gate is obtained manually
    assert selected_gate.name == "DoubleExcitation"
    assert tape.operations[-1].wires == qml.wires.Wires([0, 1, 4, 5])


@pytest.mark.parametrize(
    "circuit",
    [initial_circuit],
)
def test_append_gate(circuit):
    """Test that append_gate properly adds a gate to a circuit."""

    param = np.array([0.0])
    gate = qml.DoubleExcitation(np.array(0.0), wires=[0, 1, 2, 3])

    final_circuit = qml.optimize.adaptive.append_gate(circuit.func, param, [gate])
    qnode = qml.QNode(final_circuit, dev)
    tape = qml.workflow.construct_tape(qnode)()
    qml.assert_equal(tape.operations[-1], gate)

    final_circuit, fn = qml.optimize.adaptive.append_gate(tape, param, [gate])

    assert isinstance(final_circuit, list)
    assert isinstance(fn(final_circuit), qml.tape.QuantumScript)


@qml.qnode(dev)
def qubit_rotation_circuit():
    return qml.expval(qml.PauliZ(0))


@pytest.mark.parametrize(
    "circuit",
    [qubit_rotation_circuit],
)
def test_qubit_rotation(circuit):
    """Test that step function returns correct results for a qubit rotation circuit."""

    pool = [qml.RX(np.array([1.0]), wires=0), qml.RZ(np.array([1.0]), wires=0)]
    opt = qml.AdaptiveOptimizer(param_steps=20)
    circuit = opt.step(circuit, pool, params_zero=False)
    expval = circuit()
    tape = qml.workflow.construct_tape(circuit)()

    #  rotation around X with np.pi gives expval(Z) = -1
    assert np.allclose(expval, -1)
    qml.assert_equal(tape.operations[-1], qml.RX(np.array([np.pi]), wires=0))


@pytest.mark.parametrize(
    "interface, diff_method",
    [
        ("autograd", "parameter-shift"),
    ],
)
def test_circuit_args(interface, diff_method):
    """Test that step_and_cost function uses the correct arguments of the initial circuit."""

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit():
        qml.BasisState(np.array([1, 1, 0, 0]), wires=range(4))
        return qml.expval(qml.PauliZ(0))

    opt = qml.AdaptiveOptimizer()
    circuit, _, _ = opt.step_and_cost(circuit, pool_exc)

    assert circuit.interface == interface
    assert circuit.diff_method == diff_method
