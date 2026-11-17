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

import pytest

import pennylane as qp
from pennylane import numpy as np

symbols = ["H", "H", "H"]
geometry = np.array(
    [[0.01076341, 0.04449877, 0.0], [0.98729513, 1.63059094, 0.0], [1.87262415, -0.00815842, 0.0]],
    requires_grad=False,
)
H, qubits = qp.qchem.molecular_hamiltonian(symbols, geometry, charge=1)

hf_state = np.array([1, 1, 0, 0, 0, 0])
dev = qp.device("default.qubit", wires=qubits)

energy_h3p_hf = -1.2465499384199534


@qp.transforms.convert_to_numpy_parameters
@qp.qnode(dev, interface="numpy")
def initial_circuit():
    qp.BasisState(hf_state, wires=range(qubits))
    return qp.expval(H)


@pytest.mark.parametrize(
    "circuit, params, gates, energy_ref",
    [
        (
            initial_circuit,
            np.array([0.0]),
            [qp.DoubleExcitation(0.0, [0, 1, 2, 3])],
            energy_h3p_hf,
        ),
    ],
)
def test_private_circuit(circuit, params, gates, energy_ref):
    r"""Test that _circuit returns the correct output."""
    private_circuit = qp.AdaptiveOptimizer._circuit  # pylint: disable=protected-access
    qnode = qp.QNode(private_circuit, dev, interface="numpy")
    qnode = qp.transforms.convert_to_numpy_parameters(qnode)
    energy = qnode(params, gates, circuit.func)
    assert np.allclose(energy, energy_ref)


@pytest.mark.parametrize(
    "circuit",
    [initial_circuit],
)
def test_append_gate(circuit):
    """Test that append_gate properly adds a gate to a circuit."""

    param = np.array([0.0])
    gate = qp.DoubleExcitation(np.array(0.0), wires=[0, 1, 2, 3])

    final_circuit = qp.optimize.adaptive.append_gate(circuit.func, param, [gate])
    qnode = qp.QNode(final_circuit, dev)
    tape = qp.workflow.construct_tape(qnode)()
    qp.assert_equal(tape.operations[-1], gate)

    final_circuit, fn = qp.optimize.adaptive.append_gate(tape, param, [gate])

    assert isinstance(final_circuit, list)
    assert isinstance(fn(final_circuit), qp.tape.QuantumScript)
