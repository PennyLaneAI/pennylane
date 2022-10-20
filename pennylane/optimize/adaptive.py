# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adaptive optimizer"""

import pennylane as qml

from pennylane import numpy as np


@qml.qfunc_transform
def append_gate(tape, params, gates):
    """Append parameterized gates to an existing circuit.

    Args:
        tape (QuantumTape): quantum tape to transform by adding gates
        params (array[float]): parameters of the gates to be added
        gates (list[Operator]): list of the gates to be added
    """
    if len(params) == 1:
        gates = [gates]

    for o in tape.operations:
        qml.apply(o)

    for i, g in enumerate(gates):
        g.data[0] = params[i]
        qml.apply(g)

    for m in tape.measurements:
        qml.apply(m)


class AdaptiveOptimizer:
    """Adaptive optimizer.

    """
    def __init__(self, tol=1e-5):
        self.tol = tol

    def _equal(self, a, b):
        if a.name == b.name and a.wires == b.wires:
            return True
        return False

    def _circuit(self, params, gates, initial_circuit):

        final_circuit = append_gate(params, gates)(initial_circuit)

        return final_circuit()

    def step_and_cost(self, circuit, pool, drain=False):

        device = circuit.device

        energy = circuit()

        if drain:
            for gate in pool:
                for operation in circuit.tape.operations:
                    if self._equal(gate, operation):
                        pool.remove(gate)

        params = np.array([x.parameters[0] for x in pool], requires_grad=True)

        _qnode = qml.QNode(self._circuit, device)

        grads = qml.grad(_qnode)(params, gates=pool, initial_circuit=circuit.func)

        if np.max(abs(grads)) > self.tol:

            selected_gate = pool[np.argmax(abs(grads))]

            optimizer = qml.GradientDescentOptimizer(stepsize=0.5)

            params = np.zeros(1)

            for n in range(10):
                params, energy = optimizer.step_and_cost(
                    _qnode, params, gates=selected_gate, initial_circuit=circuit.func
                )

            circuit = append_gate(params, selected_gate)(circuit.func)

        qnode = qml.QNode(circuit, device)

        return qnode, energy
