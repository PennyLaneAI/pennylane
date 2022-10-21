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
# pylint: disable= no-value-for-parameter
import pennylane as qml

from pennylane import numpy as np


@qml.qfunc_transform
def append_gate(tape, params, gates):
    """Append parameterized gates to an existing tape.

    Args:
        tape (QuantumTape): quantum tape to transform by adding gates
        params (array[float]): parameters of the gates to be added
        gates (list[Operator]): list of the gates to be added
    """
    for o in tape.operations:
        qml.apply(o)

    for i, g in enumerate(gates):
        g.data[0] = params[i]
        qml.apply(g)

    for m in tape.measurements:
        qml.apply(m)


class AdaptiveOptimizer:
    """Adaptive optimizer.

    Quantum circuits can be built by adding gates
    `adaptively <https://www.nature.com/articles/s41467-019-10988-2>`_. The adaptive optimizer
    implements an algorithm that grows and optimizes an input quantum circuit by selecting and
    adding gates from a user-defined collection of operators. The algorithm starts with adding all
    the gates to the circuit and computing the circuit gradients with respect to the gate
    parameters. The algorithm then retains the gate which has the largest gradient and optimizes its
    parameter. The processes of growing the circuit is repeated until the computed gradients
    converge within a given threshold. The adaptive optimizer can be used to implement algorithms
    such as `ADAPT-VQE <https://www.nature.com/articles/s41467-019-10988-2>`_.

    Args:
        paramopt_steps (float): number of steps for optimizing the parameter of a selected gate


    **Examples:**

    This examples shows an implementation of the
    `ADAPT-VQE <https://www.nature.com/articles/s41467-019-10988-2>`_ algorithm for building an
    adaptive circuit for the :math:`H_3^+` cation.

    >>> import pennylane as qml
    >>> from pennylane import numpy as np

    The molecule is defined and the Hamiltonian is computed with:

    >>> symbols = ["H", "H", "H"]
    >>> geometry = np.array([[0.01076341, 0.04449877, 0.0],
    ...                      [0.98729513, 1.63059094, 0.0],
    ...                      [1.87262415, -0.00815842, 0.0]], requires_grad=False)
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge = 1)

    The collection of gates to grow the circuit adaptively contntains aa singles and doubles
    excitations:

    >>> n_electrons = 2
    >>> singles, doubles = qml.qchem.excitations(n_electrons, qubits)
    >>> operator_pool = doubles + singles

    An initial circuit preparing the Hartree-Fock state and returning the expectation value of the
    Hamiltonian is defined:

    >>> hf_state = qml.qchem.hf_state(n_electrons, qubits)
    >>> dev = qml.device("default.qubit", wires=qubits)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.BasisState(hf_state, wires=range(qubits))
    ...     return qml.expval(H)

    The optimizer is instantiated and the circuit is optimized adaptively:

    >>> opt = AdaptiveOptimizer()
    >>> params = np.zeros(len(operator_pool))
    >>> operator_pool = pool_gate(params, operator_pool)
    >>> for i in range(len(operator_pool)):
    ...     circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    ...     print('Energy:', energy)
    ...     print(qml.draw(circuit)())
    ...     print('Gradient max:', gradient)
    ...     print()
    ...     if gradient < 1e-3:
    ...         break

    ```pycon
    Energy: -1.2613705937615631
    0: ─╭BasisState(M0)─╭G²(0.20)─┤ ╭<𝓗>
    1: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
    2: ─├BasisState(M0)─│─────────┤ ├<𝓗>
    3: ─├BasisState(M0)─│─────────┤ ├<𝓗>
    4: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
    5: ─╰BasisState(M0)─╰G²(0.20)─┤ ╰<𝓗>
    Gradient max: 0.14399872776724146

    Energy: -1.2743941385501283
    0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)─┤ ╭<𝓗>
    1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─┤ ├<𝓗>
    2: ─├BasisState(M0)─│─────────├G²(0.19)─┤ ├<𝓗>
    3: ─├BasisState(M0)─│─────────╰G²(0.19)─┤ ├<𝓗>
    4: ─├BasisState(M0)─├G²(0.20)───────────┤ ├<𝓗>
    5: ─╰BasisState(M0)─╰G²(0.20)───────────┤ ╰<𝓗>
    Gradient max: 0.13493495624216287

    Energy: -1.2743974223749222
    0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)───────────┤ ╭<𝓗>
    1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─╭G(-0.00)─┤ ├<𝓗>
    2: ─├BasisState(M0)─│─────────├G²(0.19)─│─────────┤ ├<𝓗>
    3: ─├BasisState(M0)─│─────────╰G²(0.19)─╰G(-0.00)─┤ ├<𝓗>
    4: ─├BasisState(M0)─├G²(0.20)─────────────────────┤ ├<𝓗>
    5: ─╰BasisState(M0)─╰G²(0.20)─────────────────────┤ ╰<𝓗>
    Gradient max: 0.0004084175685509349
    ```


    """

    def __init__(self, paramopt_steps=10):
        self.paramopt_steps = paramopt_steps

    @staticmethod
    def _circuit(params, gates, initial_circuit):
        """Append parameterized gates to an existing circuit.

        Args:
            params (array[float]): parameters of the gates to be added
            gates (list[Operator]): list of the gates to be added
            initial_circuit (function): user defined circuit that returns an expectation value

        Returns:
            function: user defined circuit with appended gates
        """
        final_circuit = append_gate(params, gates)(initial_circuit)

        return final_circuit()

    def step(self, circuit, operator_pool):
        r"""Update the circuit with one step of the optimizer.

        Args:
            circuit (.QNode): user defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization

        Returns:
           .QNode: the optimized circuit
        """
        return self.step_and_cost(circuit, operator_pool)[0]

    def step_and_cost(self, circuit, operator_pool, drain_pool=False):
        r"""Update the circuit with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            circuit (.QNode): user defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            drain_pool (bool): flag to remove selected gates from the operator_pool

        Returns:
            tuple[.QNode, float]: the optimized circuit and the objective function output prior
            to the step
        """
        cost = circuit()
        device = circuit.device

        if drain_pool:
            repeated_gates = [
                gate
                for gate in operator_pool
                for operation in circuit.tape.operations
                if qml.equal(gate, operation, rtol=float("inf"))
            ]
            for gate in repeated_gates:
                operator_pool.remove(gate)

        params = np.array([gate.parameters[0] for gate in operator_pool], requires_grad=True)
        qnode = qml.QNode(self._circuit, device)
        grads = qml.grad(qnode)(params, gates=operator_pool, initial_circuit=circuit.func)

        selected_gate = [operator_pool[np.argmax(abs(grads))]]
        optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
        params = np.zeros(1)
        for n in range(10):
            params, cost = optimizer.step_and_cost(
                qnode, params, gates=selected_gate, initial_circuit=circuit.func
            )
        circuit = append_gate(params, selected_gate)(circuit.func)

        qnode = qml.QNode(circuit, device)

        return qnode, cost, max(abs(qml.math.toarray(grads)))
