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
# pylint: disable= no-value-for-parameter, protected-access
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
    r"""Optimizer for building fully trained quantum circuits by adding gates adaptively.

    Quantum circuits can be built by adding gates
    `adaptively <https://www.nature.com/articles/s41467-019-10988-2>`_. The adaptive optimizer
    implements an algorithm that grows and optimizes an input quantum circuit by selecting and
    adding gates from a user-defined collection of operators. The algorithm starts by adding all
    the gates to the circuit and computing the circuit gradients with respect to the gate
    parameters. The algorithm then retains the gate which has the largest gradient and optimizes its
    parameter. The process of growing the circuit is repeated until the computed gradients
    converge to zero within a given threshold. The optimizer returns the fully trained and
    adaptively-built circuit. The adaptive optimizer can be used to implement
    algorithms such as `ADAPT-VQE <https://www.nature.com/articles/s41467-019-10988-2>`_.

    Args:
        param_steps (int): number of steps for optimizing the parameter of a selected gate
        stepsize (float): step size for optimizing the parameter of a selected gate

    **Example**

    This examples shows an implementation of the
    `ADAPT-VQE <https://www.nature.com/articles/s41467-019-10988-2>`_ algorithm for building an
    adaptive circuit for the :math:`\text{H}_3^+` cation.

    >>> import pennylane as qml
    >>> from pennylane import numpy as np

    The molecule is defined and the Hamiltonian is computed with:

    >>> symbols = ["H", "H", "H"]
    >>> geometry = np.array([[0.01076341, 0.04449877, 0.0],
    ...                      [0.98729513, 1.63059094, 0.0],
    ...                      [1.87262415, -0.00815842, 0.0]], requires_grad=False)
    >>> H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge = 1)

    The collection of gates to grow the circuit adaptively contains all single and double
    excitations:

    >>> n_electrons = 2
    >>> singles, doubles = qml.qchem.excitations(n_electrons, qubits)
    >>> singles_excitations = [qml.SingleExcitation(0.0, x) for x in singles]
    >>> doubles_excitations = [qml.DoubleExcitation(0.0, x) for x in doubles]
    >>> operator_pool = doubles_excitations + singles_excitations

    An initial circuit preparing the Hartree-Fock state and returning the expectation value of the
    Hamiltonian is defined:

    >>> hf_state = qml.qchem.hf_state(n_electrons, qubits)
    >>> dev = qml.device("default.qubit", wires=qubits)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.BasisState(hf_state, wires=range(qubits))
    ...     return qml.expval(H)

    The optimizer is instantiated and the circuit is created and optimized adaptively:

    >>> opt = AdaptiveOptimizer()
    >>> for i in range(len(operator_pool)):
    ...     circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    ...     print('Energy:', energy)
    ...     print(qml.draw(circuit)())
    ...     print('Largest Gradient:', gradient)
    ...     print()
    ...     if gradient < 1e-3:
    ...         break

    .. code-block :: pycon

        Energy: -1.246549938420637
        0: ─╭BasisState(M0)─╭G²(0.20)─┤ ╭<𝓗>
        1: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
        2: ─├BasisState(M0)─│─────────┤ ├<𝓗>
        3: ─├BasisState(M0)─│─────────┤ ├<𝓗>
        4: ─├BasisState(M0)─├G²(0.20)─┤ ├<𝓗>
        5: ─╰BasisState(M0)─╰G²(0.20)─┤ ╰<𝓗>
        Largest Gradient: 0.14399872776755085

        Energy: -1.2613740231529604
        0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)─┤ ╭<𝓗>
        1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─┤ ├<𝓗>
        2: ─├BasisState(M0)─│─────────├G²(0.19)─┤ ├<𝓗>
        3: ─├BasisState(M0)─│─────────╰G²(0.19)─┤ ├<𝓗>
        4: ─├BasisState(M0)─├G²(0.20)───────────┤ ├<𝓗>
        5: ─╰BasisState(M0)─╰G²(0.20)───────────┤ ╰<𝓗>
        Largest Gradient: 0.1349349562423238

        Energy: -1.2743971719780331
        0: ─╭BasisState(M0)─╭G²(0.20)─╭G²(0.19)──────────┤ ╭<𝓗>
        1: ─├BasisState(M0)─├G²(0.20)─├G²(0.19)─╭G(0.00)─┤ ├<𝓗>
        2: ─├BasisState(M0)─│─────────├G²(0.19)─│────────┤ ├<𝓗>
        3: ─├BasisState(M0)─│─────────╰G²(0.19)─╰G(0.00)─┤ ├<𝓗>
        4: ─├BasisState(M0)─├G²(0.20)────────────────────┤ ├<𝓗>
        5: ─╰BasisState(M0)─╰G²(0.20)────────────────────┤ ╰<𝓗>
        Largest Gradient: 0.00040841755397108586
    """

    def __init__(self, param_steps=10, stepsize=0.5):
        self.param_steps = param_steps
        self.stepsize = stepsize

    @staticmethod
    def _circuit(params, gates, initial_circuit):
        """Append parameterized gates to an existing circuit.

        Args:
            params (array[float]): parameters of the gates to be added
            gates (list[Operator]): list of the gates to be added
            initial_circuit (function): user-defined circuit that returns an expectation value

        Returns:
            function: user-defined circuit with appended gates
        """
        final_circuit = append_gate(params, gates)(initial_circuit)

        return final_circuit()

    def step(self, circuit, operator_pool, params_zero=True):
        r"""Update the circuit with one step of the optimizer.

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
           .QNode: the optimized circuit
        """
        return self.step_and_cost(circuit, operator_pool, params_zero=params_zero)[0]

    def step_and_cost(self, circuit, operator_pool, drain_pool=False, params_zero=True):
        r"""Update the circuit with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            drain_pool (bool): flag to remove selected gates from the operator pool
            params_zero (bool): flag to initiate circuit parameters at zero

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

        selected_gates = [operator_pool[np.argmax(abs(grads))]]
        optimizer = qml.GradientDescentOptimizer(stepsize=self.stepsize)

        if params_zero:
            params = np.zeros(len(selected_gates))
        else:
            params = np.array(
                [gate.parameters[0]._value for gate in selected_gates], requires_grad=True
            )

        for _ in range(self.param_steps):
            params, _ = optimizer.step_and_cost(
                qnode, params, gates=selected_gates, initial_circuit=circuit.func
            )

        circuit = append_gate(params, selected_gates)(circuit.func)

        qnode = qml.QNode(circuit, device)

        return qnode, cost, max(abs(qml.math.toarray(grads)))
