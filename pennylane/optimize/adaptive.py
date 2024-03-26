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
import copy
from typing import Sequence, Callable

# pylint: disable= no-value-for-parameter, protected-access, not-callable
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.tape import QuantumTape
from pennylane import transform


@transform
def append_gate(tape: QuantumTape, params, gates) -> (Sequence[QuantumTape], Callable):
    """Append parameterized gates to an existing tape.

    Args:
        tape (QuantumTape or QNode or Callable): quantum circuit to transform by adding gates
        params (array[float]): parameters of the gates to be added
        gates (list[Operator]): list of the gates to be added

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    """
    new_operations = []

    for i, g in enumerate(gates):
        g = copy.copy(g)
        new_params = (params[i], *g.data[1:])
        g.data = new_params
        new_operations.append(g)

    new_tape = type(tape)(tape.operations + new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    return [new_tape], null_postprocessing


class AdaptiveOptimizer:
    r"""Optimizer for building fully trained quantum circuits by adding gates adaptively.

    Quantum circuits can be built by adding gates
    `adaptively <https://www.nature.com/articles/s41467-019-10988-2>`_. The adaptive optimizer
    implements an algorithm that grows and optimizes an input quantum circuit by selecting and
    adding gates from a user-defined collection of operators. The algorithm starts by adding all
    the gates to the circuit and computing the circuit gradients with respect to the gate
    parameters. The algorithm then retains the gate which has the largest gradient and optimizes its
    parameter. The process of growing the circuit can be repeated until the computed gradients
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
    >>> from pennylane import numpy as pnp

    The molecule is defined and the Hamiltonian is computed with:

    >>> symbols = ["H", "H", "H"]
    >>> geometry = pnp.array([[0.01076341, 0.04449877, 0.0],
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

    The optimizer is instantiated and then the circuit is created and optimized adaptively:

    >>> opt = AdaptiveOptimizer()
    >>> for i in range(len(operator_pool)):
    ...     circuit, energy, gradient = opt.step_and_cost(circuit, operator_pool, drain_pool=True)
    ...     print('Energy:', energy)
    ...     print(qml.draw(circuit, show_matrices=False)())
    ...     print('Largest Gradient:', gradient)
    ...     print()
    ...     if gradient < 1e-3:
    ...         break

    .. code-block :: pycon

        Energy: -1.2465499384199699
        0: ─╭|Ψ⟩─╭G²(0.20)─┤ ╭<𝓗>
        1: ─├|Ψ⟩─├G²(0.20)─┤ ├<𝓗>
        2: ─├|Ψ⟩─│─────────┤ ├<𝓗>
        3: ─├|Ψ⟩─│─────────┤ ├<𝓗>
        4: ─├|Ψ⟩─├G²(0.20)─┤ ├<𝓗>
        5: ─╰|Ψ⟩─╰G²(0.20)─┤ ╰<𝓗>
        Largest Gradient: 0.1439987277673651

        Energy: -1.2613740231522532
        0: ─╭|Ψ⟩─╭G²(0.20)─╭G²(0.19)─┤ ╭<𝓗>
        1: ─├|Ψ⟩─├G²(0.20)─├G²(0.19)─┤ ├<𝓗>
        2: ─├|Ψ⟩─│─────────├G²(0.19)─┤ ├<𝓗>
        3: ─├|Ψ⟩─│─────────╰G²(0.19)─┤ ├<𝓗>
        4: ─├|Ψ⟩─├G²(0.20)───────────┤ ├<𝓗>
        5: ─╰|Ψ⟩─╰G²(0.20)───────────┤ ╰<𝓗>
        Largest Gradient: 0.13493495624211427

        Energy: -1.2743971719772815
        0: ─╭|Ψ⟩─╭G²(0.20)─╭G²(0.19)──────────┤ ╭<𝓗>
        1: ─├|Ψ⟩─├G²(0.20)─├G²(0.19)─╭G(0.00)─┤ ├<𝓗>
        2: ─├|Ψ⟩─│─────────├G²(0.19)─│────────┤ ├<𝓗>
        3: ─├|Ψ⟩─│─────────╰G²(0.19)─╰G(0.00)─┤ ├<𝓗>
        4: ─├|Ψ⟩─├G²(0.20)────────────────────┤ ├<𝓗>
        5: ─╰|Ψ⟩─╰G²(0.20)────────────────────┤ ╰<𝓗>
        Largest Gradient: 0.0004084175253678331
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
        final_circuit = append_gate(initial_circuit, params, gates)

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
        r"""Update the circuit with one step of the optimizer, return the corresponding
        objective function value prior to the step, and return the maximum gradient

        Args:
            circuit (.QNode): user-defined circuit returning an expectation value
            operator_pool (list[Operator]): list of the gates to be used for adaptive optimization
            drain_pool (bool): flag to remove selected gates from the operator pool
            params_zero (bool): flag to initiate circuit parameters at zero

        Returns:
            tuple[.QNode, float, float]: the optimized circuit, the objective function output prior
            to the step, and the largest gradient
        """
        cost = circuit()
        qnode = copy.copy(circuit)

        if drain_pool:
            operator_pool = [
                gate
                for gate in operator_pool
                if all(
                    gate.name != operation.name or gate.wires != operation.wires
                    for operation in circuit.tape.operations
                )
            ]

        params = pnp.array([gate.parameters[0] for gate in operator_pool], requires_grad=True)
        qnode.func = self._circuit
        grads = qml.grad(qnode)(params, gates=operator_pool, initial_circuit=circuit.func)

        selected_gates = [operator_pool[pnp.argmax(abs(grads))]]
        optimizer = qml.GradientDescentOptimizer(stepsize=self.stepsize)

        if params_zero:
            params = pnp.zeros(len(selected_gates))
        else:
            params = pnp.array([gate.parameters[0] for gate in selected_gates], requires_grad=True)

        for _ in range(self.param_steps):
            params, _ = optimizer.step_and_cost(
                qnode, params, gates=selected_gates, initial_circuit=circuit.func
            )

        qnode.func = append_gate(circuit.func, params, selected_gates)

        return qnode, cost, max(abs(qml.math.toarray(grads)))
