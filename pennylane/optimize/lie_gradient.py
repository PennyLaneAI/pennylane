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
"""Lie gradient optimizers"""
import numpy as np
import pennylane as qml
from pennylane.tape import JacobianTape
from pennylane.transforms import batch_transform


@qml.qfunc_transform
def append_time_evolution(tape, hamiltonian, t):
    r"""Append an approximate time evolution to an existing circuit.
    Trotterize the Hamiltonian and with a single step

    .. math:

        U = \prod_i \exp{-it O_i}

    Then this unitary is appended to the current circuit.

    Args:
        tape: qml.QuantumTape of circuit
        hamiltonian: qml.Hamiltonian object
        t: time evolution size

    Returns:

    """
    for obj in tape.operations:
        qml.apply(obj)
    qml.templates.ApproxTimeEvolution(hamiltonian, t, 1)

    for obj in tape.measurements:
        qml.apply(obj)


@batch_transform
def algebra_commutator(tape, observables, lie_algebra_basis_names, nqubits):
    """Calculate the Lie gradient with the parameter shift rule (see `get_omegas`)"""
    tapes = []
    for obs in observables:
        tapes_plus = [JacobianTape(p + "_p") for p in lie_algebra_basis_names]
        tapes_min = [JacobianTape(p + "_m") for p in lie_algebra_basis_names]

        # loop through all operations on the input tape
        for op in tape.operations:
            for t in tapes_plus + tapes_min:
                with t:
                    qml.apply(op)
        for i, t in enumerate(tapes_plus):
            with t:
                qml.PauliRot(
                    np.pi / 2,
                    lie_algebra_basis_names[i],
                    wires=list(range(nqubits)),
                )
                for o in obs:
                    qml.expval(o)
        for i, t in enumerate(tapes_min):
            with t:
                qml.PauliRot(
                    -np.pi / 2,
                    lie_algebra_basis_names[i],
                    wires=list(range(nqubits)),
                )
                for o in obs:
                    qml.expval(o)
        tapes.append((tapes_plus, tapes_min))
    return tapes, None


class LieGradientOptimizer:
    r"""Exact Lie gradient optimizer"""

    # pylint: disable=too-few-public-methods

    def __init__(self, circuit, stepsize=0.01, **kwargs):
        r"""
        Base class for other gradient-descent-based optimizers.

        A step of the Lie gradient iterates the Lie gradient flow on :math:`\text{SU}(2^N)`.
        The function to be minimized is :math:`f(U) = \text{Tr}(U \rho_0 U^\dag H)` given a
        Hamiltonian :math:`H` and initial state :math:`\rho_0`

        .. math::

            U^{(t+1)} = \exp{\epsilon \text{grad}f(U^{(t)}}) U^{(t)}

        where :math:`\epsilon` is a user-defined hyperparameter corresponding to step size.

        The Lie gradient is given by.

        .. math::

             \text{grad}f(U^{(t)}}) = -[U \rho U^\dag, H]

        Subsequent steps of this optimizer will append a Trotterized version of the exact Lie
        gradient and grow the circuit.

        Args:
            circuit (Any): the user-defined hyperparameter :math:`\eta`
            stepsize (float): the user-defined hyperparameter :math:`\eta`
            **kwargs
        **Examples:**

        Define a Hamiltonian cost function to minimize.
        >>> hamiltonian = qml.Hamiltonian(coeffs=[-1.]*3,
        ...observables=[qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0)@qml.PauliX(1)])
        Create an initial state and return the expectation value of the Hamiltonian.
        >>> @qml.qnode(qml.device("default.qubit", wires=2))
        ... def quant_fun():
        ...     qml.RX(0.1, wires=[0])
        ...     qml.RY(0.5, wires=[1])
        ...     qml.CNOT(wires=[0,1])
        ...     qml.RY(0.6, wires=[0])
        ...     return qml.expval(hamiltonian)

        Instatiate the optimizer with the initial circuit and the cost function. Set the stepsize
        accordingly.
        >>> opt = LieGradientOptimizer(circuit = quant_fun, stepsize=0.1)
        Applying 10 steps gets us close the ground state of E=-2.23
        >>> for step in range(10):
        ...    print(step)
        ...    cost = opt.step_and_cost()
        ...    print(cost)

        """
        if not isinstance(circuit, qml.QNode):
            raise TypeError(
                f"`circuit` must be a `qml.QNode`, " f"received {type(circuit)} "
            )

        self.circuit = circuit
        self.circuit.construct([], {})
        if not isinstance(circuit.func().obs, qml.Hamiltonian):
            raise TypeError(
                f"`circuit` must return the expectation value of a `qml.Hamiltonian`,"
                f" "
                f"received {type(circuit.func().obs)} "
            )
        self.nqubits = max(circuit.device.wires) + 1

        if self.nqubits > 4:
            print(
                "WARNING: The exact Lie gradient is exponentially expensive in the number of qubits,"
                f"optimizing a {self.nqubits} qubit circuit may be slow."
            )
        restriction = kwargs.get('restriction', None)
        (
            self.lie_algebra_basis_ops,
            self.lie_algebra_basis_names,
        ) = self.get_su_n_operators(restriction)

        self.hamiltonian = circuit.func().obs
        self.coeffs, self.observables = self.hamiltonian.terms
        self.stepsize = stepsize

    def step_and_cost(
        self,
        *args,
        **kwargs,
    ):
        r"""Update the circuit with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            *args:
            **kwargs:

        Returns:

        """
        omegas = self.get_omegas()
        non_zero_lie_algebra_elements = []
        non_zero_omegas = []
        for i, element in enumerate(omegas):
            if not np.isclose(element, 0):
                non_zero_lie_algebra_elements.append(self.lie_algebra_basis_names[i])
                non_zero_omegas.append(-omegas[i])
        lie_gradient = qml.Hamiltonian(
            non_zero_omegas,
            [
                qml.grouping.string_to_pauli_word(ps)
                for ps in non_zero_lie_algebra_elements
            ],
        )

        new_circuit = append_time_evolution(lie_gradient, self.stepsize)(
            self.circuit.func
        )
        self.circuit = qml.QNode(new_circuit, self.circuit.device)
        return self.circuit()

    def step(
        self,
        *args,
        **kwargs,
    ):
        r"""Update the circuit with one step of the optimizer

        Args:
            *args:
            **kwargs:

        Returns:

        """
        self.step_and_cost(
            *args,
            **kwargs,
        )

    def get_su_n_operators(self, restriction):
        r"""Get the 2x2 SU(N) operators. The dimension of the group is N^2-1.

        Returns:
            List of (N^2)x(N^2) numpy complex arrays and corresponding paulis words
        """

        operators = []
        names = []
        # construct the corresponding pennylane observables
        wire_map = dict(zip(range(self.nqubits), range(self.nqubits)))
        if restriction is None:
            for ps in qml.grouping.pauli_group(self.nqubits):
                operators.append(ps)
                names.append(qml.grouping.pauli_word_to_string(ps, wire_map=wire_map))
        else:
            if not isinstance(restriction, qml.Hamiltonian):
                raise TypeError('`restriction` must be a `qml.Hamiltonian`')
            for ps in set(restriction.ops):
                operators.append(ps)
                names.append(qml.grouping.pauli_word_to_string(ps, wire_map=wire_map))
        return operators, names

    def get_omegas(self):
        r"""Measure the coefficients of the Lie gradient with respect to a Pauli word basis

        We want to calculate the components of the Lie gradient with respect to a Pauli word basis
        For a Hamiltonian of the form :math:`H = \sum_i O_i`, this can be achieved by calculating

        .. math:

            \omega_{i,j} = \text{Tr}([\rho, O_i] P_j)

        where :math:`P_j` is a Pauli word in the set of Pauli monomials on :math:`N` qubits.

        Via the parameter shift rule, the commutator can be calculated as

        .. math:

            [\rho, O_i] = \frac{1}{2}(V(\pi/2) \rho V^\dag(\pi/2) - V(-\pi/2) \rho V^\dag(-\pi/2))

        where :math:`V` is the unitary generated by the Pauli word :math:`V(\theta) = \exp{-i\theta P_j }`.

        Returns:
            Array of omegas for each direction in the Lie algebra.

        """

        obs_groupings, _ = qml.grouping.group_observables(self.observables, self.coeffs)
        # get all circuits we need to calculate the coefficients
        circuits = algebra_commutator(
            self.circuit.qtape,
            obs_groupings,
            self.lie_algebra_basis_names,
            self.nqubits,
        )[0]
        # For each observable O_i in the Hamiltonian, we have to calculate all Lie coefficients
        omegas = np.zeros((len(self.coeffs), len(self.lie_algebra_basis_names)))
        idx = 0

        for circuit_plus, circuit_min in circuits:
            out_plus = qml.execute(circuit_plus, self.circuit.device, gradient_fn=None)
            out_min = qml.execute(circuit_min, self.circuit.device, gradient_fn=None)
            # depending on the length of the grouped observable, store the omegas in the array
            omegas[idx : idx + len(out_plus[0]), :] = 0.5 * (
                np.array(out_plus).T - np.array(out_min).T
            )
            idx += len(out_plus[0])
        return np.dot(self.coeffs, omegas)
