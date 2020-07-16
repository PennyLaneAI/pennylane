# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Contains the ``TimeEvolution`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template


@template
def ApproxTimeEvolution(hamiltonian, time, N=None):
    r""" Applies the Trotterized time-evolution operator for an arbitrary Hamiltonian, expressed in terms
    of Pauli gates. The general
    time-evolution operator for a time-independent Hamiltonian is given by:

    .. math:: U \ = \ e^{-i H t}

    for some Hamiltonian of the form:

    .. math:: H \ = \ \displaystyle\sum_{n} c_n H_n

    In general, implementing this unitary with a set of quantum gates is difficult, as the terms :math:`H_n` don't
    necessarily commute with one another. However, we are able to exploit the Trotter-Suzuki decomposition formula:

    .. math:: e^{A \ + \ B} \ = \ \lim_{n \to \infty} \Big[ e^{A/n} e^{B/n} \Big]^n

    to implement an approximation of the time-evolution operator as:

    .. math:: U \ \approx \ \displaystyle\prod_{k \ = \ 1}^{N} \displaystyle\prod_{n} e^{-i c_n H_n t / N}

    with equality occurring as :math:`N \ \rightarrow \ \infty`.

    .. note::
        This template uses the ``PauliRot`` operation in order to implement
        exponentiated terms of the inputted Hamiltonian. This operation only takes
        terms that are explicitly written in terms of products of Pauli matrices (``PauliX``, ``PauliY``, ``PauliZ``, and ``Identity``).
        Thus, each term in the Hamiltonian must be expressed this way upon input, or else an error will be raised.

    Args:
        hamiltonian (pennylane.Hamiltonian): The PennyLane Hamiltonian object representing the Hamiltonian with which the
                                            time-evolution operator is defined. The Hamiltonian must be explicitly written
                                            in terms of products of Pauli gates (X, Y, Z, and I).
        time (int or float): The time to which the time-evolution unitary evolves a qubit register.
        N (int): The number of Trotterization steps used when approximating the time-evolution operator using
                the decomposition formula. The default value is 1.

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import TimeEvolution

            n_wires = 2
            dev = qml.device('default.qubit', wires=n_wires)

            coeffs = [1, 1]
            obs = [qml.PauliX(0), qml.PauliX(1)]
            hamiltonian = qml.Hamiltonian(coeffs, obs)

            @qml.qnode(dev)
            def circuit(time):
                TimeEvolution(hamiltonian, time)
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        >>> circuit(1)
        [-0.41614684 -0.41614684]
    """

    pauli = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}

    if N is None:
        N = 1

    ###############
    # Input checks

    if not isinstance(hamiltonian, qml.vqe.vqe.Hamiltonian):
        raise ValueError(
            "`hamiltonian` must be of type pennylane.Hamiltonian, got {}".format(
                type(hamiltonian).__name__
            )
        )

    if not isinstance(N, (int, qml.variable.Variable)):
        raise ValueError("`N` must be of type int, got {}".format(type(N).__name__))

    if not isinstance(time, (int, float, qml.variable.Variable)):
        raise ValueError("`time` must be of type float, got {}".format(type(time).__name__))

    ###############

    theta = []
    pauli_words = []
    wires = []

    for i, t in enumerate(hamiltonian.ops):

        prod = (-2 * time * hamiltonian.coeffs[i]) / N
        word = ""

        try:
            if isinstance(t.name, str):
                word = pauli[t.name]

            if isinstance(t.name, list):
                for j in t.name:
                    word += pauli[j]

        except KeyError as error:
            raise ValueError(
                "`hamiltonian` must be written in terms of Pauli matrices, got {}".format(error)
            )

        count = 0
        for j in list(word):
            if j == "I":
                count += 1

        if count != len(word):

            theta.append(prod)
            pauli_words.append(word)
            wires.append(t.wires)

    for i in range(0, N):

        for j, t in enumerate(pauli_words):
            qml.PauliRot(theta[j], t, wires=wires[j])
