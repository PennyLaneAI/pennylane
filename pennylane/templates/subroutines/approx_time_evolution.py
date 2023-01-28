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
r"""
Contains the ApproxTimeEvolution template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import PauliRot


class ApproxTimeEvolution(Operation):
    r"""Applies the Trotterized time-evolution operator for an arbitrary Hamiltonian, expressed in terms
    of Pauli gates.

    The general time-evolution operator for a time-independent Hamiltonian is given by

    .. math:: U(t) \ = \ e^{-i H t},

    for some Hamiltonian of the form:

    .. math:: H \ = \ \displaystyle\sum_{j} H_j.

    Implementing this unitary with a set of quantum gates is difficult, as the terms :math:`H_j` don't
    necessarily commute with one another. However, we are able to exploit the Trotter-Suzuki decomposition formula,

    .. math:: e^{A \ + \ B} \ = \ \lim_{n \to \infty} \Big[ e^{A/n} e^{B/n} \Big]^n,

    to implement an approximation of the time-evolution operator as

    .. math:: U \ \approx \ \displaystyle\prod_{k \ = \ 1}^{n} \displaystyle\prod_{j} e^{-i H_j t / n},

    with the approximation becoming better for larger :math:`n`.
    The circuit implementing this unitary is of the form:

    .. figure:: ../../_static/templates/subroutines/approx_time_evolution.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    It is also important to note that
    this decomposition is exact for any value of :math:`n` when each term of the Hamiltonian
    commutes with every other term.

    .. note::

       This template uses the :class:`~.PauliRot` operation in order to implement
       exponentiated terms of the input Hamiltonian. This operation only takes
       terms that are explicitly written in terms of products of Pauli matrices (:class:`~.PauliX`,
       :class:`~.PauliY`, :class:`~.PauliZ`, and :class:`~.Identity`).
       Thus, each term in the Hamiltonian must be expressed this way upon input, or else an error will be raised.

    Args:
        hamiltonian (.Hamiltonian): The Hamiltonian defining the
           time-evolution operator.
           The Hamiltonian must be explicitly written
           in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
           :class:`~.PauliZ`, and :class:`~.Identity`).
        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{- i H t}`.
        n (int): The number of Trotter steps used when approximating the time-evolution operator.

    .. details::
        :title: Usage Details

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import ApproxTimeEvolution

            n_wires = 2
            wires = range(n_wires)

            dev = qml.device('default.qubit', wires=n_wires)

            coeffs = [1, 1]
            obs = [qml.PauliX(0), qml.PauliX(1)]
            hamiltonian = qml.Hamiltonian(coeffs, obs)

            @qml.qnode(dev)
            def circuit(time):
                ApproxTimeEvolution(hamiltonian, time, 1)
                return [qml.expval(qml.PauliZ(wires=i)) for i in wires]

        >>> circuit(1)
        tensor([-0.41614684 -0.41614684], requires_grad=True)
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, hamiltonian, time, n, do_queue=True, id=None):

        if not isinstance(hamiltonian, qml.Hamiltonian):
            raise ValueError(
                f"hamiltonian must be of type pennylane.Hamiltonian, got {type(hamiltonian).__name__}"
            )

        # extract the wires that the op acts on
        wire_list = [term.wires for term in hamiltonian.ops]
        wires = qml.wires.Wires.all_wires(wire_list)

        self._hyperparameters = {"hamiltonian": hamiltonian, "n": n}

        # trainable parameters are passed to the base init method
        super().__init__(*hamiltonian.data, time, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(
        *coeffs_and_time, wires, hamiltonian, n
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.ApproxTimeEvolution.decomposition`.

        Args:
            coeffs_and_time (list[tensor_like or float]): list of coefficients of the Hamiltonian, appended by the time
                variable
            wires (Any or Iterable[Any]): wires that the operator acts on
            hamiltonian (.Hamiltonian): The Hamiltonian defining the
               time-evolution operator. The Hamiltonian must be explicitly written
               in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
               :class:`~.PauliZ`, and :class:`~.Identity`).
            n (int): The number of Trotter steps used when approximating the time-evolution operator.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        pauli = {"Identity": "I", "PauliX": "X", "PauliY": "Y", "PauliZ": "Z"}

        theta = []
        pauli_words = []
        wires = []
        coeffs = coeffs_and_time[:-1]
        time = coeffs_and_time[-1]
        for i, term in enumerate(hamiltonian.ops):

            word = ""

            try:
                if isinstance(term.name, str):
                    word = pauli[term.name]

                if isinstance(term.name, list):
                    word = "".join(pauli[j] for j in term.name)

            except KeyError as error:
                raise ValueError(
                    f"hamiltonian must be written in terms of Pauli matrices, got {error}"
                ) from error

            # skips terms composed solely of identities
            if word.count("I") != len(word):
                theta.append((2 * time * coeffs[i]) / n)
                pauli_words.append(word)
                wires.append(term.wires)

        op_list = []

        for i in range(n):
            for j, term in enumerate(pauli_words):
                op_list.append(PauliRot(theta[j], term, wires=wires[j]))

        return op_list
