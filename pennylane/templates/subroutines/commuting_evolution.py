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
Contains the CommutingEvolution template.
"""
# pylint: disable-msg=too-many-arguments,import-outside-toplevel
import pennylane as qml
from pennylane.operation import Operation, AnyWires


class CommutingEvolution(Operation):
    r"""Applies the time-evolution operator for a Hamiltonian expressed as a linear combination
    of mutually commuting Pauli words.

    A commuting Hamiltonian is of the form

    .. math:: H \ = \ \displaystyle\sum_{j} c_j P_j,

    where :math:`P_j` are mutually commutative Pauli words and :math:`c_j` are real coefficients.
    The time-evolution under a commuting Hamiltonian is given by a unitary of the form

    .. math::

        U(t) \ = \ e^{-i H t} \ = \exp(-i t \displaystyle\sum_j c_j P_j) =
        \displaystyle\prod_j \exp(-i t c_j P_j).

    If the Hamiltonian has a small number of unique eigenvalues, partial derivatives of observable
    expectation values, i.e.

    .. math:: \langle 0 | W(t)^\dagger O W(t) | 0 \rangle,

    where :math:`W(t) = V U(t) Y` for some :math:`V` and :math:`Y`, taken with respect to
    :math:`t` may be efficiently computed through generalized parameter shift rules. When
    initialized, this template will automatically compute the parameter-shift rule if given the
    Hamiltonian's eigenvalue frequencies, i.e., the unique positive differences between
    eigenvalues.

    .. warning::

       This template uses the :class:`~.ApproxTimeEvolution` operation with ``n=1`` in order to
       implement the time evolution, as a single-step Trotterization is exact for a commuting
       Hamiltonian.

       - If the input Hamiltonian contains Pauli words which do not commute, the
         compilation of the time evolution operator to a sequence of gates will
         not equate to the exact propagation under the given Hamiltonian.

       - Furthermore, if the specified frequencies do not correspond to the
         true eigenvalue frequency spectrum of the commuting Hamiltonian,
         computed gradients will be incorrect in general.

    Args:
        hamiltonian (.Hamiltonian): The commuting Hamiltonian defining the time-evolution operator.
           The Hamiltonian must be explicitly written
           in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
           :class:`~.PauliZ`, and :class:`~.Identity`).
        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{- i H t}`.

    Keyword args:
        frequencies (tuple[int or float]): The unique positive differences between eigenvalues in
            the spectrum of the Hamiltonian. If the frequencies are not given, the cost function
            partial derivative will be computed using the standard two-term shift rule applied to
            the constituent Pauli words in the Hamiltonian individually.

        shifts (tuple[int or float]): The parameter shifts to use in obtaining the
            generalized parameter shift rules. If unspecified, equidistant shifts are used.

    .. details::
        :title: Usage Details

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml

            n_wires = 2
            dev = qml.device('default.qubit', wires=n_wires)

            coeffs = [1, -1]
            obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
            hamiltonian = qml.Hamiltonian(coeffs, obs)
            frequencies = (2, 4)

            @qml.qnode(dev)
            def circuit(time):
                qml.PauliX(0)
                qml.CommutingEvolution(hamiltonian, time, frequencies)
                return qml.expval(qml.PauliZ(0))

        >>> circuit(1)
        0.6536436208636115
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, hamiltonian, time, frequencies=None, shifts=None, do_queue=True, id=None):
        # pylint: disable=import-outside-toplevel
        from pennylane.gradients.general_shift_rules import (
            generate_shift_rule,
        )

        if not isinstance(hamiltonian, qml.Hamiltonian):
            type_name = type(hamiltonian).__name__
            raise TypeError(f"hamiltonian must be of type pennylane.Hamiltonian, got {type_name}")

        trainable_hamiltonian = qml.math.requires_grad(hamiltonian.coeffs)
        if frequencies is not None and not trainable_hamiltonian:
            c, s = generate_shift_rule(frequencies, shifts).T
            recipe = qml.math.stack([c, qml.math.ones_like(c), s]).T
            self.grad_recipe = (recipe,) + (None,) * len(hamiltonian.data)
            self.grad_method = "A"

        self._hyperparameters = {
            "hamiltonian": hamiltonian,
            "frequencies": frequencies,
            "shifts": shifts,
        }

        super().__init__(
            time, *hamiltonian.parameters, wires=hamiltonian.wires, do_queue=do_queue, id=id
        )

    @staticmethod
    def compute_decomposition(
        time, *coeffs, wires, hamiltonian, **kwargs
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            time_and_coeffs (list[tensor_like or float]): list of coefficients of the Hamiltonian, prepended by the time
                variable
            wires (Any or Iterable[Any]): wires that the operator acts on
            hamiltonian (.Hamiltonian): The commuting Hamiltonian defining the time-evolution operator.
            frequencies (tuple[int or float]): The unique positive differences between eigenvalues in
                the spectrum of the Hamiltonian.
            shifts (tuple[int or float]): The parameter shifts to use in obtaining the
                generalized parameter shift rules. If unspecified, equidistant shifts are used.

        .. seealso:: :meth:`~.CommutingEvolution.decomposition`.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        # uses standard PauliRot decomposition through ApproxTimeEvolution.
        hamiltonian = qml.Hamiltonian(coeffs, hamiltonian.ops)
        return qml.ApproxTimeEvolution(hamiltonian, time, 1)

    def adjoint(self):

        hamiltonian = qml.Hamiltonian(self.parameters[1:], self.hyperparameters["hamiltonian"].ops)
        time = self.parameters[0]
        frequencies = self.hyperparameters["frequencies"]
        shifts = self.hyperparameters["shifts"]

        return CommutingEvolution(hamiltonian, -time, frequencies, shifts)
