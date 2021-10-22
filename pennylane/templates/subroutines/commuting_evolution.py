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
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.templates.subroutines.approx_time_evolution import ApproxTimeEvolution
from pennylane.gradients.general_shift_rules import get_shift_rule


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

    If the Hamiltonian has a small number of unique eigenvalues, cost function partial derivatives
    with respect to :math:`t` may be efficiently computed through generalized parameter shift
    rules. When initialized, this template will compute a gradient recipe if given the
    Hamiltonian's eigenvalue frequencies, i.e. the unique positive differences between eigenvalues
    in its spectrum. Currently, generalized parameter shift rules may only be generated for
    Hamiltonian's with equidistant valued spectra.

    .. note::

       This template uses the :class:`~.ApproxTimeEvolution` operation with `n=1` in order to
       implement the time evolution, as a single-step Trotterization is exact for a commuting
       Hamiltonian. If the input Hamiltonian contains Pauli words which do not commute, the
       compilation of the time evolution operator to a sequence of gates will not equate to the
       exact propagation under the given Hamiltonian.

    Args:
        hamiltonian (.Hamiltonian): The commuting Hamiltonian defining the time-evolution operator.
           The Hamiltonian must be explicitly written
           in terms of products of Pauli gates (:class:`~.PauliX`, :class:`~.PauliY`,
           :class:`~.PauliZ`, and :class:`~.Identity`).
        time (int or float): The time of evolution, namely the parameter :math:`t` in :math:`e^{- i H t}`.

    Keyword args:
        frequencies (list): The unique positive differences between eigenvalues in the
            Hamiltonian's spectrum. If frequencies are not given, cost function partial derivative
            will be computed using the standard two-term shift rule applied to the constituent
            Pauli words in the Hamiltonian individually.

    .. UsageDetails::

        The template is used inside a qnode:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import CommutingEvolution

            n_wires = 2
            dev = qml.device('default.qubit', wires=n_wires)

            coeffs = [1, -1]
            obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)])
            hamiltonian = qml.Hamiltonian(coeffs, obs)
            frequencies = [1,2]

            @qml.qnode(dev)
            def circuit(time):
                qml.PauliX(0)
                CommutingEvolution(hamiltonian, time, frequencies)
                return qml.expval(qml.PauliZ(0))

        >>> circuit(1)
        0.6536436208636115
    """

    num_params = 3
    num_wires = AnyWires
    par_domain = "R"
    grad_method = "A"

    def __init__(self, hamiltonian, time, frequencies=None, do_queue=True, id=None):

        if not isinstance(hamiltonian, qml.Hamiltonian):
            raise ValueError(
                "hamiltonian must be of type pennylane.Hamiltonian, got {}".format(
                    type(hamiltonian).__name__
                )
            )

        if frequencies is not None:
            self.grad_recipe = (None, get_shift_rule(frequencies)[0], None)

        super().__init__(
            hamiltonian, time, frequencies, wires=hamiltonian.wires, do_queue=do_queue, id=id
        )

    def expand(self):
        # uses standard PauliRot decomposition through ApproxTimeEvolution.
        hamiltonian = self.parameters[0]
        time = self.parameters[1]

        return ApproxTimeEvolution(hamiltonian, time, 1).expand()

    def adjoint(self):

        hamiltonian = self.parameters[0]
        time = self.parameters[1]
        frequencies = self.parameters[2]

        return CommutingEvolution(hamiltonian, frequencies, -time)
