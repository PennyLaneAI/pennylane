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
# pylint: disable-msg=too-many-arguments
import copy

from pennylane import math
from pennylane.operation import Operation
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

from .approx_time_evolution import ApproxTimeEvolution


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
            obs = [qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.X(1)]
            hamiltonian = qml.Hamiltonian(coeffs, obs)
            frequencies = (2, 4)

            @qml.qnode(dev)
            def circuit(time):
                qml.X(0)
                qml.CommutingEvolution(hamiltonian, time, frequencies)
                return qml.expval(qml.Z(0))

        >>> circuit(1)
        np.float64(0.653...)
    """

    grad_method = None

    def _flatten(self):
        h = self.hyperparameters["hamiltonian"]
        data = (self.data[0], h)
        return data, (self.hyperparameters["frequencies"], self.hyperparameters["shifts"])

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, metadata) -> "CommutingEvolution":
        return cls(data[1], data[0], frequencies=metadata[0], shifts=metadata[1])

    def __init__(self, hamiltonian, time, frequencies=None, shifts=None, id=None):
        # pylint: disable=import-outside-toplevel,too-many-positional-arguments
        from pennylane.gradients.general_shift_rules import generate_shift_rule

        if getattr(hamiltonian, "pauli_rep", None) is None:
            raise TypeError(
                f"hamiltonian must be a linear combination of pauli words. Got {hamiltonian}"
            )

        trainable_hamiltonian = any(math.requires_grad(d) for d in hamiltonian.data)
        if frequencies is not None and not trainable_hamiltonian:
            c, s = generate_shift_rule(frequencies, shifts).T
            recipe = math.stack([c, math.ones_like(c), s]).T
            self.grad_recipe = (recipe,) + (None,) * len(hamiltonian.data)
            self.grad_method = "A"

        self._hyperparameters = {
            "hamiltonian": hamiltonian,
            "frequencies": frequencies,
            "shifts": shifts,
        }

        super().__init__(time, *hamiltonian.parameters, wires=hamiltonian.wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["hamiltonian"] = new_op._hyperparameters["hamiltonian"].map_wires(
            wire_map
        )
        return new_op

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["hamiltonian"])
        context.append(self)
        return self

    @staticmethod
    def compute_decomposition(
        time, *_, wires, hamiltonian, **__
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            *time_and_coeffs (list[tensor_like or float]): list of coefficients of the Hamiltonian, prepended by the time
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
        return [ApproxTimeEvolution(hamiltonian, time, 1)]

    def adjoint(self):
        hamiltonian = self.hyperparameters["hamiltonian"]
        time = self.parameters[0]
        frequencies = self.hyperparameters["frequencies"]
        shifts = self.hyperparameters["shifts"]

        return CommutingEvolution(hamiltonian, -time, frequencies, shifts)
