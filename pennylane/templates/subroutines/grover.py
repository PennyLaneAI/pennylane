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
"""
Contains the Grover Operation template.
"""
import numpy as np
from pennylane.operation import AnyWires, Operation
from pennylane.ops import Hadamard, PauliZ, MultiControlledX, GlobalPhase
from pennylane.wires import Wires


class GroverOperator(Operation):
    r"""Performs the Grover Diffusion Operator.

    .. math::

        G = 2 |s \rangle \langle s | - I
        = H^{\bigotimes n} \left( 2 |0\rangle \langle 0| - I \right) H^{\bigotimes n}

    where :math:`n` is the number of wires, and :math:`|s\rangle` is the uniform superposition:

    .. math::

        |s\rangle = H^{\bigotimes n} |0\rangle =  \frac{1}{\sqrt{2^n}} \sum_{i=0}^{2^n-1} | i \rangle.

    For this template, the operator is implemented with a layer of Hadamards, a layer of :math:`X`,
    followed by a multi-controlled :math:`Z` gate, then another layer of :math:`X` and Hadamards.
    This is expressed in a compact form by the circuit below:

    .. figure:: ../../_static/templates/subroutines/grover.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The open circles on the controlled gate indicate control on 0 instead of 1.
    The ``Z`` gates on the last wire result from leveraging the circuit identity :math:`HXH = Z`,
    where the last ``H`` gate converts the multi-controlled :math:`Z` gate into a
    multi-controlled :math:`X` gate.

    Args:
        wires (Union[Wires, Sequence[int], or int]): the wires to apply to
        work_wires (Union[Wires, Sequence[int], or int]): optional auxiliary wires to assist
            in the decomposition of :class:`~.MultiControlledX`.

    **Example**

    The Grover Diffusion Operator amplifies the magnitude of the basis state with
    a negative phase.  For example, if the solution to the search problem is the :math:`|111\rangle`
    state, we require an oracle that flips its phase; this could be implemented using a `CCZ` gate:

    .. code-block:: python

        n_wires = 3
        wires = list(range(n_wires))

        def oracle():
            qml.Hadamard(wires[-1])
            qml.Toffoli(wires=wires)
            qml.Hadamard(wires[-1])

    We can then implement the entire Grover Search Algorithm for ``num_iterations`` iterations by alternating calls to the oracle and the diffusion operator:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=wires)

        @qml.qnode(dev)
        def GroverSearch(num_iterations=1):
            for wire in wires:
                qml.Hadamard(wire)

            for _ in range(num_iterations):
                oracle()
                qml.templates.GroverOperator(wires=wires)
            return qml.probs(wires)

    >>> GroverSearch(num_iterations=1)
    tensor([0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            0.78125], requires_grad=True)
    >>> GroverSearch(num_iterations=2)
    tensor([0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
        0.0078125, 0.9453125], requires_grad=True)

    We can see that the marked :math:`|111\rangle` state has the greatest probability amplitude.

    Optimally, the oracle-operator pairing should be repeated :math:`\lceil \frac{\pi}{4}\sqrt{2^{n}} \rceil` times.

    """

    num_wires = AnyWires
    grad_method = None

    def __repr__(self):
        return f"GroverOperator(wires={self.wires.tolist()}, work_wires={self.hyperparameters['work_wires'].tolist()})"

    def _flatten(self):
        hyperparameters = (("work_wires", self.hyperparameters["work_wires"]),)
        return tuple(), (self.wires, hyperparameters)

    def __init__(self, wires=None, work_wires=None, id=None):
        if (not hasattr(wires, "__len__")) or (len(wires) < 2):
            raise ValueError("GroverOperator must have at least two wires provided.")

        self._hyperparameters = {
            "n_wires": len(wires),
            "work_wires": Wires(work_wires) if work_wires is not None else Wires([]),
        }

        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(
        wires, work_wires, **kwargs
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.GroverOperator.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the operator acts on
            work_wires (Any or Iterable[Any]): optional auxiliary wires to assist
                in the decomposition of :class:`~.MultiControlledX`.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        ctrl_values = [0] * (len(wires) - 1)

        op_list = []

        for wire in wires[:-1]:
            op_list.append(Hadamard(wire))

        op_list.append(PauliZ(wires[-1]))
        op_list.append(
            MultiControlledX(
                control_values=ctrl_values,
                wires=wires,
                work_wires=work_wires,
            )
        )

        op_list.append(PauliZ(wires[-1]))

        for wire in wires[:-1]:
            op_list.append(Hadamard(wire))

        op_list.append(GlobalPhase(np.pi, wires))

        return op_list

    @staticmethod
    def compute_matrix(n_wires, work_wires):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the operator as a canonical matrix in the computational basis
        (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`.GroverOperator.matrix` and :func:`qml.matrix() <pennylane.matrix>`

        Args:
            n_wires (int): Number of wires the ``GroverOperator`` acts on
            work_wires (Any or Iterable[Any]): optional auxiliary wires to assist decompositions.
                *Unused argument*.

        Returns:
            tensor_like: matrix representation

        The Grover diffusion operator is :math:`2|+\rangle\langle +| - \mathbb{I}`.
        The first term is an all-ones matrix multiplied with two times the squared
        normalization factor of the all-plus state, i.e. all entries of the first term are
        :math:`2^{1-N}` for :math:`N` wires.
        """
        dim = 2**n_wires
        # Grover diffusion operator. Realize the all-ones entry via broadcasting when subtracting
        # the second term.
        return 2 / dim - np.eye(dim)
