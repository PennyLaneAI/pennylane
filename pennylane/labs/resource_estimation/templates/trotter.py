# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains templates for Suzuki-Trotter approximation based subroutines.
"""
from typing import Dict

import pennylane as qml
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=arguments-differ


class ResourceTrotterProduct(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of a given Hamiltonian.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
    Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
    the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
    symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        fragments (list[~.ResourceOperator]): A list of compressed operations corresponding to
            the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).
        num_steps (int): an integer representing the number of Trotter steps to perform
        order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)

    Resources:
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in the :code:`first_order_expansion` is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

    .. seealso:: :class:`~.TrotterProduct`

    **Example**

    The arguments can be provided directly to the :code:`estimate_resources()` function to extract the cost:

    >>> fragments = [plre.ResourceX(0), plre.ResourceY(0)]
    >>> trotter = plre.ResourceTrotterProduct(
    ...     fragments = fragments,
    ...     num_steps = 5,
    ...     order = 2,
    ... )
    >>> print(plre.estimate_resources(trotter))
    --- Resources: ---
    Total qubits: 1
    Total gates : 11
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
     {'X': 6, 'Y': 5}

    """

    resource_keys = {"num_steps", "order", "cmpr_fragments"}

    def __init__(self, fragments, num_steps, order, wires=None):
        self.queue(fragments)

        self.num_steps = num_steps
        self.order = order
        self.cmpr_fragments = tuple(op.resource_rep_from_op() for op in fragments)

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in fragments if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in fragments))
            else:
                self.wires = Wires.all_wires(ops_wires)
                self.num_wires = len(self.wires)

    def queue(self, remove_fragments=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if remove_fragments:
            for op in remove_fragments:
                context.remove(op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_steps (int): an integer representing the number of Trotter steps to perform
                * order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
                * cmpr_fragments (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).
        """
        return {
            "num_steps": self.num_steps,
            "order": self.order,
            "cmpr_fragments": self.cmpr_fragments,
        }

    @classmethod
    def resource_rep(cls, num_steps, order, cmpr_fragments) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_steps (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            cmpr_fragments (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_steps": num_steps,
            "order": order,
            "cmpr_fragments": cmpr_fragments,
        }
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, num_steps, order, cmpr_fragments, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
        Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
        the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
        symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
        :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

        .. math::

            \begin{align}
                S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
                S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
                &\vdots \\
                S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
            \end{align}

        where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
        :math:`n`-step Suzuki-Trotter approximation is then defined as:

        .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

        For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

        Args:
            n (int): an integer representing the number of Trotter steps to perform
            order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
            first_order_expansion (list[CompressedResourceOp]): A list of compressed operations corresponding to the exponentiated terms of the hamiltonian (:math:`e^{i t O_{j}}`).

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        Resources:
            The resources are defined according to the recurrsive formula presented above. Specifically, each
            operator in the :code:`first_order_expansion` is called a number of times given by the formula:

            .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

            Furthermore, the first and last terms of the hamiltonian appear in pairs due to the symmetric form
            of the recurrsive formula. Those counts are further simplified by grouping like terms as:

            .. math::

                \begin{align}
                    C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                    C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
                \end{align}

        **Example**

        The arguments can be provided directly to the :code:`resources()` function to extract the cost:

        >>> fragments = [plre.ResourceX(0), plre.ResourceY(0)]
        >>> trotter = plre.ResourceTrotterProduct(
        ...     fragments = fragments,
        ...     num_steps = 5,
        ...     order = 2,
        ... )
        >>> print(plre.estimate_resources(trotter))
        --- Resources: ---
        Total qubits: 1
        Total gates : 11
        Qubit breakdown:
         clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
         {'X': 6, 'Y': 5}

        """
        k = order // 2
        gate_list = []

        if order == 1:
            for cp_rep in cmpr_fragments:
                gate_list.append(GateCount(cp_rep, num_steps))
            return gate_list

        cp_rep_first = cmpr_fragments[0]
        cp_rep_last = cmpr_fragments[-1]
        cp_rep_rest = cmpr_fragments[1:-1]

        for cp_rep in cp_rep_rest:
            gate_list.append(GateCount(cp_rep, 2 * num_steps * (5 ** (k - 1))))

        gate_list.append(GateCount(cp_rep_first, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(GateCount(cp_rep_last, num_steps * (5 ** (k - 1))))

        return gate_list
