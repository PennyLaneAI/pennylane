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

import numpy as np

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires

# pylint: disable=arguments-differ,too-many-arguments,super-init-not-called


class ResourceTrotterProduct(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of a Hamiltonian operator.

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

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        first_order_expansion (list[~pennylane.labs.resource_estimation.ResourceOperator]): A list of operators
            constituting the first order expansion of the Hamiltonian to be approximately exponentiated.
        num_steps (int): number of Trotter steps to perform
        order (int): order of the Suzuki-Trotter approximation, must be 1 or even
        wires (list[int] or optional): The wires on which the operator acts. If provided, these wire
            labels will be used instead of the wires provided by the ResourceOperators in the 
            :code:`first_order_expansion`.

    Resources:
        The resources are defined according to the recursive formula presented above.
        The number of times an operator, :math:`e^{itO_{j}}`, is applied depends on the
        number of Trotter steps (`n`) and the order of the approximation (`m`) and is given by:

        .. math:: C_{O_j} = 2 * n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, because of the symmetric form of the recursive formula, the first and last terms get grouped.
        This reduces the counts for those terms to:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

    .. seealso:: :class:`~.TrotterProduct`

    The resources can be computed as:

    **Example**

    >>> import pennylane.labs.resource_estimation as plre
    >>> num_steps, order = (1, 2)
    >>> first_order_expansion = [plre.ResourceRX(), plre.ResourceRY()] # H = X + Y
    >>> gate_set = {"RX", "RY"}
    >>> res = plre.estimate(plre.ResourceTrotterProduct(first_order_expansion, num_steps, order), gate_set=gate_set)
    >>> print(res)
    --- Resources: ---
     Total qubits: 1
     Total gates : 3
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'RX': 2, 'RY': 1}

    """

    resource_keys = {"first_order_expansion", "num_steps", "order", "num_wires"}

    def __init__(self, first_order_expansion, num_steps, order, wires=None):

        self.dequeue(op_to_remove=first_order_expansion)
        self.queue()

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in first_order_expansion)
        except AttributeError as error:
            raise ValueError(
                "All components of first_order_expansion must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        self.first_order_expansion = cmpr_ops
        self.num_steps = num_steps
        self.order = order

        if wires:  # User defined wires take precedent
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)

        else:  # Otherwise determine the wires from the ops in the first order expansion
            ops_wires = Wires.all_wires(
                [op.wires for op in first_order_expansion if op.wires is not None]
            )
            fewest_unique_wires = max(op.num_wires for op in cmpr_ops)

            if len(ops_wires) < fewest_unique_wires:  # If the expansion didn't provide enough wire
                self.wires = None  # labels we assume they all act on the same set
                self.num_wires = fewest_unique_wires

            else:  # If there are more wire labels, use that as the operator wires
                self.wires = ops_wires
                self.num_wires = len(self.wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * first_order_expansion (list[CompressedResourceOp]): A list of operators,
                  in the compressed representation, constituting the first order expansion of the Hamiltonian to be approximately exponentiated.
                * num_steps (int): number of Trotter steps to perform
                * order (int): order of the Suzuki-Trotter approximation, must be 1 or even
                * num_wires (int): the number of wires on which the operator acts

        """
        return {
            "first_order_expansion": self.first_order_expansion,
            "num_steps": self.num_steps,
            "order": self.order,
            "num_wires": self.num_wires,
        }

    @classmethod
    def resource_rep(
        cls, first_order_expansion, num_steps, order, num_wires
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            first_order_expansion (list[~pennylane.labs.resource_estimation.CompressedResourceOp]): A list of operators,
                in the compressed representation, constituting
                the first order expansion of the Hamiltonian to be approximately exponentiated.
            num_steps (int): number of Trotter steps to perform
            order (int): order of the Suzuki-Trotter approximation, must be 1 or even
            num_wires (int): the number of wires on which the operator acts

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "first_order_expansion": first_order_expansion,
            "num_steps": num_steps,
            "order": order,
            "num_wires": num_wires,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        first_order_expansion,
        num_steps,
        order,
        num_wires,  # pylint: disable=unused-argument
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            first_order_expansion (list[~pennylane.labs.resource_estimation.CompressedResourceOp]): A list of operators,
                in the compressed representation, constituting
                the first order expansion of the Hamiltonian to be approximately exponentiated.
            num_steps (int): number of Trotter steps to perform
            order (int): order of the Suzuki-Trotter approximation, must be 1 or even
            num_wires (int): the number of wires on which the operator acts

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        k = order // 2
        gate_list = []

        if order == 1:
            for op in first_order_expansion:
                gate_list.append(plre.GateCount(op, num_steps))
            return gate_list

        # For first and last fragment
        first_frag = first_order_expansion[0]
        last_frag = first_order_expansion[-1]
        gate_list.append(plre.GateCount(first_frag, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(plre.GateCount(last_frag, num_steps * (5 ** (k - 1))))

        # For rest of the fragments
        for op in first_order_expansion[1:-1]:
            gate_list.append(plre.GateCount(op, 2 * num_steps * (5 ** (k - 1))))

        return gate_list


class ResourceTrotterCDF(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of compressed double factorized Hamiltonian.

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

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a compressed double factorized
            Hamiltonian to be approximately exponentiated
        num_steps (int): number of Trotter steps to perform
        order (int): order of the approximation, must be 1 or even.
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are defined according to the recursive formula presented above.
        The number of times an operator, :math:`e^{itO_{j}}`, is applied depends on the
        number of Trotter steps (`n`) and the order of the approximation (`m`) and is given by:

        .. math:: C_{O_j} = 2 * n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, because of the symmetric form of the recursive formula, the first and last terms get grouped.
        This reduces the counts for those terms to:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}


        The resources for a single step expansion of compressed double factorized Hamiltonian are calculated
        based on `arXiv:2506.15784 <https://arxiv.org/abs/2506.15784>`_.


    .. seealso:: :class:`~.TrotterProduct`

    The resources can be computed as:

    **Example**

    >>> import pennylane.labs.resource_estimation as plre
    >>> num_steps, order = (1, 2)
    >>> compact_ham = plre.CompactHamiltonian.cdf(num_orbitals = 4, num_fragments = 4)
    >>> res = plre.estimate(plre.ResourceTrotterCDF(compact_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total qubits: 8
     Total gates : 2.238E+4
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 8
     Gate breakdown:
      {'T': 2.075E+4, 'S': 504.0, 'Z': 336.0, 'Hadamard': 336.0, 'CNOT': 448.0}
    """

    resource_keys = {"compact_ham", "num_steps", "order"}

    def __init__(self, compact_ham, num_steps, order, wires=None):

        if compact_ham.method_name != "cdf":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceTrotterCDF."
                f"This method works with cdf Hamiltonian, {compact_ham.method_name} provided"
            )
        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham

        self.num_wires = 2 * compact_ham.params["num_orbitals"]
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a compressed double factorized
                  Hamiltonian to be approximately exponentiated
                * num_steps (int): number of Trotter steps to perform
                * order (int): order of the approximation, must be 1 or even.

        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
        }

    @classmethod
    def resource_rep(cls, compact_ham, num_steps, order) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a compressed double factorized
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        num_wires = 2 * compact_ham.params["num_orbitals"]
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a compressed double factorized
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        num_frags = compact_ham.params["num_fragments"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {"cmpr_factors_and_counts": ((plre.ResourceRZ.resource_rep(), 2 * num_orb),)},
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": (
                    (plre.ResourceMultiRZ.resource_rep(num_wires=2), (2 * num_orb - 1) * num_orb),
                )
            },
        )

        basis_rot = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot, 2 * num_frags * num_steps))

            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, (num_frags - 1) * num_steps))
            return gate_list

        # For first and last fragment
        gate_list.append(plre.GateCount(basis_rot, 4 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        # For rest of the fragments
        gate_list.append(
            plre.GateCount(basis_rot, 4 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )
        gate_list.append(
            plre.GateCount(op_twobody, 2 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )

        return gate_list

    @classmethod
    def controlled_resource_decomp(
        cls, compact_ham, num_steps, order, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ):
        """Returns the controlled resource decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a compressed double factorized
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even.
            ctrl_num_ctrl_wires (int): the number of control wires for the controlled operations
            ctrl_num_ctrl_values (int): the number of control values for the controlled operations

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        Resources:
            The original resources are controlled only on the Z rotation gates.

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        num_frags = compact_ham.params["num_fragments"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceRZ.resource_rep(),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    ),
                    (2 * num_orb),
                )
            },
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    ),
                    (2 * num_orb - 1) * num_orb,
                )
            },
        )

        basis_rot = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot, 2 * num_frags * num_steps))

            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, (num_frags - 1) * num_steps))
            return gate_list

        # For first and last fragment
        gate_list.append(plre.GateCount(basis_rot, 4 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        # For rest of the fragments
        gate_list.append(
            plre.GateCount(basis_rot, 4 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )
        gate_list.append(
            plre.GateCount(op_twobody, 2 * num_steps * (num_frags - 2) * (5 ** (k - 1)))
        )

        return gate_list


class ResourceTrotterTHC(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of tensor hypercontracted Hamiltonian.

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

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian to be approximately exponentiated
        num_steps (int): number of Trotter steps to perform
        order (int): order of the approximation, must be 1 or even
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are defined according to the recursive formula presented above.
        The number of times an operator, :math:`e^{itO_{j}}`, is applied depends on the
        number of Trotter steps (`n`) and the order of the approximation (`m`) and is given by:

        .. math:: C_{O_j} = 2 * n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, because of the symmetric form of the recursive formula, the first and last terms get grouped.
        This reduces the counts for those terms to:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}


        The resources for a single step expansion of tensor hypercontracted Hamiltonian are calculated
        based on `arXiv:2407.04432 <https://arxiv.org/abs/2407.04432>`_

    .. seealso:: :class:`~.TrotterProduct`

    **Example**

    The resources can be computed as:

    >>> import pennylane.labs.resource_estimation as plre
    >>> num_steps, order = (1, 2)
    >>> compact_ham = plre.CompactHamiltonian.thc(num_orbitals=4, tensor_rank=4)
    >>> res = plre.estimate(plre.ResourceTrotterTHC(compact_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total qubits: 8
     Total gates : 8.520E+3
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 8
     Gate breakdown:
      {'T': 7.888E+3, 'S': 216.0, 'Z': 144.0, 'Hadamard': 144.0, 'CNOT': 128.0}

    """

    resource_keys = {"compact_ham", "num_steps", "order"}

    def __init__(self, compact_ham, num_steps, order, wires=None):

        if compact_ham.method_name != "thc":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceTrotterTHC."
                f"This method works with thc Hamiltonian, {compact_ham.method_name} provided"
            )
        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham

        self.num_wires = compact_ham.params["tensor_rank"] * 2
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                  Hamiltonian to be approximately exponentiated
                * num_steps (int): number of Trotter steps to perform
                * order (int): order of the approximation, must be 1 or even

        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
        }

    @classmethod
    def resource_rep(cls, compact_ham, num_steps, order) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
        }
        num_wires = compact_ham.params["tensor_rank"] * 2
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {"cmpr_factors_and_counts": ((plre.ResourceRZ.resource_rep(), 2 * num_orb),)},
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": (
                    (
                        plre.ResourceMultiRZ.resource_rep(num_wires=2),
                        (2 * tensor_rank - 1) * tensor_rank,
                    ),
                )
            },
        )

        basis_rot_onebody = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})
        basis_rot_twobody = resource_rep(plre.ResourceBasisRotation, {"dim_N": tensor_rank})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps))
            gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps))
            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, num_steps))
            return gate_list

        # For one-body tensor
        gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))

        # For two-body tensor
        gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps * (5 ** (k - 1))))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        return gate_list

    @classmethod
    def controlled_resource_decomp(
        cls, compact_ham, num_steps, order, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ):
        """Returns the controlled resource decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a tensor hypercontracted
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even
            ctrl_num_ctrl_wires (int): the number of control wires for the controlled operations
            ctrl_num_ctrl_values (int): the number of control values for the controlled operations

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        Resources:
            The original resources are controlled only on the Z rotation gates

        """
        k = order // 2
        gate_list = []
        num_orb = compact_ham.params["num_orbitals"]
        tensor_rank = compact_ham.params["tensor_rank"]

        op_onebody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceRZ.resource_rep(),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    ),
                    (2 * num_orb),
                )
            },
        )

        op_twobody = resource_rep(
            plre.ResourceProd,
            {
                "cmpr_factors_and_counts": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    ),
                    (2 * tensor_rank - 1) * tensor_rank,
                )
            },
        )

        basis_rot_onebody = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})
        basis_rot_twobody = resource_rep(plre.ResourceBasisRotation, {"dim_N": tensor_rank})

        if order == 1:
            gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps))
            gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps))
            gate_list.append(plre.GateCount(op_onebody, num_steps))
            gate_list.append(plre.GateCount(op_twobody, num_steps))
            return gate_list

        # For one-body tensor
        gate_list.append(plre.GateCount(basis_rot_onebody, 2 * num_steps * (5 ** (k - 1)) + 2))
        gate_list.append(plre.GateCount(op_onebody, num_steps * (5 ** (k - 1)) + 1))

        # For two-body tensor
        gate_list.append(plre.GateCount(basis_rot_twobody, 2 * num_steps * (5 ** (k - 1))))
        gate_list.append(plre.GateCount(op_twobody, num_steps * (5 ** (k - 1))))

        return gate_list


class ResourceTrotterVibrational(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of vibrational Hamiltonian.

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

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^\text{th}` order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibrational
            Hamiltonian to be approximately exponentiated
        num_steps (int): number of Trotter steps to perform
        order (int): order of the approximation, must be 1 or even
        phase_grad_precision (float): precision for the phase gradient calculation, default value is
            `1e-6`
        coeff_precision (float): precision for the loading of coefficients, default value is
            `1e-3`
        wires (list[int] or optional): the wires on which the operator acts

    Resources:
        The resources are defined according to the recursive formula presented above.
        The number of times an operator, :math:`e^{itO_{j}}`, is applied depends on the
        number of Trotter steps (`n`) and the order of the approximation (`m`) and is given by:

        .. math:: C_{O_j} = 2 * n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, because of the symmetric form of the recursive formula, the first and last terms get grouped.
        This reduces the counts for those terms to:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

        The resources for a single step expansion of vibrational Hamiltonian are calculated based on.
        `arXiv:2504.10602 <https://arxiv.org/pdf/2504.10602>`_

    The resources can be computed as:

    **Example**

    >>> import pennylane.labs.resource_estimation as plre
    >>> compact_ham = plre.CompactHamiltonian.vibrational(num_modes=2, grid_size=4, taylor_degree=2)
    >>> num_steps = 10
    >>> order = 2
    >>> res = plre.estimate(plre.ResourceTrotterVibrational(compact_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total qubits: 83.0
     Total gates : 1.238E+5
     Qubit breakdown:
      clean qubits: 75.0, dirty qubits: 0.0, algorithmic qubits: 8
     Gate breakdown:
      {'Z': 1, 'S': 1, 'T': 749.0, 'X': 1.216E+3, 'Toffoli': 2.248E+4, 'CNOT': 3.520E+4, 'Hadamard': 6.422E+4}
    """

    resource_keys = {"compact_ham", "num_steps", "order", "phase_grad_precision", "coeff_precision"}

    def __init__(
        self,
        compact_ham,
        num_steps,
        order,
        phase_grad_precision=1e-6,
        coeff_precision=1e-3,
        wires=None,
    ):

        if compact_ham.method_name != "vibrational":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceTrotterVibrational."
                f"This method works with vibrational Hamiltonian, {compact_ham.method_name} provided"
            )

        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham
        self.phase_grad_precision = phase_grad_precision
        self.coeff_precision = coeff_precision

        self.num_wires = compact_ham.params["num_modes"] * compact_ham.params["grid_size"]
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibrational
                  Hamiltonian to be approximately exponentiated.
                * num_steps (int): number of Trotter steps to perform
                * order (int): order of the approximation, must be 1 or even
                * phase_grad_precision (float): precision for the phase gradient calculation, default value
                  is `1e-6`
                * coeff_precision (float): precision for the loading of coefficients, default value is
                  `1e-3`

        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
            "phase_grad_precision": self.phase_grad_precision,
            "coeff_precision": self.coeff_precision,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, num_steps, order, phase_grad_precision=1e-6, coeff_precision=1e-3
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibrational
                Hamiltonian to be approximately exponentiated.
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even
            phase_grad_precision (float): precision for the phase gradient calculation, default value is
                `1e-6`
            coeff_precision (float): precision for the loading of coefficients, default value is
                `1e-3`

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
            "phase_grad_precision": phase_grad_precision,
            "coeff_precision": coeff_precision,
        }
        num_wires = compact_ham.params["num_modes"] * compact_ham.params["grid_size"]
        return CompressedResourceOp(cls, num_wires, params)

    @staticmethod
    def _cached_terms(grid_size, taylor_degree, coeff_precision, cached_tree, path, index):
        r"""Recursive function to compute the resources for the trotterization of vibrational Hamiltonian
        while caching the coefficients."""

        cur_path, len_path = tuple(path), len(path)
        coeff_wires = abs(np.floor(np.log2(coeff_precision)))
        gate_cache = []

        x = plre.ResourceX.resource_rep()
        if 1 < len_path <= taylor_degree and cur_path not in cached_tree[len_path]:

            if len(cached_tree[len_path]):
                prev_state = cached_tree[len_path][-1]

                if len_path == 2 and prev_state[0] == prev_state[1]:
                    out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size)
                    gate_cache.append(plre.GateCount(out_square, 1))
                elif len_path == 4 and len(set(prev_state)) == 1:
                    out_square = plre.ResourceOutOfPlaceSquare.resource_rep(
                        register_size=grid_size * 2
                    )
                    gate_cache.append(plre.GateCount(out_square, 1))
                else:
                    multiplier = plre.ResourceOutMultiplier.resource_rep(
                        grid_size, grid_size * (len_path - 1)
                    )
                    gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the Square / Multiplier for current state
            if len_path == 2 and cur_path[-1] == cur_path[-2]:
                out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size)
                gate_cache.append(plre.GateCount(out_square, 1))
            elif len_path == 4 and len(set(cur_path)) == 1:
                out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size * 2)
                gate_cache.append(plre.GateCount(out_square, 1))
            else:
                multiplier = plre.ResourceOutMultiplier.resource_rep(
                    grid_size, grid_size * (len_path - 1)
                )
                gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the coefficient Initializer for current state
            # assuming that half the bits in the coefficient are 1
            gate_cache.append(plre.GateCount(x, coeff_wires / 2))

            # Add the Multiplier for current coefficient
            multiplier = plre.ResourceOutMultiplier.resource_rep(grid_size * len_path, coeff_wires)
            gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the Adder for Resource state
            adder = plre.ResourceSemiAdder.resource_rep(
                max_register_size=2 * max(coeff_wires, 2 * grid_size)
            )
            gate_cache.append(plre.GateCount(adder, 1))

            # Adjoint the Multiplier for current coefficient
            multiplier = plre.ResourceOutMultiplier.resource_rep(grid_size * len_path, coeff_wires)
            gate_cache.append(plre.GateCount(multiplier, 1))

            # Adjoint the coefficient Initializer for current state
            # assuming that half the bits in the coefficient are 1
            gate_cache.append(plre.GateCount(x, coeff_wires / 2))

            cached_tree[len_path].append(cur_path)

        if len_path < taylor_degree and index + 1:
            gate_cache_curr, cached_tree = ResourceTrotterVibrational._cached_terms(
                grid_size, taylor_degree, coeff_precision, cached_tree, path + [index], index
            )  # Depth first search traversal with current element
            gate_cache += gate_cache_curr
            gate_cache_next, cached_tree = ResourceTrotterVibrational._cached_terms(
                grid_size, taylor_degree, coeff_precision, cached_tree, path, index - 1
            )  # Depth first search traversal with next element
            gate_cache += gate_cache_next

        return gate_cache, cached_tree

    @staticmethod
    def _rep_circuit(compact_ham, coeff_precision, num_rep):
        r"""Returns the expansion of the circuit with given number of repetitions."""

        num_modes = compact_ham.params["num_modes"]
        grid_size = compact_ham.params["grid_size"]
        taylor_degree = compact_ham.params["taylor_degree"]

        gate_lst = []
        # Shifted QFT for kinetic part

        t = plre.ResourceT.resource_rep()
        gate_lst.append(plre.GateCount(t, num_rep * (num_modes * np.ceil(np.log2(num_modes) - 1))))

        kinetic_deg = 2
        cached_tree = {index: [] for index in range(1, kinetic_deg + 1)}
        gate_cache, cached_tree = ResourceTrotterVibrational._cached_terms(
            grid_size, kinetic_deg, coeff_precision, cached_tree, path=[], index=num_modes - 1
        )
        gate_lst += gate_cache * num_rep

        cached_tree = {index: [] for index in range(1, taylor_degree + 1)}
        gate_cache, cached_tree = ResourceTrotterVibrational._cached_terms(
            grid_size, taylor_degree, coeff_precision, cached_tree, path=[], index=num_modes - 1
        )
        gate_lst += gate_cache * num_rep

        # Adjoints for the last Squares / Multipliers
        for idx in range(2, taylor_degree):
            last_state = cached_tree[idx][-1]
            if idx == 2 and last_state[-1] == last_state[-2]:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size), num_rep
                    )
                )
            elif idx == 4 and len(set(last_state)) == 1:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size * 2),
                        num_rep,
                    )
                )
            else:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutMultiplier.resource_rep(grid_size, grid_size * (idx - 1)),
                        num_rep,
                    )
                )

        # Shifted QFT Adjoint
        gate_lst.append(plre.GateCount(t, num_rep * (num_modes * np.ceil(np.log2(num_modes) - 1))))

        return gate_lst

    @classmethod
    def resource_decomp(
        cls, compact_ham, num_steps, order, phase_grad_precision, coeff_precision, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibrational
                Hamiltonian to be approximately exponentiated.
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even
            phase_grad_precision (float): precision for the phase gradient calculation, default value is
                `1e-6`
            coeff_precision (float): precision for the loading of coefficients, default value is
                `1e-3`

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        k = order // 2
        gate_list = []
        num_modes = compact_ham.params["num_modes"]
        grid_size = compact_ham.params["grid_size"]
        taylor_degree = compact_ham.params["taylor_degree"]

        phase_grad_wires = abs(np.floor(np.log2(phase_grad_precision)))
        coeff_wires = abs(np.floor(np.log2(coeff_precision)))

        x = plre.ResourceX.resource_rep()

        phase_grad = plre.ResourcePhaseGradient.resource_rep(phase_grad_wires)

        # Allocate the phase gradient registers
        gate_list.append(AllocWires(phase_grad_wires * (taylor_degree - 1)))
        # Resource Registers
        gate_list.append(GateCount(phase_grad, taylor_degree - 1))

        # Allocate auxiliary registers for the coefficients
        gate_list.append(AllocWires(4 * grid_size + 2 * coeff_wires))

        # Basis state prep per mode, implemented only for the first step
        gate_list.append(plre.GateCount(x, num_modes * grid_size))

        if order == 1:
            gate_list += ResourceTrotterVibrational._rep_circuit(
                compact_ham, coeff_precision, num_steps
            )
        else:
            gate_list += ResourceTrotterVibrational._rep_circuit(
                compact_ham, coeff_precision, 2 * num_steps * (5 ** (k - 1))
            )

        # Adjoint of Basis state prep, implemented only for the last step
        gate_list.append(plre.GateCount(x, num_modes * grid_size))

        # Free auxiliary registers for the coefficients
        gate_list.append(FreeWires(4 * grid_size + 2 * coeff_wires))

        # Deallocate the phase gradient registers
        gate_list.append(FreeWires(phase_grad_wires * (taylor_degree - 1)))

        return gate_list


class ResourceTrotterVibronic(ResourceOperator):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of real-space vibronic Hamiltonian.

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

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real-space vibronic
            Hamiltonian to be approximately exponentiated
        num_steps (int): number of Trotter steps to perform
        order (int): order of the approximation (must be 1 or even)
        phase_grad_precision (float): precision for the phase gradient calculation, default value is
            `1e-6`
        coeff_precision (float): precision for the loading of coefficients, default value is
            `1e-3`
        wires (list[int] or optional): the wires on which the operator acts.

    Resources:
        The resources are defined according to the recursive formula presented above.
        The number of times an operator, :math:`e^{itO_{j}}`, is applied depends on the
        number of Trotter steps (`n`) and the order of the approximation (`m`) and is given by:

        .. math:: C_{O_j} = 2 * n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, because of the symmetric form of the recursive formula, the first and last terms get grouped.
        This reduces the counts for those terms to:

        .. math::

            \begin{align}
                C_{O_{0}} &= n \cdot 5^{\frac{m}{2} - 1} + 1,  \\
                C_{O_{N}} &= n \cdot 5^{\frac{m}{2} - 1}.
            \end{align}

        The resources for a single step expansion of real-space vibronic Hamiltonian are calculated
        based on `arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`_.


    The resources can be computed as:

    **Example**

    >>> import pennylane.labs.resource_estimation as plre
    >>> compact_ham = plre.CompactHamiltonian.vibronic(num_modes=2, num_states=4, grid_size=4, taylor_degree=2)
    >>> num_steps = 10
    >>> order = 2
    >>> res = plre.estimate(plre.ResourceTrotterVibronic(compact_ham, num_steps, order))
    >>> print(res)
    --- Resources: ---
     Total qubits: 85.0
     Total gates : 1.332E+5
     Qubit breakdown:
      clean qubits: 75.0, dirty qubits: 0.0, algorithmic qubits: 10
     Gate breakdown:
      {'Z': 1, 'S': 1, 'T': 749.0, 'X': 1.456E+3, 'Hadamard': 6.638E+4, 'Toffoli': 2.320E+4, 'CNOT': 4.144E+4}
    """

    resource_keys = {"compact_ham", "num_steps", "order", "phase_grad_precision", "coeff_precision"}

    def __init__(
        self,
        compact_ham,
        num_steps,
        order,
        phase_grad_precision=1e-6,
        coeff_precision=1e-3,
        wires=None,
    ):

        if compact_ham.method_name != "vibronic":
            raise TypeError(
                f"Unsupported Hamiltonian representation for ResourceTrotterVibronic."
                f"This method works with vibronic Hamiltonian, {compact_ham.method_name} provided"
            )

        self.num_steps = num_steps
        self.order = order
        self.compact_ham = compact_ham
        self.phase_grad_precision = phase_grad_precision
        self.coeff_precision = coeff_precision

        self.num_wires = (
            int(np.ceil(np.log2(compact_ham.params["num_states"])))
            + compact_ham.params["num_modes"] * compact_ham.params["grid_size"]
        )
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real-space vibronic
                  Hamiltonian to be approximately exponentiated
                * num_steps (int): number of Trotter steps to perform
                * order (int): order of the approximation, must be 1 or even
                * phase_grad_precision (float): precision for the phase gradient calculation, default value is
                  `1e-6`
                * coeff_precision (float): precision for the loading of coefficients, default value is
                  `1e-3`

        """
        return {
            "compact_ham": self.compact_ham,
            "num_steps": self.num_steps,
            "order": self.order,
            "phase_grad_precision": self.phase_grad_precision,
            "coeff_precision": self.coeff_precision,
        }

    @classmethod
    def resource_rep(
        cls, compact_ham, num_steps, order, phase_grad_precision=1e-6, coeff_precision=1e-3
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibronic
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even
            phase_grad_precision (float): precision for the phase gradient calculation, default value is
                `1e-6`
            coeff_precision (float): precision for the loading of coefficients, default value is
                `1e-3`

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "compact_ham": compact_ham,
            "num_steps": num_steps,
            "order": order,
            "phase_grad_precision": phase_grad_precision,
            "coeff_precision": coeff_precision,
        }
        num_wires = (
            int(np.ceil(np.log2(compact_ham.params["num_states"])))
            + compact_ham.params["num_modes"] * compact_ham.params["grid_size"]
        )
        return CompressedResourceOp(cls, num_wires, params)

    @staticmethod
    def _cached_terms(
        num_states, grid_size, taylor_degree, coeff_precision, cached_tree, path, index
    ):
        r"""Recursive function to compute the resources for the trotterization of vibronic Hamiltonian
        while caching the coefficients."""

        cur_path, len_path = tuple(path), len(path)
        coeff_wires = abs(int(np.floor(np.log2(coeff_precision))))
        gate_cache = []

        if 1 < len_path <= taylor_degree and cur_path not in cached_tree[len_path]:

            if len(cached_tree[len_path]):
                prev_state = cached_tree[len_path][-1]

                if len_path == 2 and prev_state[0] == prev_state[1]:
                    out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size)
                    gate_cache.append(plre.GateCount(out_square, 1))
                elif len_path == 4 and len(set(prev_state)) == 1:
                    out_square = plre.ResourceOutOfPlaceSquare.resource_rep(
                        register_size=grid_size * 2
                    )
                    gate_cache.append(plre.GateCount(out_square, 1))
                else:
                    multiplier = plre.ResourceOutMultiplier.resource_rep(
                        grid_size, grid_size * (len_path - 1)
                    )
                    gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the Square / Multiplier for current state
            if len_path == 2 and cur_path[-1] == cur_path[-2]:
                out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size)
                gate_cache.append(plre.GateCount(out_square, 1))
            elif len_path == 4 and len(set(cur_path)) == 1:
                out_square = plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size * 2)
                gate_cache.append(plre.GateCount(out_square, 1))
            else:
                multiplier = plre.ResourceOutMultiplier.resource_rep(
                    grid_size, grid_size * (len_path - 1)
                )
                gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the coefficient Initializer for current state
            # assuming that half the bits in the coefficient are 1
            coeff_unitaries = (
                resource_rep(
                    plre.ResourceProd,
                    {
                        "cmpr_factors_and_counts": (
                            (plre.ResourceX.resource_rep(), int(coeff_wires / 2)),
                        )
                    },
                ),
            ) * num_states

            select_op = resource_rep(plre.ResourceSelect, {"cmpr_ops": coeff_unitaries})
            gate_cache.append(plre.GateCount(select_op, 1))

            # Add the Multiplier for current coefficient
            multiplier = plre.ResourceOutMultiplier.resource_rep(grid_size * len_path, coeff_wires)
            gate_cache.append(plre.GateCount(multiplier, 1))

            # Add the Adder for Resource state
            adder = plre.ResourceSemiAdder.resource_rep(
                max_register_size=2 * max(coeff_wires, 2 * grid_size)
            )
            gate_cache.append(plre.GateCount(adder, 1))

            # Adjoint the Multiplier for current coefficient
            multiplier = plre.ResourceOutMultiplier.resource_rep(grid_size * len_path, coeff_wires)
            gate_cache.append(plre.GateCount(multiplier, 1))

            # Adjoint the coefficient Initializer for current state
            # assuming that half the bits in the coefficient are 1
            gate_cache.append(plre.GateCount(select_op, 1))

            cached_tree[len_path].append(cur_path)

        if len_path < taylor_degree and index + 1:
            gate_cache_curr, cached_tree = ResourceTrotterVibronic._cached_terms(
                num_states,
                grid_size,
                taylor_degree,
                coeff_precision,
                cached_tree,
                path + [index],
                index,
            )  # DFS with current element
            gate_cache += gate_cache_curr
            gate_cache_next, cached_tree = ResourceTrotterVibronic._cached_terms(
                num_states, grid_size, taylor_degree, coeff_precision, cached_tree, path, index - 1
            )  # DFS with next element
            gate_cache += gate_cache_next

        return gate_cache, cached_tree

    @staticmethod
    def _rep_circuit(compact_ham, coeff_precision, num_rep):
        r"""Returns the expansion of the circuit with given number of repetitions."""

        num_modes = compact_ham.params["num_modes"]
        num_states = compact_ham.params["num_states"]
        grid_size = compact_ham.params["grid_size"]
        taylor_degree = compact_ham.params["taylor_degree"]

        gate_lst = []
        # Shifted QFT for kinetic part
        t = plre.ResourceT.resource_rep()
        gate_lst.append(plre.GateCount(t, num_rep * (num_modes * np.ceil(np.log2(num_modes) - 1))))

        kinetic_deg = 2
        cached_tree = {index: [] for index in range(1, kinetic_deg + 1)}
        gate_cache, cached_tree = ResourceTrotterVibronic._cached_terms(
            num_states,
            grid_size,
            kinetic_deg,
            coeff_precision,
            cached_tree,
            path=[],
            index=num_modes - 1,
        )
        gate_lst += gate_cache * num_rep

        cached_tree = {index: [] for index in range(1, taylor_degree + 1)}
        gate_cache, cached_tree = ResourceTrotterVibronic._cached_terms(
            num_states,
            grid_size,
            taylor_degree,
            coeff_precision,
            cached_tree,
            path=[],
            index=num_modes - 1,
        )
        gate_lst += gate_cache * num_rep

        # Adjoints for the last Squares / Multipliers
        for idx in range(2, taylor_degree):
            last_state = cached_tree[idx][-1]
            if idx == 2 and last_state[-1] == last_state[-2]:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size), num_rep
                    )
                )
            elif idx == 4 and len(set(last_state)) == 1:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutOfPlaceSquare.resource_rep(register_size=grid_size * 2),
                        num_rep,
                    )
                )
            else:
                gate_lst.append(
                    plre.GateCount(
                        plre.ResourceOutMultiplier.resource_rep(grid_size, grid_size * (idx - 1)),
                        num_rep,
                    )
                )

        # Shifted QFT Adjoint
        gate_lst.append(plre.GateCount(t, num_rep * (num_modes * np.ceil(np.log2(num_modes) - 1))))

        return gate_lst

    @classmethod
    def resource_decomp(
        cls, compact_ham, num_steps, order, phase_grad_precision, coeff_precision, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Args:
            compact_ham (~pennylane.labs.resource_estimation.CompactHamiltonian): a real space vibronic
                Hamiltonian to be approximately exponentiated
            num_steps (int): number of Trotter steps to perform
            order (int): order of the approximation, must be 1 or even
            phase_grad_precision (float): precision for the phase gradient calculation, default value is
                `1e-6`
            coeff_precision (float): precision for the loading of coefficients, default value is
                `1e-3`

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        k = order // 2
        gate_list = []
        num_modes = compact_ham.params["num_modes"]
        num_states = compact_ham.params["num_states"]
        grid_size = compact_ham.params["grid_size"]
        taylor_degree = compact_ham.params["taylor_degree"]

        phase_grad_wires = abs(np.floor(np.log2(phase_grad_precision)))
        coeff_wires = abs(np.floor(np.log2(coeff_precision)))

        x = plre.ResourceX.resource_rep()

        phase_grad = plre.ResourcePhaseGradient.resource_rep(phase_grad_wires)

        # Allocate the phase gradient registers
        gate_list.append(AllocWires(phase_grad_wires * (taylor_degree - 1)))
        # Resource Registers
        gate_list.append(GateCount(phase_grad, taylor_degree - 1))

        # Allocate auxiliary registers for the coefficients
        gate_list.append(AllocWires(4 * grid_size + 2 * coeff_wires))

        # Basis state prep per mode, implemented only for the first step
        gate_list.append(plre.GateCount(x, num_modes * grid_size))

        # electronic state
        gate_list.append(
            plre.GateCount(resource_rep(plre.ResourceHadamard), int(np.ceil(np.log2(num_states))))
        )

        if order == 1:
            gate_list += ResourceTrotterVibronic._rep_circuit(
                compact_ham, coeff_precision, num_steps
            )
        else:
            gate_list += ResourceTrotterVibronic._rep_circuit(
                compact_ham, coeff_precision, 2 * num_steps * (5 ** (k - 1))
            )

        # Adjoint for electronic state
        gate_list.append(
            plre.GateCount(resource_rep(plre.ResourceHadamard), int(np.ceil(np.log2(num_states))))
        )

        # Adjoint of Basis state prep, implemented only for the first step
        gate_list.append(plre.GateCount(x, num_modes * grid_size))

        # Free auxiliary registers for the coefficients
        gate_list.append(FreeWires(4 * grid_size + 2 * coeff_wires))

        # Deallocate the phase gradient registers
        gate_list.append(FreeWires(phase_grad_wires * (taylor_degree - 1)))

        return gate_list
