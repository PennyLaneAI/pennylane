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
from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires

# pylint: disable=arguments-differ, too-many-arguments


class ResourceTrotterCDF(ResourceOperator):  # pylint: disable=too-many-ancestors
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
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in a single step expansion of the exponentiation is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

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
    >>> res = plre.estimate_resources(plre.ResourceTrotterCDF(compact_ham, num_steps, order))
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

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
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
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
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
            in the decomposition

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
    def default_controlled_resource_decomp(
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
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range((2 * num_orb - 1) * num_orb)
                )
            },
        )

        basis_rot = resource_rep(plre.ResourceBasisRotation, {"dim_N": num_orb})

        if order == 1:
            print("Basis rot: ", num_frags * num_steps)
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


class ResourceTrotterTHC(ResourceOperator):  # pylint: disable=too-many-ancestors
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
        The resources are defined according to the recursive formula presented above. Specifically, each
        operator in the single step expansion of the exponentiation is called a number of times given by the formula:

        .. math:: C_{O_{j}} = 2n \cdot 5^{\frac{m}{2} - 1}

        Furthermore, the first and last terms of the Hamiltonian appear in pairs due to the symmetric form
        of the recursive formula. Those counts are further simplified by grouping like terms as:

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
    >>> res = plre.estimate_resources(plre.ResourceTrotterTHC(compact_ham, num_steps, order))
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

        if wires is not None:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
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
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(cls, compact_ham, num_steps, order, **kwargs) -> list[GateCount]:
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
            in the decomposition

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
    def default_controlled_resource_decomp(
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
                "cmpr_factors": tuple(
                    resource_rep(
                        plre.ResourceControlled,
                        {
                            "base_cmpr_op": plre.ResourceMultiRZ.resource_rep(num_wires=2),
                            "num_ctrl_wires": ctrl_num_ctrl_wires,
                            "num_ctrl_values": ctrl_num_ctrl_values,
                        },
                    )
                    for i in range((2 * tensor_rank - 1) * tensor_rank)
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
