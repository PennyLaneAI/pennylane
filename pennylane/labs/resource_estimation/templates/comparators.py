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
r"""Resource operators for PennyLane subroutine templates."""

from pennylane.labs import resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ,protected-access,too-many-arguments,unused-argument,super-init-not-called


class ResourceSingleQubitCompare(ResourceOperator):
    r"""Resource class for comparing two qubits.

    This operation provides the cost for implementing a comparison between two qubits.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_. Specifically,
        the resources are given as :math:`1` TempAND gate, :math:`4` CNOT gates,
        and :math:`3` X gates.

    **Example**

    The resources for this operation are computed using:

    >>> single_qubit_compare = plre.ResourceSingleQubitCompare()
    >>> print(plre.estimate_resources(single_qubit_compare))
    --- Resources: ---
     Total qubits: 4
     Total gates : 8
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Toffoli': 1, 'CNOT': 4, 'X': 3}
    """

    def __init__(self, wires=None):
        self.num_wires = 4
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: An empty dictionary
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_. Specifically,
            the resources are given as :math:`1` TempAND gate, :math:`4` CNOT gates,
            and :math:`3` X gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(plre.ResourceTempAND), 1))
        gate_lst.append(GateCount(resource_rep(plre.ResourceCNOT), 4))
        gate_lst.append(GateCount(resource_rep(plre.ResourceX), 3))

        return gate_lst


class ResourceTwoQubitCompare(ResourceOperator):
    r"""Resource class for comparing two qubits.

    This operation provides the cost for implementing a comparison between two qubit registers.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_. Specifically,
        the resources are given as :math:`1` ancilla, :math:`2` controlled SWAP gates,
        and :math:`3` CNOT gates.

    **Example**

    The resources for this operation are computed using:

    >>> two_qubit_compare = plre.ResourceTwoQubitCompare()
    >>> print(plre.estimate_resources(two_qubit_compare))
    --- Resources: ---
     Total qubits: 5
     Total gates : 9
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Toffoli': 2, 'CNOT': 7}
    """

    def __init__(self, wires=None):
        self.num_wires = 4
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: An empty dictionary
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_resource_decomp(cls, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_. Specifically,
            the resources are given as :math:`1` ancilla, :math:`2` controlled SWAP gates,
            and :math:`3` CNOT gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_list = []

        gate_list.append(AllocWires(1))
        gate_list.append(GateCount(resource_rep(plre.ResourceCSWAP), 2))
        gate_list.append(GateCount(resource_rep(plre.ResourceCNOT), 3))
        gate_list.append(FreeWires(1))

        return gate_list


class ResourceIntegerComparator(ResourceOperator):
    r"""Resource class for comparing a state to a positive integer.

    This operation provides the cost for comparing basis state, `x`, and a fixed positive integer, val.
    It flips a target qubit if :math:`x \geq val` or :math:`x < val`, depending on the parameter geq.

    Args:
        val (int): the integer to be compared against
        register_size (int): size of the register for basis state
        geq (bool): If set to ``True``, the comparison made will be :math:`x \geq val`. If
            ``False``, the comparison made will be :math:`x < val`.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:

        This decomposition uses the minimum number of ``MultiControlledX`` gates.
        The given integer is first converted into its binary representation, and the decomposition proceeds by
        iteratively examining significant prefixes of this binary representation.
        For example, when geq is `False`, val is 22 (Binary 010110), and num_wires is 6:
            Initial Prefix: For all 6-bit number where the first two control qubits are in the 00 state, the
            flipping condition is always `True`, a ``MultiControlledX`` gate can be applied with the first two wires as
            controls and 2 control values.
            Next Prefix: Subsequently, the next significant bit prefix is examined. The target value 22 begins with 0101.
            Therefore, all 6-bit numbers begining with 0100 will satisfy the condition, so a ``MultiControlledX`` gate can
            be applied with the first four wires as controls and 0100 as control values.
            This iterative procedure continues, with MultiControlledX gates being added for each significant bit prefix of
            the target value, until the full conditional operation is realized with the minimum number of multi-controlled operations.

    **Example**

    The resources for this operation are computed using:

    >>> integer_compare = plre.ResourceIntegerComparator(val=4, register_size=6)
    >>> print(plre.estimate_resources(integer_compare))
    --- Resources: ---
     Total qubits: 9
     Total gates : 19
     Qubit breakdown:
      clean qubits: 2, dirty qubits: 0, algorithmic qubits: 7
     Gate breakdown:
      {'X': 8, 'Toffoli': 3, 'Hadamard': 6, 'CNOT': 2}
    """

    def __init__(self, val, register_size, geq=False, wires=None):
        self.val = val
        self.register_size = register_size
        self.geq = geq
        self.num_wires = register_size + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * val (int): the integer to be compared against
                * register_size (int): size of the register for basis state
                * geq (bool): If set to ``True``, the comparison made will be :math:`x \geq val`. If
                ``False``, the comparison made will be :math:`x < val`.
        """
        return {"val": self.val, "register_size": self.register_size, "geq": self.geq}

    @classmethod
    def resource_rep(cls, val, register_size, geq=False):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            val (int): the integer to be compared against
            register_size (int): size of the register for basis state
            geq (bool): If set to ``True``, the comparison made will be :math:`x \geq val`. If
                ``False``, the comparison made will be :math:`x < val`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"val": val, "register_size": register_size, "geq": geq})

    @classmethod
    def default_resource_decomp(cls, val, register_size, geq=False, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            val (int): the integer to be compared against
            register_size (int): size of the register for basis state
            geq (bool): If set to ``True``, the comparison made will be :math:`x \geq val`. If
                ``False``, the comparison made will be :math:`x < val`.

        Resources:
            This decomposition uses the minimum number of ``MultiControlledX`` gates.
            The given integer is first converted into its binary representation, and the decomposition proceeds by
            iteratively examining significant prefixes of this binary representation.
            For example, when geq is `False`, val is 22 (Binary 010110), and num_wires is 6:
                Initial Prefix: For all 6-bit number where the first two control qubits are in the 00 state, the
                flipping condition is always `True`, a ``MultiControlledX`` gate can be applied with the first two wires as
                controls and 2 control values.
                Next Prefix: Subsequently, the next significant bit prefix is examined. The target value 22 begins with 0101.
                Therefore, all 6-bit numbers begining with 0100 will satisfy the condition, so a ``MultiControlledX`` gate can
                be applied with the first four wires as controls and 0100 as control values.
                This iterative procedure continues, with MultiControlledX gates being added for each significant bit prefix of
                the target value, until the full conditional operation is realized with the minimum number of multi-controlled operations.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        if val == 0:
            if geq:
                gate_lst.append(GateCount(resource_rep(plre.ResourceX), 1))
            return gate_lst

        if val > 2 ** (register_size) - 1:
            if not geq:
                gate_lst.append(GateCount(resource_rep(plre.ResourceX), 1))
            return gate_lst

        binary_str = format(val, f"0{register_size}b")
        if geq:

            first_zero = binary_str.find("0")

            if first_zero == -1:
                mcx = resource_rep(
                    plre.ResourceMultiControlledX,
                    {"num_ctrl_wires": register_size, "num_ctrl_values": 0},
                )
                gate_lst.append(GateCount(mcx, 1))
                return gate_lst

            mcx = resource_rep(
                plre.ResourceMultiControlledX,
                {"num_ctrl_wires": first_zero + 1, "num_ctrl_values": 1},
            )
            gate_lst.append(GateCount(mcx, 1))

            while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
                gate_lst.append(
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": first_zero + 1, "num_ctrl_values": 1},
                        ),
                        1,
                    )
                )

            gate_lst.append(
                GateCount(
                    resource_rep(
                        plre.ResourceMultiControlledX,
                        {"num_ctrl_wires": register_size, "num_ctrl_values": 0},
                    ),
                    1,
                )
            )
            return gate_lst

        last_significant = binary_str.rfind("1")

        gate_lst.append(GateCount(resource_rep(plre.ResourceX), 2 * (last_significant + 1)))

        first_significant = binary_str.find("1")
        gate_lst.append(
            GateCount(
                resource_rep(
                    plre.ResourceMultiControlledX,
                    {"num_ctrl_wires": first_significant + 1, "num_ctrl_values": 0},
                ),
                1,
            )
        )

        while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
            gate_lst.append(
                GateCount(
                    resource_rep(
                        plre.ResourceMultiControlledX,
                        {"num_ctrl_wires": first_significant + 1, "num_ctrl_values": 0},
                    ),
                    1,
                )
            )

        return gate_lst


class ResourceRegisterComparator(ResourceOperator):
    r"""Resource class for comparing two quantum registers.

    This operation provides the cost for implementing a comparison between two
    values, a and b encoded in two quantum registers.

    Args:
        a_num_qubits (int): the size of the first register
        b_num_qubits (int): the size of the second register
        geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
            ``False``, the comparison made will be :math:`a < b`.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B of `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_ for registers of same size.
        If the size of registers differ, unary iteration technique from
        `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ is used
        to combine the results from extra qubits.

    **Example**

    The resources for this operation are computed using:

    >>> register_compare = plre.ResourceRegisterComparator(4, 6)
    >>> print(plre.estimate_resources(register_compare))
    --- Resources: ---
     Total qubits: 21
     Total gates : 16
     Qubit breakdown:
      clean qubits: 10, dirty qubits: 0, algorithmic qubits: 11
     Gate breakdown:
      {'X': 12, 'Toffoli': 4}
    """

    def __init__(self, a_num_qubits, b_num_qubits, geq=False, wires=None):
        self.a_num_qubits = a_num_qubits
        self.b_num_qubits = b_num_qubits
        self.geq = geq
        self.num_wires = a_num_qubits + b_num_qubits + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * a_num_qubits (int): the size of the first register
                * b_num_qubits (int): the size of the second register
                * geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a < b`.
        """
        return {
            "a_num_qubits": self.a_num_qubits,
            "b_num_qubits": self.b_num_qubits,
            "geq": self.geq,
        }

    @classmethod
    def resource_rep(cls, a_num_qubits, b_num_qubits, geq=False):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            a_num_qubits (int): the size of the first register
            b_num_qubits (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a < b`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls, {"a_num_qubits": a_num_qubits, "b_num_qubits": b_num_qubits, "geq": geq}
        )

    @classmethod
    def default_resource_decomp(cls, a_num_qubits, b_num_qubits, geq=False, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            a_num_qubits (int): the size of the first register
            b_num_qubits (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
            ``False``, the comparison made will be :math:`a < b`.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_ for registers of same size.
            If the size of registers differ, unary iteration technique from
            `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ is used
            to combine the results from extra qubits.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_list = []
        compare_size = min(a_num_qubits, b_num_qubits)
        gate_list.append(AllocWires(2 * compare_size))

        one_qubit_compare = resource_rep(plre.ResourceSingleQubitCompare)
        two_qubit_compare = resource_rep(plre.ResourceTwoQubitCompare)
        if a_num_qubits == b_num_qubits:

            gate_list.append(GateCount(two_qubit_compare, a_num_qubits - 1))
            gate_list.append(GateCount(one_qubit_compare, 1))

            gate_list.append(
                GateCount(
                    resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": two_qubit_compare}),
                    a_num_qubits - 1,
                )
            )
            gate_list.append(
                GateCount(
                    resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": one_qubit_compare}),
                    1,
                )
            )

            gate_list.append(GateCount(resource_rep(plre.ResourceX), 1))
            gate_list.append(GateCount(resource_rep(plre.ResourceCNOT), 1))
            gate_list.append(FreeWires(a_num_qubits + b_num_qubits))

            return gate_list

        diff = abs(a_num_qubits - b_num_qubits)

        gate_list.append(GateCount(two_qubit_compare, compare_size - 1))
        gate_list.append(GateCount(one_qubit_compare, 1))

        gate_list.append(
            GateCount(
                resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": two_qubit_compare}),
                compare_size - 1,
            )
        )
        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": one_qubit_compare}), 1)
        )
        mcx = resource_rep(
            plre.ResourceMultiControlledX, {"num_ctrl_wires": diff, "num_ctrl_values": diff}
        )
        gate_list.append(GateCount(mcx, 1))
        gate_list.append(GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": mcx}), 1))

        # collecting the results
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceMultiControlledX, {"num_ctrl_wires": 2, "num_ctrl_values": 1}
                )
            )
        )
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceAdjoint,
                    {
                        "base_cmpr_op": resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        )
                    },
                ),
                1,
            )
        )

        if geq:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 1))
        gate_list.append(FreeWires(2 * compare_size))

        return gate_list
