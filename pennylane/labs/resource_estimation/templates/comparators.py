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
from pennylane.labs.resource_estimation.ops.op_math.symbolic import _apply_adj
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ,unused-argument


class ResourceSingleQubitComparator(ResourceOperator):
    r"""Resource class for comparing the values encoded in two input qubits.

    This operation modifies the input qubits. The original values can be restored
    by applying the operation's adjoint.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
        <https://arxiv.org/pdf/1711.10460>`_. Specifically,
        the resources are given as :math:`1` ``TempAND`` gate, :math:`4` ``CNOT`` gates,
        and :math:`3` ``X`` gates.
        The circuit which applies the comparison operation on qubits :math:`(x,y)` is
        defined as:

        .. code-block:: bash

              x: ─╭●───────╭●─╭●──── x
              y: ─├○────╭●─╰X─│───X─ x=y
            |0>: ─╰X─╭●─│─────│───── x<y
            |0>: ────╰X─╰X────╰X──── x>y

    **Example**

    The resources for this operation are computed using:

    >>> single_qubit_compare = plre.ResourceSingleQubitComparator()
    >>> print(plre.estimate(single_qubit_compare))
    --- Resources: ---
     Total qubits: 4
     Total gates : 8
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Toffoli': 1, 'CNOT': 4, 'X': 3}
    """

    num_wires = 4

    def __init__(self, wires=None):
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
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
            <https://arxiv.org/pdf/1711.10460>`_. Specifically,
            the resources are given as :math:`1` ``TempAND`` gate, :math:`4` ``CNOT`` gates,
            and :math:`3` ``X`` gates.

            The circuit which applies the comparison operation on wires :math:`(x,y)` is
            defined as:

        .. code-block:: bash

              x: ─╭●───────╭●─╭●──── x
              y: ─├○────╭●─╰X─│───X─ x=y
            |0>: ─╰X─╭●─│─────│───── x<y
            |0>: ────╰X─╰X────╰X──── x>y

        Returns:
            list[GateCount]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(plre.ResourceTempAND), 1))
        gate_lst.append(GateCount(resource_rep(plre.ResourceCNOT), 4))
        gate_lst.append(GateCount(resource_rep(plre.ResourceX), 3))

        return gate_lst


class ResourceTwoQubitComparator(ResourceOperator):
    r"""Resource class for comparing the integer values encoded in
    two quantum registers of two qubits each.

    This operation modifies the input registers. The original values can be restored
    by applying the operation's adjoint.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
        <https://arxiv.org/pdf/1711.10460>`_. Specifically,
        the resources are given as :math:`2` ``CSWAP`` gates,
        :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
        requires one clean auxiliary qubit.
        The circuit which applies the comparison operation on registers :math:`(x0,x1)`
        and :math:`(y0, y1)` is defined as:

        .. code-block:: bash

             x1 : ─╭X─╭●────╭●───────┤
             y1 : ─╰●─│─────├SWAP────┤
             x0 : ─╭X─├SWAP─│─────╭X─┤
             y0 : ─╰●─│─────╰SWAP─╰●─┤
            |1> : ────╰SWAP──────────┤

    **Example**

    The resources for this operation are computed using:

    >>> two_qubit_compare = plre.ResourceTwoQubitComparator()
    >>> print(plre.estimate(two_qubit_compare))
    --- Resources: ---
     Total qubits: 5
     Total gates : 10
     Qubit breakdown:
      clean qubits: 1, dirty qubits: 0, algorithmic qubits: 4
     Gate breakdown:
      {'Toffoli': 2, 'CNOT': 7, 'X': 1}
    """

    num_wires = 4

    def __init__(self, wires=None):
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
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/pdf/1711.10460>`_. Specifically,
            the resources are given as :math:`2` ``CSWAP`` gates,
            :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
            requires one clean auxiliary qubit.
            The circuit which applies the comparison operation on registers :math:`(x0,x1)`
            and :math:`(y0, y1)` is defined as:

            .. code-block:: bash

                 x1 : ─╭X─╭●────╭●───────┤
                 y1 : ─╰●─│─────├SWAP────┤
                 x0 : ─╭X─├SWAP─│─────╭X─┤
                 y0 : ─╰●─│─────╰SWAP─╰●─┤
                |1> : ────╰SWAP──────────┤


        Returns:
            list[GateCount]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_list = []

        gate_list.append(AllocWires(1))
        gate_list.append(GateCount(resource_rep(plre.ResourceCSWAP), 2))
        gate_list.append(GateCount(resource_rep(plre.ResourceCNOT), 3))
        gate_list.append(GateCount(resource_rep(plre.ResourceX), 1))
        gate_list.append(FreeWires(1))

        return gate_list

    @classmethod
    def tempand_based_decomp(cls, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/pdf/1711.10460>`_. Specifically,
            the resources are given as :math:`2` ``CSWAP`` gates,
            :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
            is modified to use TempAND gates for building blocks of CSWAP gates.

            .. code-block:: bash

                 x1 : ─╭X─╭●────╭●───────┤
                 y1 : ─╰●─│─────├SWAP────┤
                 x0 : ─╭X─├SWAP─│─────╭X─┤
                 y0 : ─╰●─│─────╰SWAP─╰●─┤
                |1> : ────╰SWAP──────────┤
        """
        gate_list = []

        gate_list.append(AllocWires(2))
        gate_list.append(GateCount(resource_rep(plre.ResourceTempAND), 2))
        gate_list.append(GateCount(resource_rep(plre.ResourceCNOT), 8))
        gate_list.append(GateCount(resource_rep(plre.ResourceX), 3))

        return gate_list


class ResourceIntegerComparator(ResourceOperator):
    r"""This operation applies a controlled `X` gate using integer comparison as the condition.

    Given a basis state :math:`\vert n \rangle`, where :math:`n` is a positive
    integer, and a fixed positive integer :math:`L`, a target qubit is flipped if
    :math:`n \geq L`. Alternatively, the flipping condition can be :math:`n \lt L`.

    Args:
        value (int): The value :math:`L` that the state’s decimal representation is compared against.
        register_size (int): size of the register for basis state
        geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
            ``False``, the comparison made will be :math:`n \lt L`.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        This decomposition uses the minimum number of ``MultiControlledX`` gates.
        The given integer is first converted into its binary representation, and compared to the quantum register
        iteratively, starting with the most significant bit, and progressively including more qubits.
        For example, when :code:`geq` is ``False``, :code:`value` is :math:`22` (Binary :math:`010110`), and
        :code:`num_wires` is :math:`6`:

        - Evaluating most significant bit: For all :math:`6`-bit numbers where the first two control qubits
          are in the :math:`00` state, :math:`n \lt 22` condition is always ``True``. A ``MultiControlledX`` gate
          can be applied with these two wires as controls and control values corresponding to :math:`00`.
        - Refining with subsequent bits: Considering the next most significant bit, since the target value
          begins with :math:`0101`. Therefore, all :math:`6`-bit numbers beginning with :math:`0100` will
          satisfy the condition, so a ``MultiControlledX`` gate can
          be applied with the first four wires as controls and control values corresponding to :math:`0100`.
        - This iterative procedure continues, with ``MultiControlledX`` gates being added for each significant bit of
          the target value, until the full conditional operation is realized with the minimum number of multi-controlled operations.

        The circuit which applies the comparison operation for the above example is defined as:

            .. code-block:: bash

                0: ────╭○─╭○─╭○─┤
                1: ────├○─├●─├●─┤
                2: ────│──├○─├○─┤
                3: ────│──├○─├●─┤
                4: ────│──│──├○─┤
                5: ──-─│──│──│──┤
                6: ────╰X─╰X─╰X─┤


    **Example**

    The resources for this operation are computed using:


    >>> integer_compare = plre.ResourceIntegerComparator(value=4, register_size=6)
    >>> print(plre.estimate(integer_compare))
    --- Resources: ---
     Total qubits: 9
     Total gates : 19
     Qubit breakdown:
      clean qubits: 2, dirty qubits: 0, algorithmic qubits: 7
     Gate breakdown:
      {'X': 8, 'Toffoli': 3, 'Hadamard': 6, 'CNOT': 2}
    """

    resource_keys = {"value", "register_size", "geq"}

    def __init__(self, value, register_size, geq=False, wires=None):
        self.value = value
        self.register_size = register_size
        self.geq = geq
        self.num_wires = register_size + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * value (int): The value :math:`L` that the state’s decimal representation is compared
                  against.
                * register_size (int): size of the register for basis state
                * geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
                  ``False``, the comparison made will be :math:`n \lt L`.

        """
        return {"value": self.value, "register_size": self.register_size, "geq": self.geq}

    @classmethod
    def resource_rep(cls, value, register_size, geq=False):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            value (int): The value :math:`L` that the state’s decimal representation is compared against.
                register_size (int): size of the register for basis state
            geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
                ``False``, the comparison made will be :math:`n \lt L`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = register_size + 1
        return CompressedResourceOp(
            cls, num_wires, {"value": value, "register_size": register_size, "geq": geq}
        )

    @classmethod
    def resource_decomp(cls, value, register_size, geq=False, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            value (int): The value :math:`L` that the state’s decimal representation is compared against.
            register_size (int): size of the register for basis state
            geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
                ``False``, the comparison made will be :math:`n \lt L`.

        Resources:
            This decomposition uses the minimum number of ``MultiControlledX`` gates.
            The given integer is first converted into its binary representation, and compared to the quantum register
            iteratively, starting with the most significant bit, and progressively including more qubits.
            For example, when :code:`geq` is ``False``, :code:`value` is :math:`22` (Binary :math:`010110`), and
            :code:`num_wires` is :math:`6`:

            - Evaluating most significant bit: For all :math:`6`-bit number where the first two control qubits
              are in the :math:`00` state, :math:`n \lt 22` condition is always ``True``. A ``MultiControlledX`` gate
              can be applied with these two wires as controls and control values corresponding to :math:`00`.
            - Refining with subsequent bits: Considering the next most significant bit, since the target value
              begins with :math:`0101`. Therefore, all :math:`6`-bit numbers beginning with :math:`0100` will
              satisfy the condition, so a ``MultiControlledX`` gate can
              be applied with the first four wires as controls and control values corresponding to :math:`0100`.
            - This iterative procedure continues, with ``MultiControlledX`` gates being added for each significant bit of
              the target value, until the full conditional operation is realized with the minimum number of multi-controlled operations.

            The circuit which applies the comparison operation for the above example is defined as:

            .. code-block:: bash

                0: ────╭○─╭○─╭○─┤
                1: ────├○─├●─├●─┤
                2: ────│──├○─├○─┤
                3: ────│──├○─├●─┤
                4: ────│──│──├○─┤
                5: ──-─│──│──│──┤
                6: ────╰X─╰X─╰X─┤

        Returns:
            list[GateCount]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        if value == 0:
            if geq:
                gate_lst.append(GateCount(resource_rep(plre.ResourceX), 1))
            return gate_lst

        if value > 2 ** (register_size) - 1:
            if not geq:
                gate_lst.append(GateCount(resource_rep(plre.ResourceX), 1))
            return gate_lst

        binary_str = format(value, f"0{register_size}b")
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
    r"""This operation applies a controlled `X` gate using register comparison as the condition.

    Given the basis states :math:`\vert a \rangle`, and  :math:`\vert b \rangle`,
    where :math:`a` and :math:`b` are positive
    integers, a target qubit is flipped if
    :math:`a \geq b`. Alternatively, the flipping condition can be :math:`a \lt b`.

    Args:
        first_register (int): the size of the first register
        second_register (int): the size of the second register
        geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
            ``False``, the comparison made will be :math:`a \lt b`.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B of `arXiv:1711.10460
        <https://arxiv.org/pdf/1711.10460>`_ for registers of same size.
        If the size of registers differ, the unary iteration technique from
        `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_ is used
        to combine the results from extra qubits.

    **Example**

    The resources for this operation are computed using:

    >>> register_compare = plre.ResourceRegisterComparator(4, 6)
    >>> print(plre.estimate(register_compare))
    --- Resources: ---
     Total qubits: 11
     Total gates : 89
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 11
     Gate breakdown:
      {'Toffoli': 17, 'CNOT': 51, 'X': 18, 'Hadamard': 3}
    """

    resource_keys = {"first_register", "second_register", "geq"}

    def __init__(self, first_register, second_register, geq=False, wires=None):
        self.first_register = first_register
        self.second_register = second_register
        self.geq = geq
        self.num_wires = first_register + second_register + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * first_register (int): the size of the first register
                * second_register (int): the size of the second register
                * geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                  ``False``, the comparison made will be :math:`a \lt b`.

        """
        return {
            "first_register": self.first_register,
            "second_register": self.second_register,
            "geq": self.geq,
        }

    @classmethod
    def resource_rep(cls, first_register, second_register, geq=False):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            first_register (int): the size of the first register
            second_register (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a \lt b`.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = first_register + second_register + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {"first_register": first_register, "second_register": second_register, "geq": geq},
        )

    @classmethod
    def resource_decomp(cls, first_register, second_register, geq=False, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            first_register (int): the size of the first register
            second_register (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a \lt b`.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/pdf/1711.10460>`_ for registers of same size.
            If the size of registers differ, the unary iteration technique from
            `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_ is used
            to combine the results from extra qubits.

        Returns:
            list[GateCount]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_list = []
        compare_register = min(first_register, second_register)

        one_qubit_compare = resource_rep(plre.ResourceSingleQubitComparator)
        two_qubit_compare = plre.ResourceTwoQubitComparator.tempand_based_decomp(**kwargs)

        if first_register == second_register:

            for op in two_qubit_compare:
                gate_list.append(op * (first_register - 1))
            gate_list.append(GateCount(one_qubit_compare, 1))

            for op in two_qubit_compare:
                gate_list.append(_apply_adj(op) * (first_register - 1))

            gate_list.append(
                GateCount(
                    resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": one_qubit_compare}),
                    1,
                )
            )
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 1))
            gate_list.append(GateCount(resource_rep(plre.ResourceCNOT), 1))

            return gate_list

        diff = abs(first_register - second_register)

        for op in two_qubit_compare:
            gate_list.append(op * (compare_register - 1))
        gate_list.append(GateCount(one_qubit_compare, 1))

        for op in two_qubit_compare:
            gate_list.append(_apply_adj(op) * (compare_register - 1))

        gate_list.append(
            GateCount(resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": one_qubit_compare}), 1)
        )
        mcx = resource_rep(
            plre.ResourceMultiControlledX, {"num_ctrl_wires": diff, "num_ctrl_values": diff}
        )
        gate_list.append(GateCount(mcx, 2))

        # collecting the results
        gate_list.append(
            GateCount(
                resource_rep(
                    plre.ResourceMultiControlledX, {"num_ctrl_wires": 2, "num_ctrl_values": 1}
                ),
                2,
            )
        )

        if geq:
            gate_list.append(GateCount(resource_rep(plre.ResourceX), 1))

        return gate_list
