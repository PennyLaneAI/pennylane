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

import pennylane.estimator as qre
from pennylane.estimator.ops.op_math.symbolic import _apply_adj
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,unused-argument


class SingleQubitComparator(ResourceOperator):
    r"""Resource class for comparing the values encoded in two input qubits.

    This operation modifies the input qubits. The original values can be restored
    by applying the operation's adjoint.

    Args:
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_. Specifically,
        the resources are given as :math:`1` ``TemporaryAND`` gate, :math:`4` ``CNOT`` gates,
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

    >>> import pennylane.estimator as qre
    >>> single_qubit_compare = qre.SingleQubitComparator()
    >>> print(qre.estimate(single_qubit_compare))
    --- Resources: ---
     Total wires: 4
        algorithmic wires: 4
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 8
      'Toffoli': 1,
      'CNOT': 4,
      'X': 3
    """

    num_wires = 4

    def __init__(self, wires: WiresLike = None):
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: An empty dictionary
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 5 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_. Specifically,
            the resources are given as :math:`1` ``TemporaryAND`` gate, :math:`4` ``CNOT`` gates,
            and :math:`3` ``X`` gates.

            The circuit which applies the comparison operation on wires :math:`(x,y)` is
            defined as:

        .. code-block:: bash

              x: ─╭●───────╭●─╭●──── x
              y: ─├○────╭●─╰X─│───X─ x=y
            |0>: ─╰X─╭●─│─────│───── x<y
            |0>: ────╰X─╰X────╰X──── x>y

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        gate_lst.append(GateCount(resource_rep(qre.TemporaryAND), 1))
        gate_lst.append(GateCount(resource_rep(qre.CNOT), 4))
        gate_lst.append(GateCount(resource_rep(qre.X), 3))

        return gate_lst


class TwoQubitComparator(ResourceOperator):
    r"""Resource class for comparing the integer values encoded in
    two quantum registers of two qubits each.

    This operation modifies the input registers. The original values can be restored
    by applying the operation's adjoint.

    Args:
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_. Specifically,
        the resources are given as :math:`2` ``CSWAP`` gates,
        :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
        requires one zeroed auxiliary qubit.
        The circuit which applies the comparison operation on registers :math:`(x_0,x_1)`
        and :math:`(y_0, y_1)` is defined as:

        .. code-block:: bash

             x1 : ─╭X─╭●────╭●───────┤
             y1 : ─╰●─│─────├SWAP────┤
             x0 : ─╭X─├SWAP─│─────╭X─┤
             y0 : ─╰●─│─────╰SWAP─╰●─┤
            |1> : ────╰SWAP──────────┤

        Note that this operation provides an alternate decomposition using ``TemporaryAND``. See
        the ``TemporaryAND_based_decomp`` method for more details.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> two_qubit_compare = qre.TwoQubitComparator()
    >>> print(qre.estimate(two_qubit_compare))
    --- Resources: ---
     Total wires: 5
        algorithmic wires: 4
        allocated wires: 1
             zero state: 1
             any state: 0
     Total gates : 10
      'Toffoli': 2,
      'CNOT': 7,
      'X': 1
    """

    num_wires = 4

    def __init__(self, wires: WiresLike = None):
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: An empty dictionary
        """
        return {}

    @classmethod
    def resource_rep(cls) -> dict:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_. Specifically,
            the resources are given as :math:`2` ``CSWAP`` gates,
            :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
            requires one zeroed auxiliary qubit.
            The circuit which applies the comparison operation on registers :math:`(x0,x1)`
            and :math:`(y0, y1)` is defined as:

            .. code-block:: bash

                 x1 : ─╭X─╭●────╭●───────┤
                 y1 : ─╰●─│─────├SWAP────┤
                 x0 : ─╭X─├SWAP─│─────╭X─┤
                 y0 : ─╰●─│─────╰SWAP─╰●─┤
                |1> : ────╰SWAP──────────┤


        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_list = []

        gate_list.append(Allocate(1))
        gate_list.append(GateCount(resource_rep(qre.CSWAP), 2))
        gate_list.append(GateCount(resource_rep(qre.CNOT), 3))
        gate_list.append(GateCount(resource_rep(qre.X), 1))
        gate_list.append(Deallocate(1))

        return gate_list

    @classmethod
    def TemporaryAND_based_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_. Specifically,
            the resources are given as :math:`2` ``CSWAP`` gates,
            :math:`3` ``CNOT`` gates, and :math:`1` ``X`` gate. This decomposition
            is modified to use TemporaryAND gates for building blocks of CSWAP gates.

            .. code-block:: bash

               x1: ─╭X───────╭●──────────╭●──────────┤
               y1: ─╰●───────│────────╭X─├●────╭X────┤
              |0>: ──────────│────────│──╰──╭●─│─────┤
               x0: ─╭X─╭X────├●────╭X─│─────│──│──╭X─┤
               y0: ─╰●─│─────│─────│──╰●────╰X─╰●─╰●─┤
              |1>: ────╰●──X─╰───X─╰●────────────────┤
        """
        gate_list = []

        gate_list.append(Allocate(2))
        gate_list.append(GateCount(resource_rep(qre.TemporaryAND), 2))
        gate_list.append(GateCount(resource_rep(qre.CNOT), 8))
        gate_list.append(GateCount(resource_rep(qre.X), 3))

        return gate_list


class IntegerComparator(ResourceOperator):
    r"""This operation applies a controlled ``X`` gate using integer comparison as the condition.

    Given a basis state :math:`\vert n \rangle`, where :math:`n` is a positive
    integer, and a fixed positive integer :math:`L`, a target qubit is flipped if
    :math:`n \geq L`. Alternatively, the flipping condition can be :math:`n \lt L`.

    Args:
        value (int): The value :math:`L` that the state’s decimal representation is compared against.
        register_size (int | None): size of the register for basis state
        geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
            ``False``, the comparison made will be :math:`n \lt L`.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        This decomposition uses the minimum number of ``MultiControlledX`` gates required for the given integer value.
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

    >>> import pennylane.estimator as qre
    >>> integer_compare = qre.IntegerComparator(value=4, register_size=6)
    >>> print(qre.estimate(integer_compare))
    --- Resources: ---
     Total wires: 9
        algorithmic wires: 7
        allocated wires: 2
             zero state: 2
             any state: 0
     Total gates : 19
      'Toffoli': 3,
      'CNOT': 2,
      'X': 8,
      'Hadamard': 6
    """

    resource_keys = {"value", "register_size", "geq"}

    def __init__(
        self,
        value: int,
        register_size: int | None = None,
        geq: bool = False,
        wires: WiresLike = None,
    ):
        if register_size is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `register_size` and `wires`.")
            register_size = len(wires) - 1

        self.value = value
        self.register_size = register_size
        self.geq = geq
        self.num_wires = register_size + 1
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
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
    def resource_rep(
        cls, value: int, register_size: int, geq: bool = False
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            value (int): The value :math:`L` that the state’s decimal representation is compared against.
            register_size (int): size of the register for basis state
            geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
                ``False``, the comparison made will be :math:`n \lt L`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = register_size + 1
        return CompressedResourceOp(
            cls, num_wires, {"value": value, "register_size": register_size, "geq": geq}
        )

    @classmethod
    def resource_decomp(cls, value: int, register_size: int, geq: bool = False) -> list[GateCount]:
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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        if value == 0:
            if geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        if value > 2 ** (register_size) - 1:
            if not geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        binary_str = format(value, f"0{register_size}b")
        if geq:
            first_zero = binary_str.find("0")

            if first_zero == -1:
                mcx = resource_rep(
                    qre.MultiControlledX,
                    {"num_ctrl_wires": register_size, "num_zero_ctrl": 0},
                )
                gate_lst.append(GateCount(mcx, 1))
                return gate_lst

            mcx = resource_rep(
                qre.MultiControlledX,
                {"num_ctrl_wires": first_zero + 1, "num_zero_ctrl": 1},
            )
            gate_lst.append(GateCount(mcx, 1))

            while (first_zero := binary_str.find("0", first_zero + 1)) != -1:
                gate_lst.append(
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": first_zero + 1, "num_zero_ctrl": 1},
                        ),
                        1,
                    )
                )

            gate_lst.append(
                GateCount(
                    resource_rep(
                        qre.MultiControlledX,
                        {"num_ctrl_wires": register_size, "num_zero_ctrl": 0},
                    ),
                    1,
                )
            )
            return gate_lst

        last_significant = binary_str.rfind("1")

        gate_lst.append(GateCount(resource_rep(qre.X), 2 * (last_significant + 1)))

        first_significant = binary_str.find("1")
        gate_lst.append(
            GateCount(
                resource_rep(
                    qre.MultiControlledX,
                    {"num_ctrl_wires": first_significant + 1, "num_zero_ctrl": 0},
                ),
                1,
            )
        )

        while (first_significant := binary_str.find("1", first_significant + 1)) != -1:
            gate_lst.append(
                GateCount(
                    resource_rep(
                        qre.MultiControlledX,
                        {"num_ctrl_wires": first_significant + 1, "num_zero_ctrl": 0},
                    ),
                    1,
                )
            )

        return gate_lst


class RegisterComparator(ResourceOperator):
    r"""This operation applies a controlled ``X`` gate using register comparison as the condition.

    Given the basis states :math:`\vert a \rangle`, and  :math:`\vert b \rangle`,
    where :math:`a` and :math:`b` are positive
    integers, a target qubit is flipped if
    :math:`a \geq b`. Alternatively, the flipping condition can be :math:`a \lt b`.

    Args:
        first_register (int): the size of the first register
        second_register (int): the size of the second register
        geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
            ``False``, the comparison made will be :math:`a \lt b`.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are obtained from appendix B of `arXiv:1711.10460
        <https://arxiv.org/abs/1711.10460>`_ for registers of the same size.
        If the sizes of the registers differ, the unary iteration technique from
        `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_ is used
        to combine the results from extra qubits.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> register_compare = qre.RegisterComparator(4, 6)
    >>> print(qre.estimate(register_compare))
    --- Resources: ---
     Total wires: 17
        algorithmic wires: 11
        allocated wires: 6
             zero state: 6
             any state: 0
     Total gates : 131
      'Toffoli': 11,
      'CNOT': 63,
      'X': 36,
      'Hadamard': 21
    """

    resource_keys = {"first_register", "second_register", "geq"}

    def __init__(
        self, first_register: int, second_register: int, geq: bool = False, wires: WiresLike = None
    ):
        self.first_register = first_register
        self.second_register = second_register
        self.geq = geq
        self.num_wires = first_register + second_register + 1
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
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
    def resource_rep(
        cls, first_register: int, second_register: int, geq: bool = False
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            first_register (int): the size of the first register
            second_register (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a \lt b`.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = first_register + second_register + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {"first_register": first_register, "second_register": second_register, "geq": geq},
        )

    @classmethod
    def resource_decomp(
        cls, first_register: int, second_register: int, geq: bool = False
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            first_register (int): the size of the first register
            second_register (int): the size of the second register
            geq (bool): If set to ``True``, the comparison made will be :math:`a \geq b`. If
                ``False``, the comparison made will be :math:`a \lt b`.

        Resources:
            The resources are obtained from appendix B, Figure 3 in `arXiv:1711.10460
            <https://arxiv.org/abs/1711.10460>`_ for registers of the same size.
            If the sizes of the registers differ, the unary iteration technique from
            `arXiv:1805.03662 <https://arxiv.org/abs/1805.03662>`_ is used
            to combine the results from extra qubits.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_list = []
        compare_register = min(first_register, second_register)

        one_qubit_compare = resource_rep(qre.SingleQubitComparator)
        two_qubit_compare = qre.TwoQubitComparator.TemporaryAND_based_decomp()

        if first_register == second_register:
            for op in two_qubit_compare:
                gate_list.append(op * (first_register - 1))
            gate_list.append(GateCount(one_qubit_compare, 1))

            for op in two_qubit_compare:
                gate_list.append(_apply_adj(op) * (first_register - 1))

            gate_list.append(
                GateCount(
                    resource_rep(qre.Adjoint, {"base_cmpr_op": one_qubit_compare}),
                    1,
                )
            )
            gate_list.append(GateCount(resource_rep(qre.X), 1))
            gate_list.append(GateCount(resource_rep(qre.CNOT), 1))

            return gate_list

        diff = abs(first_register - second_register)

        for op in two_qubit_compare:
            gate_list.append(op * (compare_register - 1))
        gate_list.append(GateCount(one_qubit_compare, 1))

        for op in two_qubit_compare:
            gate_list.append(_apply_adj(op) * (compare_register - 1))

        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": one_qubit_compare}), 1)
        )
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": diff, "num_zero_ctrl": diff})
        gate_list.append(GateCount(mcx, 2))

        # collecting the results
        gate_list.append(
            GateCount(
                resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 2, "num_zero_ctrl": 1}),
                2,
            )
        )

        if geq:
            gate_list.append(GateCount(resource_rep(qre.X), 1))

        return gate_list
