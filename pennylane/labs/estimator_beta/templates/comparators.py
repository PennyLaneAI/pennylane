# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for PennyLane comparison templates."""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import WiresLike

# pylint: disable= signature-differs, arguments-differ


class OutOfPlaceIntegerComparator(ResourceOperator):
    r"""Resource class for an out-of-place integer comparator.

    Compares an n-bit quantum register :math:`|x\rangle` against a classical
    integer :math:`L`, storing the result :math:`x < L` (or :math:`x \geq L`)
    in a dedicated output qubit.

    The circuit computes the borrow chain of the subtraction :math:`x - L`.
    The n - 1 intermediate borrow qubits are kept dirty after the
    forward pass, enabling the inverse to be performed with Clifford gates
    only (0 Toffoli cost).

    Args:
        value (int): The classical integer :math:`L` to compare against.
        register_size (int): Number of qubits :math:`n` encoding :math:`x`.
        geq (bool): If True, compute :math:`x \geq L` instead of
            :math:`x < L`. This adds a single ``X`` gate on the output qubit
            (0 Toffoli cost). Default ``False``.
        wires (WiresLike | None): The wires the operation acts on.

    Resources:
        The resources are computed based on Figure 6 of Appendix E in
        `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. This decomposition
        is useful when extra auxiliary wires are available and an inverse of the operation is required in the same circuit.

        The resources are as follows: `register_size - 1` ``TemporaryAND`` gates, `register_size - 1` ``CNOT`` gates, and the number of ``X``
        gates depend on the number of ones in binary representation of the value we are comparing to. There are also `register_size - 1`
        auxiliary qubits that are allocated in the circuit and are deallocated by its adjoint.

    """

    resource_keys = {"value", "register_size", "geq"}

    def __init__(
        self,
        value: int,
        register_size: int | None = None,
        geq: bool = False,
        wires: WiresLike | None = None,
    ):

        if register_size is None:
            if wires is None:
                raise ValueError("Must provide at least one of `register_size` and `wires`.")
            register_size = len(wires) - 1

        self.value = value
        self.register_size = register_size
        self.geq = geq

        # n input qubits + 1 output qubit
        self.num_wires = register_size + 1

        if wires and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(wires)}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * value (int): The value :math:`L` that the state’s decimal representation is compared against.
                * register_size (int): size of the register for basis state
                * geq (bool): If set to ``True``, the comparison made will be :math:`n \geq L`. If
                  ``False``, the comparison made will be :math:`n \lt L`.
        """
        return {
            "value": self.value,
            "register_size": self.register_size,
            "geq": self.geq,
        }

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

        params = {"value": value, "register_size": register_size, "geq": geq}
        num_wires = register_size + 1
        return CompressedResourceOp(cls, num_wires, params)

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
            The resources are computed based on Figure 6 of Appendix E in
            `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. This decomposition
            is useful when extra auxiliary wires are available and an inverse of the operation is required in the same circuit.

            The resources are as follows: `register_size - 1` ``TemporaryAND`` gates, `register_size - 1` ``CNOT`` gates, and the number of ``X``
            gates depend on the number of ones in binary representation of the value we are comparing to. There are also `register_size - 1`
            auxiliary qubits that are allocated in the circuit and are deallocated by its adjoint.


        Returns:
            list[GateCount]: A list of gate counts representing the resources of the operator.
        """
        num_ones = value.bit_count()
        gate_lst = []
        if value == 0:
            if geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        if value > 2 ** (register_size) - 1:
            if not geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        gate_lst.append(qre.Allocate(register_size - 1))

        gate_lst.append(GateCount(resource_rep(qre.TemporaryAND), register_size - 1))

        gate_lst.append(GateCount(resource_rep(qre.CNOT), register_size - 1))
        gate_lst.append(GateCount(resource_rep(qre.X), 4 * num_ones - 1))
        if geq:
            gate_lst.append(GateCount(resource_rep(qre.X), 1))

        return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources of the adjoint of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            target_resource_params (dict): Dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are computed based on Figure 6 of Appendix E in
            `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. This decomposition
            is useful when extra auxiliary wires are available and an inverse of the operation is required in the same circuit.

            The resources are as follows: `register_size - 1` ``TemporaryAND`` gates, `register_size - 1` ``CNOT`` gates, and the number of ``X``
            gates depend on the number of ones in binary representation of the value we are comparing to. There are also `register_size - 1`
            auxiliary qubits that are allocated in the circuit and are deallocated by its adjoint.


        Returns:
            list[GateCount]: A list of gate counts representing the resources of the adjoint of the operator.
        """
        value = target_resource_params["value"]
        register_size = target_resource_params["register_size"]
        geq = target_resource_params["geq"]
        num_ones = value.bit_count()

        gate_lst = []
        if value == 0:
            if not geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        if value > 2 ** (register_size) - 1:
            if geq:
                gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

        gate_lst.append(
            GateCount(
                resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                register_size - 1,
            )
        )
        gate_lst.append(GateCount(resource_rep(qre.CNOT), register_size - 1))
        gate_lst.append(GateCount(resource_rep(qre.X), 4 * num_ones - 1))
        if geq:
            gate_lst.append(GateCount(resource_rep(qre.X), 1))

        gate_lst.append(qre.Deallocate(register_size - 1))

        return gate_lst


class RegisterEquality(ResourceOperator):
    r"""Resource class for testing the equality of two quantum registers.

    Compares two n-bit quantum registers :math:`|i\rangle` and :math:`|j\rangle`,
    storing the result (:math:`i == j`) in a dedicated output qubit.

    The circuit computes the bitwise XOR of the two registers
    using CNOTs, then uses a ``TemporaryAND`` cascade to flag whether
    all XOR results are zero (i.e., the registers are equal).

    Args:
        register_size (int): Number of qubits n in each register.
        wires (WiresLike | None): The wires the operation acts on.

    Resources:
        The circuit computes the bitwise XOR of the two registers using
        CNOTs, then checks whether all results are zero via a Toffoli
        cascade. The circuit is represented as:

            0: ─╭●────────────────┤
            1: ─│──╭●─────────────┤
            2: ─│──│──╭●──────────┤
            3: ─╰X─│──│───X─╭●────┤
            4: ────╰X─│───X─├●────┤
            5: ───────╰X──X─│──╭●─┤
            6: ─────────────╰⊕─├●─┤
            7: ────────────────╰⊕─┤

    """

    resource_keys = {"register_size"}

    def __init__(
        self,
        register_size: int | None = None,
        wires: WiresLike | None = None,
    ):

        if register_size is None:
            if wires is None:
                raise ValueError("Must provide at least one of `register_size` and `wires`.")
            register_size = (len(wires) - 1) // 2

        self.register_size = register_size

        self.num_wires = 2 * self.register_size + 1

        if wires and len(wires) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(wires)}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * register_size (int): size of the registers for basis state

        """
        return {
            "register_size": self.register_size,
        }

    @classmethod
    def resource_rep(cls, register_size: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            register_size (int): Number of qubits n in each register.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """

        num_wires = 2 * register_size + 1
        return CompressedResourceOp(cls, num_wires, {"register_size": register_size})

    @classmethod
    def resource_decomp(cls, register_size: int) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            register_size (int): Number of qubits n in each register.

        Resources:
            The circuit computes the bitwise XOR of the two registers using
            CNOTs, then checks whether all results are zero via a Toffoli
            cascade. The circuit is represented as:

                0: ─╭●────────────────┤
                1: ─│──╭●─────────────┤
                2: ─│──│──╭●──────────┤
                3: ─╰X─│──│───X─╭●────┤
                4: ────╰X─│───X─├●────┤
                5: ───────╰X──X─│──╭●─┤
                6: ─────────────╰⊕─├●─┤
                7: ────────────────╰⊕─┤

        Returns:
            list[GateCount]: A list of gate counts representing the resources of the operator.
        """
        gate_lst = []
        if register_size == 0:
            return gate_lst

        if register_size == 1:
            gate_lst.append(GateCount(resource_rep(qre.X), 1))
            gate_lst.append(GateCount(resource_rep(qre.CNOT), 2))
            return gate_lst

        mcx = resource_rep(
            qre.MultiControlledX, {"num_ctrl_wires": register_size, "num_zero_ctrl": 0}
        )
        gate_lst.append(GateCount(mcx, 1))
        gate_lst.append(GateCount(resource_rep(qre.X), register_size))
        gate_lst.append(GateCount(resource_rep(qre.CNOT), register_size))

        return gate_lst
