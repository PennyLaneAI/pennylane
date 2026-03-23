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

from pennylane import wires
import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

class OutOfPlaceIntegerComparator(ResourceOperator):
    r"""Resource class for an out-of-place integer comparator.

    Compares an n-bit quantum register |x> against a classical
    integer L, storing the result x < L (or x >= L)
    in a dedicated output qubit.

    The circuit computes the borrow chain of the subtraction x - L.
    The n - 1 intermediate borrow qubits are kept dirty after the
    forward pass, enabling the inverse to be performed with Clifford gates
    only (0 Toffoli cost).

   Args:
       value (int): The classical integer L to compare against.
       register_size (int): Number of qubits n encoding x.
       geq (bool): If True, compute x >= L instead of
           x < L.  This adds a single X gate on the output qubit
           (0 Toffoli cost).  Default False.
       wires (Sequence[int] | None): The wires the operation acts on.

    Resources:
        The resources are computed based on Figure 6 of Appendix E in
        `Su et al. (2021) <https://arxiv.org/abs/2105.12767>`_. This decomposition
        is useful when extra auxiliary wires are available and an inverse of the operation is required in the same circuit.

"""

resource_keys = {"value", "register_size", "geq"}

def __init__(
    self,
    value: int,
    register_size: int,
    geq: bool = False,
    wires: wires.WiresLike | None = None,
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

    if len(wires) != self.num_wires:
        raise ValueError(
            f"Expected {self.num_wires} wires, but got {len(wires)}"
        )
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
def resource_rep(cls, value, register_size, geq=False) -> CompressedResourceOp:
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
    num_wires = 2 * register_size
    return CompressedResourceOp(cls, num_wires, params)

@classmethod
def resource_decomp(
    cls, value: int, register_size: int, geq: bool = False
    ) -> list[GateCount]:
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

        Returns:
            list[GateCount]: A list of gate counts representing the resources of the operator.
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

    gate_lst.append(qre.Allocate(register_size - 1))

    gate_lst.append(GateCount(resource_rep(qre.TemporaryAND, {"elbow":"left"}), register_size - 1))

    gate_lst.append(GateCount(resource_rep(qre.CNOT), 2*(register_size - 1)))
    if geq:
        gate_lst.append(GateCount(resource_rep(qre.X), 1))

    return gate_lst

@classmethod
def adjoint_resource_decomp(
    cls, value: int, register_size: int, geq: bool = False
    ) -> list[GateCount]:
    r"""Returns a list representing the resources of the adjoint of the operator. Each object in the list represents a gate and the
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

        Returns:
            list[GateCount]: A list of gate counts representing the resources of the adjoint of the operator.
    """

    gate_lst = []
    if value == 0:
        if not geq:
            gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

    if value > 2 ** (register_size) - 1:
        if geq:
            gate_lst.append(GateCount(resource_rep(qre.X), 1))
            return gate_lst

    gate_lst.append(GateCount(resource_rep(qre.TemporaryAND, {"elbow":"right"}), register_size - 1))
    gate_lst.append(GateCount(resource_rep(qre.CNOT), 2*(register_size - 1)))
    if geq:
        gate_lst.append(GateCount(resource_rep(qre.X), 1))

    gate_lst.append(qre.Deallocate(register_size - 1))

    return gate_lst
