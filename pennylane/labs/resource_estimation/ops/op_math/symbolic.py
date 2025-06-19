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
r"""Resource operators for symbolic operations."""
from functools import singledispatch
from typing import Dict, Iterable, List, Tuple, Union

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
)
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments,too-many-positional-arguments


class ResourceAdjoint(ResourceOperator):
    r"""Resource class for the symbolic Adjoint operation.

    A symbolic class used to represent the adjoint of some base operation.

    Args:
        base_op (~.pennylane.labs.resource_estimation.ResourceOperator): The operator that we
            want the adjoint of.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        This symbolic operation represents the adjoint of some base operation. The resources are
        determined as follows. If the base operation implements the
        :code:`.default_adjoint_resource_decomp()` method, then the resources are obtained from
        this.

        Otherwise, the adjoint resources are given as the adjoint of each operation in the
        base operation's resources.

    .. seealso:: :class:`~.ops.op_math.adjoint.AdjointOperation`

    **Example**

    The adjoint operation can be constructed like this:

    >>> qft = plre.ResourceQFT(num_wires=3)
    >>> adj_qft = plre.ResourceAdjoint(qft)

    We can see how the resources differ by choosing a suitable gateset and estimating resources:

    >>> gate_set = {
    ...     "SWAP",
    ...     "Adjoint(SWAP)",
    ...     "Hadamard",
    ...     "Adjoint(Hadamard)",
    ...     "ControlledPhaseShift",
    ...     "Adjoint(ControlledPhaseShift)",
    ... }
    >>>
    >>> print(plre.estimate_resources(qft, gate_set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 7
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Hadamard': 3, 'SWAP': 1, 'ControlledPhaseShift': 3}
    >>>
    >>> print(plre.estimate_resources(adj_qft, gate_set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 7
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Adjoint(ControlledPhaseShift)': 3, 'Adjoint(SWAP)': 1, 'Adjoint(Hadamard)': 3}

    """

    resource_keys = {"base_cmpr_op"}

    def __init__(self, base_op: ResourceOperator, wires=None) -> None:
        self.queue(remove_op=base_op)
        base_cmpr_op = base_op.resource_rep_from_op()

        self.base_op = base_cmpr_op

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None or base_op.wires
            self.num_wires = base_op.num_wires

    def queue(self, remove_op, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.remove(remove_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_cmpr_op (~.pennylane.labs.resource_estimation.ResourceOperator): The operator
                that we want the adjoint of.

        """
        return {"base_cmpr_op": self.base_op}

    @classmethod
    def resource_rep(cls, base_cmpr_op) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_cmpr_op (~.pennylane.labs.resource_estimation.ResourceOperator): The operator
                that we want the adjoint of.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"base_cmpr_op": base_cmpr_op})

    @classmethod
    def default_resource_decomp(cls, base_cmpr_op: CompressedResourceOp, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): A
                compressed resource representation for the operator we want the adjoint of.
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            This symbolic operation represents the adjoint of some base operation. The resources are
            determined as follows. If the base operation implements the
            :code:`.default_adjoint_resource_decomp()` method, then the resources are obtained from
            this.

            Otherwise, the adjoint resources are given as the adjoint of each operation in the
            base operation's resources.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: :class:`~.ops.op_math.adjoint.AdjointOperation`

        **Example**

        The adjoint operation can be constructed like this:

        >>> qft = plre.ResourceQFT(num_wires=3)
        >>> adj_qft = plre.ResourceAdjoint(qft)

        We can see how the resources differ by choosing a suitable gateset and estimating resources:

        >>> gate_set = {
        ...     "SWAP",
        ...     "Adjoint(SWAP)",
        ...     "Hadamard",
        ...     "Adjoint(Hadamard)",
        ...     "ControlledPhaseShift",
        ...     "Adjoint(ControlledPhaseShift)",
        ... }
        >>>
        >>> print(plre.estimate_resources(qft, gate_set))
        --- Resources: ---
        Total qubits: 3
        Total gates : 7
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'Hadamard': 3, 'SWAP': 1, 'ControlledPhaseShift': 3}
        >>>
        >>> print(plre.estimate_resources(adj_qft, gate_set))
        --- Resources: ---
        Total qubits: 3
        Total gates : 7
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'Adjoint(ControlledPhaseShift)': 3, 'Adjoint(SWAP)': 1, 'Adjoint(Hadamard)': 3}

        """
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        try:
            return base_class.adjoint_resource_decomp(**base_params)
        except ResourcesNotDefined:
            gate_lst = []
            decomp = base_class.resource_decomp(**base_params, **kwargs)

            for gate in decomp[::-1]:  # reverse the order
                gate_lst.append(_apply_adj(gate))
            return gate_lst

    @classmethod
    def default_adjoint_resource_decomp(cls, base_cmpr_op: CompressedResourceOp):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): A
                compressed resource representation for the operator we want the adjoint of.

        Resources:
            The adjoint of an adjointed operation is just the original operation. The resources
            are given as one instance of the base operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(base_cmpr_op)]

    @staticmethod
    def tracking_name(base_cmpr_op: CompressedResourceOp) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Adjoint({base_name})"


class ResourceControlled(ResourceOperator):
    r"""Resource class for the symbolic Controlled operation.

    A symbolic class used to represent the application of some base operation controlled on the
    state of some control qubits.

    Args:
        base_op (~.pennylane.labs.resource_estimation.ResourceOperator): The base operator to be
            controlled.
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_ctrl_values (int): the number of control qubits, that are controlled when in the
            :math:`|0\rangle` state

    Resources:
        The resources are determined as follows. If the base operator implements the
        :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
        this.

        Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
        which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
        to an additional cost of two :class:`~.ResourceX` gates per :code:`num_ctrl_values`.
        Secondly, the base operation resources are extracted and we add to the cost the controlled
        variant of each operation in the resources.

    .. seealso:: :class:`~.ops.op_math.controlled.ControlledOp`

    **Example**

    The controlled operation can be constructed like this:

    >>> x = plre.ResourceX()
    >>> cx = plre.ResourceControlled(x, num_ctrl_wires=1, num_ctrl_values=0)
    >>> ccx = plre.ResourceControlled(x, num_ctrl_wires=2, num_ctrl_values=2)

    We can observe the expected gates when we estimate the resources.

    >>> print(plre.estimate_resources(cx))
    --- Resources: ---
    Total qubits: 2
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
    {'CNOT': 1}
    >>>
    >>> print(plre.estimate_resources(ccx))
    --- Resources: ---
    Total qubits: 3
    Total gates : 5
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
    {'X': 4, 'Toffoli': 1}

    """

    resource_keys = {"base_cmpr_op", "num_ctrl_wires", "num_ctrl_values"}

    def __init__(
        self,
        base_op: ResourceOperator,
        num_ctrl_wires: int,
        num_ctrl_values: int,
        wires=None,
    ) -> None:
        self.queue(remove_base_op=base_op)
        base_cmpr_op = base_op.resource_rep_from_op()

        self.base_op = base_cmpr_op
        self.num_ctrl_wires = num_ctrl_wires
        self.num_ctrl_values = num_ctrl_values

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            num_base_wires = base_op.num_wires
            self.num_wires = num_ctrl_wires + num_base_wires

    def queue(self, remove_base_op, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.remove(remove_base_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.pennylane.labs.resource_estimation.ResourceOperator]): the class type of the base operator to be controlled
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * num_ctrl_wires (int): the number of qubits the operation is controlled on
                * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
                * num_work_wires (int): the number of additional qubits that can be used for decomposition
        """

        return {
            "base_cmpr_op": self.base_op,
            "num_ctrl_wires": self.num_ctrl_wires,
            "num_ctrl_values": self.num_ctrl_values,
        }

    @classmethod
    def resource_rep(
        cls,
        base_cmpr_op,
        num_ctrl_wires,
        num_ctrl_values,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.pennylane.labs.resource_estimation.ResourceOperator]): the class type of the base operator to be controlled
            base_params (dict): the resource parameters required to extract the cost of the base operator
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
            },
        )

    @classmethod
    def default_resource_decomp(
        cls, base_cmpr_op, num_ctrl_wires, num_ctrl_values, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): The base
                operator to be controlled.
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the
                :math:`|0\rangle` state

        Resources:
            The resources are determined as follows. If the base operator implements the
            :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
            this.

            Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
            which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
            to an additional cost of two :class:`~.ResourceX` gates per :code:`num_ctrl_values`.
            Secondly, the base operation resources are extracted and we add to the cost the controlled
            variant of each operation in the resources.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: :class:`~.ops.op_math.controlled.ControlledOp`

        **Example**

        The controlled operation can be constructed like this:

        >>> x = plre.ResourceX()
        >>> cx = plre.ResourceControlled(x, num_ctrl_wires=1, num_ctrl_values=0)
        >>> ccx = plre.ResourceControlled(x, num_ctrl_wires=2, num_ctrl_values=2)

        We can observe the expected gates when we estimate the resources.

        >>> print(plre.estimate_resources(cx))
        --- Resources: ---
        Total qubits: 2
        Total gates : 1
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
        Gate breakdown:
        {'CNOT': 1}
        >>>
        >>> print(plre.estimate_resources(ccx))
        --- Resources: ---
        Total qubits: 3
        Total gates : 5
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'X': 4, 'Toffoli': 1}

        """

        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        try:
            return base_class.controlled_resource_decomp(
                ctrl_num_ctrl_wires=num_ctrl_wires,
                ctrl_num_ctrl_values=num_ctrl_values,
                **base_params,
            )
        except re.ResourcesNotDefined:
            pass

        gate_lst = []
        if num_ctrl_values != 0:
            x = resource_rep(re.ResourceX)
            gate_lst.append(GateCount(x, 2 * num_ctrl_values))

        decomp = base_class.resource_decomp(**base_params, **kwargs)
        for action in decomp:
            if isinstance(action, GateCount):
                gate = action.gate
                c_gate = cls.resource_rep(
                    gate,
                    num_ctrl_wires,
                    num_ctrl_values=0,  # we flipped already and added the X gates above
                )
                gate_lst.append(GateCount(c_gate, action.count))

            else:
                gate_lst.append(action)

        return gate_lst

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        base_cmpr_op,
        num_ctrl_wires,
        num_ctrl_values,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            ctrl_num_ctrl_values (int): The subset of those control qubits, which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): The base
                operator to be controlled.
            num_ctrl_wires (int): the number of control qubits of the operation
            num_ctrl_values (int): The subset of control qubits of the operation, that are controlled
                when in the :math:`|0\rangle` state.

        Resources:
            The resources are derived by simply combining the control qubits, control-values and
            work qubits into a single instance of :class:`~.ResourceControlled` gate, controlled
            on the whole set of control-qubits.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [
            GateCount(
                cls.resource_rep(
                    base_cmpr_op,
                    ctrl_num_ctrl_wires + num_ctrl_wires,
                    ctrl_num_ctrl_values + num_ctrl_values,
                )
            ),
        ]

    @staticmethod
    def tracking_name(
        base_cmpr_op: CompressedResourceOp,
        num_ctrl_wires: int,
        num_ctrl_values: int,
    ):
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"C({base_name}, num_ctrl_wires={num_ctrl_wires},num_ctrl_values={num_ctrl_values})"


class ResourcePow(ResourceOperator):
    r"""Resource class for the symbolic Pow operation.

    A symbolic class used to represent some base operation raised to a power.

    Args:
        base_op (~.pennylane.labs.resource_estimation.ResourceOperator): The operator that we
            want to exponentiate.
        z (float): the exponent (default value is 1)
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are determined as follows. If the power :math:`z = 0`, then we have the identitiy
        gate and we have no resources. If the base operation class :code:`base_class` implements the
        :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
        the resources of the operation raised to the power :math:`z` are given by extracting the base
        operation's resources (via :code:`.resources()`) and raising each operation to the same power.

    .. seealso:: :class:`~.ops.op_math.pow.PowOperation`

    **Example**

    The operation raised to a power :math:`z` can be constructed like this:

    >>> z = plre.ResourceZ()
    >>> z_2 = plre.ResourcePow(z, 2)
    >>> z_5 = plre.ResourcePow(z, 5)

    We obtain the expected resources.

    >>> print(plre.estimate_resources(z_2, gate_set={"Identity", "Z"}))
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'Identity': 1}
    >>>
    >>> print(plre.estimate_resources(z_5, gate_set={"Identity", "Z"}))
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'Z': 1}

    """

    resource_keys = {"base_cmpr_op", "z"}

    def __init__(self, base_op: ResourceOperator, z: int, wires=None) -> None:
        self.queue(remove_op=base_op)
        base_cmpr_op = base_op.resource_rep_from_op()

        self.z = z
        self.base_op = base_cmpr_op

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None or base_op.wires
            self.num_wires = base_op.num_wires

    def queue(self, remove_op, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.remove(remove_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.pennylane.labs.resource_estimation.ResourceOperator]): The class type of the base operator to be raised to some power.
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * z (int): the power that the operator is being raised to
        """
        return {
            "base_cmpr_op": self.base_op,
            "z": self.z,
        }

    @classmethod
    def resource_rep(cls, base_cmpr_op, z) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.pennylane.labs.resource_estimation.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            z (int): the power that the operator is being raised to

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"base_cmpr_op": base_cmpr_op, "z": z})

    @classmethod
    def default_resource_decomp(cls, base_cmpr_op, z, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): A
                compressed resource representation for the operator we want to exponentiate.
            z (float): the exponent (default value is 1)

        Resources:
            The resources are determined as follows. If the power :math:`z = 0`, then we have the identitiy
            gate and we have no resources. If the base operation class :code:`base_class` implements the
            :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
            the resources of the operation raised to the power :math:`z` are given by extracting the base
            operation's resources (via :code:`.resources()`) and raising each operation to the same power.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: :class:`~.ops.op_math.pow.PowOperation`

        **Example**

        The operation raised to a power :math:`z` can be constructed like this:

        >>> z = plre.ResourceZ()
        >>> z_2 = plre.ResourcePow(z, 2)
        >>> z_5 = plre.ResourcePow(z, 5)

        We obtain the expected resources.

        >>> print(plre.estimate_resources(z_2, gate_set={"Identity", "Z"}))
        --- Resources: ---
        Total qubits: 1
        Total gates : 1
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'Identity': 1}
        >>>
        >>> print(plre.estimate_resources(z_5, gate_set={"Identity", "Z"}))
        --- Resources: ---
        Total qubits: 1
        Total gates : 1
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'Z': 1}

        """
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)

        if z == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]

        if z == 1:
            return [GateCount(base_cmpr_op)]

        try:
            return base_class.pow_resource_decomp(pow_z=z, **base_params)
        except re.ResourcesNotDefined:
            return [GateCount(base_cmpr_op, z)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, base_cmpr_op, z):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            pow_z (int): the exponent that the pow-operator is being raised to
            base_cmpr_op (~.pennylane.labs.resource_estimation.CompressedResourceOp): A
                compressed resource representation for the operator we want to exponentiate.
            z (float): the exponent that the base operator is being raised to (default value is 1)

        Resources:
            The resources are derived by simply adding together the :math:`z` exponent and the
            :math:`z_{0}` exponent into a single instance of :class:`~.ResourcePow` gate, raising
            the base operator to the power :math:`z + z_{0}`.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(base_cmpr_op, pow_z * z))]

    @staticmethod
    def tracking_name(base_cmpr_op: CompressedResourceOp, z: int) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Pow({base_name}, {z})"


class ResourceProd(ResourceOperator):
    r"""Resource class for the symbolic Prod operation.

    A symbolic class used to represent a product of some base operations.

    Args:
        res_ops (tuple[~.pennylane.labs.resource_estimation.ResourceOperator]): A tuple of
            resource operators or a nested tuple of resource operators and counts.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        This symbolic class represents a product of operations. The resources are defined trivially
        as the counts for each operation in the product.

    .. seealso:: :class:`~.ops.op_math.prod.Prod`

    **Example**

    The product of operations can be constructed from a list of operations or
    a nested tuple where each operator is accompanied with the number of counts.
    Note, each operation in the product must be a valid :class:`~.pennylane.labs.resource_estimation.ResourceOperator`

    We can construct a product operator as follows:

    >>> factors = [plre.ResourceX(), plre.ResourceY(), plre.ResourceZ()]
    >>> prod_xyz = plre.ResourceProd(factors)
    >>>
    >>> print(plre.estimate_resources(prod_xyz))
    --- Resources: ---
    Total qubits: 1
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
     {'X': 1, 'Y': 1, 'Z': 1}

    We can also specify the factors as a tuple with

    >>> factors = [(plre.ResourceX(), 2), (plre.ResourceZ(), 3)]
    >>> prod_x2z3 = plre.ResourceProd(factors)
    >>>
    >>> print(plre.estimate_resources(prod_x2z3))
    --- Resources: ---
    Total qubits: 1
    Total gates : 5
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
     {'X': 2, 'Z': 3}

    """

    resource_keys = {"cmpr_factors_and_counts"}

    def __init__(
        self,
        res_ops: Iterable[Union[ResourceOperator, Tuple[int, ResourceOperator]]],
        wires=None,
    ) -> None:

        ops = []
        counts = []
        for op_or_tup in res_ops:
            op, count = op_or_tup if isinstance(op_or_tup, tuple) else (op_or_tup, 1)

            ops.append(op)
            counts.append(count)

        self.queue(ops)

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        except AttributeError as error:
            raise ValueError(
                "All factors of the Product must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        self.cmpr_factors_and_counts = tuple(zip(cmpr_ops, counts))

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in ops if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in ops))
            else:
                self.wires = Wires.all_wires(ops_wires)
                self.num_wires = len(self.wires)

    def queue(self, ops_to_remove, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        for op in ops_to_remove:
            context.remove(op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * cmpr_factors_and_counts (Tuple[Tuple[~.labs.resource_estimation.CompressedResourceOp, int]]):
            A sequence of tuples containing the operations, in the compressed representation, and
            a count for how many times they are repeated corresponding to the factors in the product.
        """
        return {"cmpr_factors_and_counts": self.cmpr_factors_and_counts}

    @classmethod
    def resource_rep(cls, cmpr_factors_and_counts) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_factors_and_counts (Tuple[Tuple[~.labs.resource_estimation.CompressedResourceOp, int]]):
                A sequence of tuples containing the operations, in the compressed representation, and
                a count for how many times they are repeated corresponding to the factors in the product.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"cmpr_factors_and_counts": cmpr_factors_and_counts})

    @classmethod
    def default_resource_decomp(cls, cmpr_factors_and_counts, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            cmpr_factors_and_counts (Tuple[Tuple[~.labs.resource_estimation.CompressedResourceOp, int]]):
                A sequence of tuples containing the operations, in the compressed representation, and
                a count for how many times they are repeated corresponding to the factors in the product.

        Resources:
            This symbolic class represents a product of operations. The resources are defined
            trivially as the counts for each operation in the product.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: :class:`~.ops.op_math.prod.Prod`

        **Example**

        The product of operations can be constructed as follows. Note, each operation in the
        product must be a valid :class:`~.pennylane.labs.resource_estimation.ResourceOperator`

        >>> factors = [plre.ResourceX(), plre.ResourceY(), plre.ResourceZ()]
        >>> prod_xyz = plre.ResourceProd(factors)
        >>>
        >>> print(plre.estimate_resources(prod_xyz))
        --- Resources: ---
        Total qubits: 1
        Total gates : 3
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'X': 1, 'Y': 1, 'Z': 1}

        We can also specify the factors as a tuple with

        >>> factors = [(plre.ResourceX(), 2), (plre.ResourceZ(), 3)]
        >>> prod_x2z3 = plre.ResourceProd(factors)
        >>>
        >>> print(plre.estimate_resources(prod_x2z3))
        --- Resources: ---
        Total qubits: 1
        Total gates : 5
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'X': 2, 'Z': 3}

        """
        return [GateCount(cmpr_op, count) for cmpr_op, count in cmpr_factors_and_counts]


class ResourceChangeBasisOp(ResourceOperator):
    r"""Change of Basis resource operator.

    A symbolic class used to represent a change of basis operation. This is a special
    type of operator which can be expressed as
    :math:`\hat{U}_{compute} \cdot \hat{V} \cdot \hat{U}_{uncompute}`. If no :code:`uncompute_op` is
    provided then the adjoint of the :code:`compute_op` is used by default.

    Args:
        compute_op (~.pennylane.labs.resource_estimation.ResourceOperator): A resource operator
            representing the basis change operation.
        base_op (~.pennylane.labs.resource_estimation.ResourceOperator): A resource operator
            representing the base operation.
        uncompute_op (~.pennylane.labs.resource_estimation.ResourceOperator, optional): An optional
            resource operator representing the inverse of the basis change operation.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        This symbolic class represents a product of the three provided operations. The resources are
        defined trivially as the sum of the costs of each.

    .. seealso:: :class:`~.ops.op_math.prod.Prod`

    **Example**

    Note, each operation in the product must be a valid :class:`~.pennylane.labs.resource_estimation.ResourceOperator`
    The change of basis operation can be constructed as follows:

    >>> compute_u = plre.ResourceS()
    >>> base_v = plre.ResourceZ()
    >>> cb_op = plre.ResourceChangeBasisOp(compute_u, base_v)
    >>> print(plre.estimate_resources(cb_op, gate_set={"Z", "S", "Adjoint(S)"}))
    --- Resources: ---
    Total qubits: 1
    Total gates : 3
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'S': 1, 'Z': 1, 'Adjoint(S)': 1}

    We can also set the :code:`uncompute_op` directly.

    >>> uncompute_u = plre.ResourceProd([plre.ResourceZ(), plre.ResourceS()])
    >>> cb_op = plre.ResourceChangeBasisOp(compute_u, base_v, uncompute_u)
    >>> print(plre.estimate_resources(cb_op, gate_set={"Z", "S", "Adjoint(S)"}))
    --- Resources: ---
    Total qubits: 1
    Total gates : 4
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'S': 2, 'Z': 2}

    """

    resource_keys = {"cmpr_compute_op", "cmpr_base_op", "cmpr_uncompute_op"}

    def __init__(
        self,
        compute_op: ResourceOperator,
        base_op: ResourceOperator,
        uncompute_op: Union[None, ResourceOperator] = None,
        wires=None,
    ) -> None:
        uncompute_op = uncompute_op or ResourceAdjoint(compute_op)
        ops_to_remove = [compute_op, base_op, uncompute_op]

        self.queue(ops_to_remove)

        try:
            self.cmpr_compute_op = compute_op.resource_rep_from_op()
            self.cmpr_base_op = base_op.resource_rep_from_op()
            self.cmpr_uncompute_op = uncompute_op.resource_rep_from_op()

        except AttributeError as error:
            raise ValueError(
                "All ops of the ChangeofBasisOp must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in ops_to_remove if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in ops_to_remove))
            else:
                self.wires = Wires.all_wires(ops_wires)
                self.num_wires = len(self.wires)

    def queue(self, ops_to_remove, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        for op in ops_to_remove:
            context.remove(op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * cmpr_compute_op (CompressedResourceOp): A compressed resource operator, corresponding
            to the compute operation.
            * cmpr_base_op (CompressedResourceOp): A compressed resource operator, corresponding
            to the base operation.
            * cmpr_uncompute_op (CompressedResourceOp): A compressed resource operator, corresponding
            to the uncompute operation.

        """
        return {
            "cmpr_compute_op": self.cmpr_compute_op,
            "cmpr_base_op": self.cmpr_base_op,
            "cmpr_uncompute_op": self.cmpr_uncompute_op,
        }

    @classmethod
    def resource_rep(
        cls, cmpr_compute_op, cmpr_base_op, cmpr_uncompute_op=None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_compute_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the compute operation.
            cmpr_base_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the base operation.
            cmpr_uncompute_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the uncompute operation.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        cmpr_uncompute_op = cmpr_uncompute_op or resource_rep(
            ResourceAdjoint, {"base_cmpr_op": cmpr_compute_op}
        )
        return CompressedResourceOp(
            cls,
            {
                "cmpr_compute_op": cmpr_compute_op,
                "cmpr_base_op": cmpr_base_op,
                "cmpr_uncompute_op": cmpr_uncompute_op,
            },
        )

    @classmethod
    def default_resource_decomp(cls, cmpr_compute_op, cmpr_base_op, cmpr_uncompute_op, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            cmpr_compute_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the compute operation.
            cmpr_base_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the base operation.
            cmpr_uncompute_op (CompressedResourceOp): A compressed resource operator, corresponding
                to the uncompute operation.

        Resources:
            This symbolic class represents a product of the three provided operations. The resources are
            defined trivially as the sum of the costs of each.

        .. seealso:: :class:`~.ops.op_math.prod.Prod`

        **Example**

        Note, each operation in the product must be a valid :class:`~.pennylane.labs.resource_estimation.ResourceOperator`
        The change of basis operation can be constructed as follows:

        >>> compute_u = plre.ResourceS()
        >>> base_v = plre.ResourceZ()
        >>> cb_op = plre.ResourceChangeBasisOp(compute_u, base_v)
        >>> print(plre.estimate_resources(cb_op, gate_set={"Z", "S", "Adjoint(S)"}))
        --- Resources: ---
        Total qubits: 1
        Total gates : 3
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'S': 1, 'Z': 1, 'Adjoint(S)': 1}

        We can also set the :code:`uncompute_op` directly.

        >>> uncompute_u = plre.ResourceProd([plre.ResourceZ(), plre.ResourceS()])
        >>> cb_op = plre.ResourceChangeBasisOp(compute_u, base_v, uncompute_u)
        >>> print(plre.estimate_resources(cb_op, gate_set={"Z", "S", "Adjoint(S)"}))
        --- Resources: ---
        Total qubits: 1
        Total gates : 4
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
        Gate breakdown:
        {'S': 2, 'Z': 2}

        """
        return [
            GateCount(cmpr_compute_op),
            GateCount(cmpr_base_op),
            GateCount(cmpr_uncompute_op),
        ]


@singledispatch
def _apply_adj(action):
    raise TypeError(f"Unsupported type {action}")


@_apply_adj.register
def _(action: GateCount):
    gate = action.gate
    return GateCount(resource_rep(ResourceAdjoint, {"base_cmpr_op": gate}), action.count)


@_apply_adj.register
def _(action: AllocWires):
    return FreeWires(action.num_wires)


@_apply_adj.register
def _(action: FreeWires):
    return AllocWires(action.num_wires)
