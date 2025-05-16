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
from collections import defaultdict
from functools import singledispatch
from typing import Dict

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resources_base import  _scale_dict
from pennylane.labs.resource_estimation.qubit_manager import GrabWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    CompressedResourceOp,
)

from pennylane.queuing import QueuingManager
from pennylane.wires import Wires

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments,too-many-positional-arguments


class ResourceAdjoint(ResourceOperator):
    r"""Resource class for the symbolic AdjointOperation.

    A symbolic class used to represent the adjoint of some base operation.

    Args:
        base (~.operation.Operator): The operator that we want the adjoint of.

    Resource Parameters:
        * base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
        * base_params (dict): the resource parameters required to extract the cost of the base operator

    Resources:
        This symbolic operation represents the adjoint of some base operation. The resources are
        determined as follows. If the base operation class :code:`base_class` implements the
        :code:`.adjoint_resource_decomp()` method, then the resources are obtained from this.

        Otherwise, the adjoint resources are given as the adjoint of each operation in the
        base operation's resources (via :code:`.resources()`).

    .. seealso:: :class:`~.ops.op_math.adjoint.AdjointOperation`

    **Example**

    The adjoint operation can be constructed like this:

    >>> qft = re.ResourceQFT(wires=range(3))
    >>> adjoint_qft = re.ResourceAdjoint(qft)
    >>> adjoint_qft.resources(**adjoint_qft.resource_params)
    defaultdict(<class 'int'>, {Adjoint(Hadamard): 3, Adjoint(SWAP): 1,
    Adjoint(ControlledPhaseShift): 3})

    Alternatively, we can call the resources method on from the class:

    >>> re.ResourceAdjoint.resources(
    ...     base_class = re.ResourceQFT,
    ...     base_params = {"num_wires": 3},
    ... )
    defaultdict(<class 'int'>, {Adjoint(Hadamard): 3, Adjoint(SWAP): 1,
    Adjoint(ControlledPhaseShift): 3})

    .. details::
        :title: Usage Details

        We can configure the resources for the adjoint of a base operation by modifying
        its :code:`.adjoint_resource_decomp(**resource_params)` method. Consider for example this
        custom PauliZ class, where the adjoint resources are not defined (this is the default
        for a general :class:`~.ResourceOperator`).

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def adjoint_resource_decomp(cls):
                    raise re.ResourcesNotDefined

        When this method is not defined, the adjoint resources are computed by taking the
        adjoint of the resources of the operation.

        >>> CustomZ.resources()
        {S: 2}
        >>> re.ResourceAdjoint.resources(CustomZ, {})
        defaultdict(<class 'int'>, {Adjoint(S): 2})

        We can update the adjoint resources with the observation that the PauliZ gate is self-adjoint,
        so the resources should just be the same as the base operation:

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def adjoint_resource_decomp(cls):
                    return {cls.resource_rep(): 1}

        >>> re.ResourceAdjoint.resources(CustomZ, {})
        {CustomZ: 1}

    """

    resource_keys = {"base_cmpr_op", "base_resources"}

    def __init__(self, base_op: ResourceOperator, wires=None) -> None:
        self.queue(remove_op=base_op)
        base_op = base_op.resource_rep_from_op()
        self.queue()

        self.base_op = base_op

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            self.num_wires = base_op.num_wires

    def queue(self, remove_op=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if remove_op:
            context.remove(remove_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
                * base_params (dict): the resource parameters required to extract the cost of the base operator

        """
        return {"base_cmpr_op": self.base_op, "base_resources": None}

    @classmethod
    def default_resource_decomp(cls, base_cmpr_op, base_resources, **kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
            base_params (dict): the resource parameters required to extract the cost of the base operator

        Resources:
            This symbolic operation represents the adjoint of some base operation. The resources are
            determined as follows. If the base operation class :code:`base_class` implements the
            :code:`.adjoint_resource_decomp()` method, then the resources are obtained from this.

            Otherwise, the adjoint resources are given as the adjoint of each operation in the
            base operation's resources (via :code:`.resources()`).

        **Example**

        The adjoint operation can be constructed like this:

        >>> qft = re.ResourceQFT(wires=range(3))
        >>> adjoint_qft = re.ResourceAdjoint(qft)
        >>> adjoint_qft.resources(**adjoint_qft.resource_params)
        defaultdict(<class 'int'>, {Adjoint(Hadamard): 3, Adjoint(SWAP): 1,
        Adjoint(ControlledPhaseShift): 3})

        Alternatively, we can call the resources method on from the class:

        >>> re.ResourceAdjoint.resources(
        ...     base_class = re.ResourceQFT,
        ...     base_params = {"num_wires": 3},
        ... )
        defaultdict(<class 'int'>, {Adjoint(Hadamard): 3, Adjoint(SWAP): 1,
        Adjoint(ControlledPhaseShift): 3})

        .. details::
            :title: Usage Details

            We can configure the resources for the adjoint of a base operation by modifying
            its :code:`.adjoint_resource_decomp(**resource_params)` method. Consider for example this
            custom PauliZ class, where the adjoint resources are not defined (this is the default
            for a general :class:`~.ResourceOperator`).

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def adjoint_resource_decomp(cls):
                        raise re.ResourcesNotDefined

            When this method is not defined, the adjoint resources are computed by taking the
            adjoint of the resources of the operation.

            >>> CustomZ.resources()
            {S: 2}
            >>> re.ResourceAdjoint.resources(CustomZ, {})
            defaultdict(<class 'int'>, {Adjoint(S): 2})

            We can update the adjoint resources with the observation that the PauliZ gate is self-adjoint,
            so the resources should just be the same as the base operation:

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def adjoint_resource_decomp(cls):
                        return {cls.resource_rep(): 1}

            >>> re.ResourceAdjoint.resources(CustomZ, {})
            {CustomZ: 1}
        """
        if base_cmpr_op:  # we have a base compressed op
            base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
            try:
                return base_class.adjoint_resource_decomp(**base_params)
            except ResourcesNotDefined:
                gate_lst = []
                decomp = base_class.resource_decomp(**base_params, **kwargs)

                for gate in decomp[::-1]:  # reverse the order
                    gate_lst.append(_apply_adj(gate))
                return gate_lst
        
        if base_resources:
            for cmpr_op, count in base_resources.gate_types.items():
                adj_cmpr_op = cls.resource_rep(base_cmpr_op=cmpr_op)
                gate_lst.append(GateCount(adj_cmpr_op, count))
        
        raise ResourcesNotDefined

    @classmethod
    def resource_rep(cls, base_cmpr_op=None, base_resources=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
            base_params (dict): the resource parameters required to extract the cost of the base operator

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"base_cmpr_op": base_cmpr_op, "base_resources": base_resources})

    @staticmethod
    def default_adjoint_resource_decomp(base_cmpr_op, base_resources):
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
            base_params (dict): the resource parameters required to extract the cost of the base operator

        Resources:
            The adjoint of an adjointed operation is just the original operation. The resources
            are given as one instance of the base operation.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        if base_cmpr_op:
            return [GateCount(base_cmpr_op)]

        gate_lst = []
        for cmpr_op, count in base_resources.gate_types.items():
            adj_cmpr_op = re.ResourceAdjoint.resource_rep(base_cmpr_op=cmpr_op)
            gate_lst.append(GateCount(adj_cmpr_op, count))
        return gate_lst

    @staticmethod
    def tracking_name(base_cmpr_op, base_resources) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        if base_cmpr_op:
            base_name = base_cmpr_op.op_type.tracking_name(**base_cmpr_op.params)
            return f"Adjoint({base_name})"
        return f"Adjoint({base_resources})"


class ResourceControlled(ResourceOperator):
    r"""Resource class for the symbolic ControlledOp.

    A symbolic class used to represent the application of some base operation controlled on the state
    of some control qubits.

    Args:
        base (~.operation.Operator): the operator that is controlled
        control_wires (Any): The wires to control on.
        control_values (Iterable[Bool]): The values to control on. Must be the same
            length as ``control_wires``. Defaults to ``True`` for all control wires.
            Provided values are converted to `Bool` internally.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Resource Parameters:
        * base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
        * base_params (dict): the resource parameters required to extract the cost of the base operator
        * num_ctrl_wires (int): the number of qubits the operation is controlled on
        * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        * num_work_wires (int): the number of additional qubits that can be used for decomposition

    Resources:
        The resources are determined as follows. If the base operation class :code:`base_class`
        implements the :code:`.controlled_resource_decomp()` method, then the resources are obtained
        directly from this.

        Otherwise, the controlled resources are given in two steps. Firstly, any control qubits which
        should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds to an additional
        cost of two :class:`~.ResourceX` gates per :code:`num_ctrl_values`. Secondly, the base operation
        resources are extracted (via :code:`.resources()`) and we add to the cost the controlled
        variant of each operation in the resources.

    .. seealso:: :class:`~.ops.op_math.controlled.ControlledOp`

    **Example**

    The controlled operation can be constructed like this:

    >>> qft = re.ResourceQFT(wires=range(3))
    >>> controlled_qft = re.ResourceControlled(
    ...    qft, control_wires=['c0', 'c1', 'c2'], control_values=[1, 1, 1], work_wires=['w1', 'w2'],
    ... )
    >>> controlled_qft.resources(**controlled_qft.resource_params)
    defaultdict(<class 'int'>, {C(Hadamard,3,0,2): 3, C(SWAP,3,0,2): 1, C(ControlledPhaseShift,3,0,2): 3})

    Alternatively, we can call the resources method on from the class:

    >>> re.ResourceControlled.resources(
    ...     base_class = re.ResourceQFT,
    ...     base_params = {"num_wires": 3},
    ...     num_ctrl_wires = 3,
    ...     num_ctrl_values = 0,
    ...     num_work_wires = 2,
    ... )
    defaultdict(<class 'int'>, {C(Hadamard,3,0,2): 3, C(SWAP,3,0,2): 1, C(ControlledPhaseShift,3,0,2): 3})

    .. details::
        :title: Usage Details

        We can configure the resources for the controlled of a base operation by modifying
        its :code:`.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires,
        **resource_params)` method. Consider for example this custom PauliZ class, where the
        controlled resources are not defined (this is the default for a general :class:`~.ResourceOperator`).


        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def controlled_resource_decomp(cls, num_ctrl_wires, num_ctrl_values, num_work_wires):
                    raise re.ResourcesNotDefined

        When this method is not defined, the controlled resources are computed by taking the
        controlled of each operation in the resources of the base operation.

        >>> CustomZ.resources()
        {S: 2}
        >>> re.ResourceControlled.resources(CustomZ, {}, num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0)
        defaultdict(<class 'int'>, {C(S,2,0,3): 2})

        We can update the controlled resources with the observation that the PauliZ gate when controlled
        on a single wire is equivalent to :math:`\hat{CZ} = \hat{H} \cdot \hat{CNOT} \cdot \hat{H}`.
        so we can modify the base operation:

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def controlled_resource_decomp(cls, num_ctrl_wires, num_ctrl_values, num_work_wires):
                    if num_ctrl_wires == 1 and num_ctrl_values == 0:
                        return {
                            re.ResourceHadamard.resource_rep(): 2,
                            re.ResourceCNOT.resource_rep(): 1,
                        }
                    raise re.ResourcesNotDefined

        >>> re.ResourceControlled.resources(CustomZ, {}, num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0)
        {Hadamard: 2, CNOT: 1}

    """

    resource_keys = {"base_cmpr_op", "base_resources", "num_ctrl_wires", "num_ctrl_values"}

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
            base_wires = base_op.num_wires
            self.num_wires = num_ctrl_wires + base_wires

    def queue(self, remove_base_op=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if remove_base_op:
            context.remove(remove_base_op)
        context.append(self)
        return self

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * num_ctrl_wires (int): the number of qubits the operation is controlled on
                * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
                * num_work_wires (int): the number of additional qubits that can be used for decomposition
        """

        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "num_ctrl_values": self.num_ctrl_values,
            "base_cmpr_op": self.base_op,
            "base_resources": None,
        }

    @classmethod
    def resource_rep(
        cls, num_ctrl_wires, num_ctrl_values, base_cmpr_op=None, base_resources=None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
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
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "base_cmpr_op": base_cmpr_op,
                "base_resources": base_resources,
            },
        )

    @classmethod
    def default_resource_decomp(
        cls, base_cmpr_op, base_resources, num_ctrl_wires, num_ctrl_values, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
            base_params (dict): the resource parameters required to extract the cost of the base operator
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_work_wires (int): the number of additional qubits that can be used for decomposition

        Resources:
            The resources are determined as follows. If the base operation class :code:`base_class`
            implements the :code:`.controlled_resource_decomp()` method, then the resources are obtained
            directly from this.

            Otherwise, the controlled resources are given in two steps. Firstly, any control qubits which
            should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds to an additional
            cost of two :class:`~.ResourceX` gates per :code:`num_ctrl_values`. Secondly, the base operation
            resources are extracted (via :code:`.resources()`) and we add to the cost the controlled
            variant of each operation in the resources.

        .. seealso:: :class:`~.ops.op_math.controlled.ControlledOp`

        **Example**

        The controlled operation can be constructed like this:

        >>> qft = re.ResourceQFT(wires=range(3))
        >>> controlled_qft = re.ResourceControlled(
        ...    qft, control_wires=['c0', 'c1', 'c2'], control_values=[1, 1, 1], work_wires=['w1', 'w2'],
        ... )
        >>> controlled_qft.resources(**controlled_qft.resource_params)
        defaultdict(<class 'int'>, {C(Hadamard,3,0,2): 3, C(SWAP,3,0,2): 1, C(ControlledPhaseShift,3,0,2): 3})

        Alternatively, we can call the resources method on from the class:

        >>> re.ResourceControlled.resources(
        ...     base_class = re.ResourceQFT,
        ...     base_params = {"num_wires": 3},
        ...     num_ctrl_wires = 3,
        ...     num_ctrl_values = 0,
        ...     num_work_wires = 2,
        ... )
        defaultdict(<class 'int'>, {C(Hadamard,3,0,2): 3, C(SWAP,3,0,2): 1, C(ControlledPhaseShift,3,0,2): 3})

        .. details::
            :title: Usage Details

            We can configure the resources for the controlled of a base operation by modifying
            its :code:`.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires,
            **resource_params)` method. Consider for example this custom PauliZ class, where the
            controlled resources are not defined (this is the default for a general :class:`~.ResourceOperator`).

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def controlled_resource_decomp(cls, num_ctrl_wires, num_ctrl_values, num_work_wires):
                        raise re.ResourcesNotDefined

            When this method is not defined, the controlled resources are computed by taking the
            controlled of each operation in the resources of the base operation.

            >>> CustomZ.resources()
            {S: 2}
            >>> re.ResourceControlled.resources(CustomZ, {}, num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0)
            defaultdict(<class 'int'>, {C(S,2,0,3): 2})

            We can update the controlled resources with the observation that the PauliZ gate when controlled
            on a single wire is equivalent to :math:`\hat{CZ} = \hat{H} \cdot \hat{CNOT} \cdot \hat{H}`.
            so we can modify the base operation:

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def controlled_resource_decomp(cls, num_ctrl_wires, num_ctrl_values, num_work_wires):
                        if num_ctrl_wires == 1 and num_ctrl_values == 0:
                            return {
                                re.ResourceHadamard.resource_rep(): 2,
                                re.ResourceCNOT.resource_rep(): 1,
                            }
                        raise re.ResourcesNotDefined

            >>> re.ResourceControlled.resources(CustomZ, {}, num_ctrl_wires=1, num_ctrl_values=0, num_work_wires=0)
            {Hadamard: 2, CNOT: 1}

        """
        
        if base_cmpr_op:
            base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
            try:
                return base_class.controlled_resource_decomp(
                    ctrl_num_ctrl_wires = num_ctrl_wires, ctrl_num_ctrl_values = num_ctrl_values, **base_params
                )
            except re.ResourcesNotDefined:
                pass

            gate_lst = []

            if num_ctrl_values == 0:
                decomp = base_class.resource_decomp(**base_params, **kwargs)
                for action in decomp:
                    if isinstance(action, GateCount):
                        gate = action.gate
                        c_gate = cls.resource_rep(
                            num_ctrl_wires, num_ctrl_values, base_cmpr_op=gate,
                        )
                        gate_lst.append(GateCount(c_gate, action.count))

                    else:
                        gate_lst.append(action)

                return gate_lst

            no_control = cls.resource_rep(base_class, base_params, num_ctrl_wires, 0, num_work_wires)
            x = re.ResourceX.resource_rep()

            gate_lst.append(GateCount(no_control))
            gate_lst.append(GateCount(x, 2 * num_ctrl_values))
            return gate_lst

    @classmethod
    def controlled_resource_decomp(
        cls,
        outer_num_ctrl_wires,
        outer_num_ctrl_values,
        outer_num_work_wires,
        base_class,
        base_params,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for a controlled version of the operator.

        Args:
            outer_num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            outer_num_ctrl_values (int): The subset of those control qubits, which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            outer_num_work_wires (int): the number of additional qubits that can be used in the
                decomposition for the further controlled, base control oepration.
            base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
            base_params (dict): the resource parameters required to extract the cost of the base operator
            num_ctrl_wires (int): the number of control qubits of the operation
            num_ctrl_values (int): The subset of control qubits of the operation, that are controlled
                when in the :math:`|0\rangle` state.
            num_work_wires (int): The number of additional qubits that can be used for the
                decomposition of the operation.

        Resources:
            The resources are derived by simply combining the control qubits, control-values and
            work qubits into a single instance of :class:`~.ResourceControlled` gate, controlled
            on the whole set of control-qubits.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [
            GateCount(
                cls.resource_rep(
                    base_class,
                    base_params,
                    outer_num_ctrl_wires + num_ctrl_wires,
                    outer_num_ctrl_values + num_ctrl_values,
                    outer_num_work_wires + num_work_wires,
                )
            ),
        ]

    @staticmethod
    def tracking_name(num_ctrl_wires, num_ctrl_values, base_cmpr_op, base_resources):
        r"""Returns the tracking name built with the operator's parameters."""
        if base_cmpr_op:
            base_name = base_cmpr_op.op_type.tracking_name(**base_cmpr_op.params)
            return f"C({base_name},{num_ctrl_wires},{num_ctrl_values})"
        
        return f"C({base_resources},{num_ctrl_wires},{num_ctrl_values})"


class ResourcePow(ResourceOperator):
    r"""Resource class for the symbolic Pow operation.

    A symbolic class used to represent some base operation raised to a power.

    Args:
        base (~.operation.Operator): the operator to be raised to a power
        z (float): the exponent (default value is 1)

    Resource Parameters:
        * base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
        * base_params (dict): the resource parameters required to extract the cost of the base operator
        * z (int): the power that the operator is being raised to

    Resources:
        The resources are determined as follows. If the power :math:`z = 0`, then we have the identitiy
        gate and we have no resources. If the base operation class :code:`base_class` implements the
        :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
        the resources of the operation raised to the power :math:`z` are given by extracting the base
        operation's resources (via :code:`.resources()`) and raising each operation to the same power.

    .. seealso:: :class:`~.ops.op_math.pow.PowOperation`

    **Example**

    The operation raised to a power :math:`z` can be constructed like this:

    >>> qft = re.ResourceQFT(wires=range(3))
    >>> pow_qft = re.ResourcePow(qft, 2)
    >>> pow_qft.resources(**pow_qft.resource_params)
    defaultdict(<class 'int'>, {Pow(Hadamard, 2): 3, Pow(SWAP, 2): 1, Pow(ControlledPhaseShift, 2): 3})

    Alternatively, we can call the resources method on from the class:

    >>> re.ResourcePow.resources(
    ...     base_class = re.ResourceQFT,
    ...     base_params = {"num_wires": 3},
    ...     z = 2,
    ... )
    defaultdict(<class 'int'>, {Pow(Hadamard, 2): 3, Pow(SWAP, 2): 1, Pow(ControlledPhaseShift, 2): 3})

    .. details::
        :title: Usage Details

        We can configure the resources for the power of a base operation by modifying
        its :code:`.pow_resource_decomp(**resource_params, z)` method. Consider for example this
        custom PauliZ class, where the pow-resources are not defined (this is the default
        for a general :class:`~.ResourceOperator`).

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def pow_resource_decomp(cls, z):
                    raise re.ResourcesNotDefined

        When this method is not defined, the resources are computed by taking the power of
        each operation in the resources of the base operation.

        >>> CustomZ.resources()
        {S: 2}
        >>> re.ResourcePow.resources(CustomZ, {}, z=2)
        defaultdict(<class 'int'>, {Pow(S, 2): 2})

        We can update the resources with the observation that the PauliZ gate is self-inverse,
        so the resources should when :math:`z mod 2 = 0` should just be the identity operation:

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def pow_resource_decomp(cls, z):
                    if z%2 == 0:
                        return {re.ResourceIdentity.resource_rep(): 1}
                    return {cls.resource_rep(): 1}

        >>> re.ResourcePow.resources(CustomZ, {}, z=2)
        {Identity: 1}
        >>> re.ResourcePow.resources(CustomZ, {}, z=3)
        {CustomZ: 1}

    """

    def __init__(self, base_op: CompressedResourceOp, z: int, wires=None) -> None:
        if isinstance(base_op, ResourceOperator):
            self.queue(base_op)
            base_op = base_op.resource_rep_from_op()
        else:
            self.queue()

        self.z = z
        self.base_op = base_op

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            self.wires = None
            if isinstance(base_op, ResourceOperator):
                self.num_wires = base_op.num_wires
            else:
                self.num_wires = (
                    base_op.params["num_wires"]
                    if "num_wires" in base_op.params
                    else base_op.op_type.num_wires
                )

    def queue(self, base_op=None, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        if base_op:
            context.remove(base_op)
        context.append(self)
        return self

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, z, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            z (int): the power that the operator is being raised to

        Resources:
            The resources are determined as follows. If the power :math:`z = 0`, then we have the identitiy
            gate and we have no resources. If the base operation class :code:`base_class` implements the
            :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
            the resources of the operation raised to the power :math:`z` are given by extracting the base
            operation's resources (via :code:`.resources()`) and raising each operation to the same power.

        **Example**

        The operation raised to a power :math:`z` can be constructed like this:

        >>> qft = re.ResourceQFT(wires=range(3))
        >>> pow_qft = re.ResourcePow(qft, 2)
        >>> pow_qft.resources(**pow_qft.resource_params)
        defaultdict(<class 'int'>, {Pow(Hadamard, 2): 3, Pow(SWAP, 2): 1, Pow(ControlledPhaseShift, 2): 3})

        Alternatively, we can call the resources method on from the class:

        >>> re.ResourcePow.resources(
        ...     base_class = re.ResourceQFT,
        ...     base_params = {"num_wires": 3},
        ...     z = 2,
        ... )
        defaultdict(<class 'int'>, {Pow(Hadamard, 2): 3, Pow(SWAP, 2): 1, Pow(ControlledPhaseShift, 2): 3})

        .. details::
            :title: Usage Details

            We can configure the resources for the power of a base operation by modifying
            its :code:`.pow_resource_decomp(**resource_params, z)` method. Consider for example this
            custom PauliZ class, where the pow-resources are not defined (this is the default
            for a general :class:`~.ResourceOperator`).

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def pow_resource_decomp(cls, z):
                        raise re.ResourcesNotDefined

            When this method is not defined, the resources are computed by taking the power of
            each operation in the resources of the base operation.

            >>> CustomZ.resources()
            {S: 2}
            >>> re.ResourcePow.resources(CustomZ, {}, z=2)
            defaultdict(<class 'int'>, {Pow(S, 2): 2})

            We can update the resources with the observation that the PauliZ gate is self-inverse,
            so the resources should when :math:`z mod 2 = 0` should just be the identity operation:

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def pow_resource_decomp(cls, z):
                        if z%2 == 0:
                            return {re.ResourceIdentity.resource_rep(): 1}
                        return {cls.resource_rep(): 1}

            >>> re.ResourcePow.resources(CustomZ, {}, z=2)
            {Identity: 1}
            >>> re.ResourcePow.resources(CustomZ, {}, z=3)
            {CustomZ: 1}

        """
        if z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep())]

        if z == 1:
            return [GateCount(base_class.resource_rep(**base_params))]

        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            gate_lst = []
            decomp = base_class.resources(**base_params, **kwargs)
            for action in decomp:
                if isinstance(action, GateCount):
                    gate = action.gate
                    pow_gate = cls.resource_rep(gate.op_type, gate.params, z)
                    gate_lst.append(pow_gate, action.count)
                else:
                    gate_lst.append(action)

            return gate_lst

        except re.ResourcesNotDefined:
            pass

        return [GateCount(base_class.resource_rep(**base_params), z)]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * z (int): the power that the operator is being raised to
        """
        return {
            "base_class": self.base_op.op_type,
            "base_params": self.base_op.params,
            "z": self.z,
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, z) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            z (int): the power that the operator is being raised to

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls, {"base_class": base_class, "base_params": base_params, "z": z}
        )

    @classmethod
    def pow_resource_decomp(cls, z0, base_class, base_params, z) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z0 (int): the power that the power-operator is being raised to
            base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): The resource parameters required to extract the cost of the base operator.
            z (int): the power that the base operator is being raised to

        Resources:
            The resources are derived by simply adding together the :math:`z` exponent and the
            :math:`z_{0}` exponent into a single instance of :class:`~.ResourcePow` gate, raising
            the base operator to the power :math:`z + z_{0}`.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return [GateCount(cls.resource_rep(base_class, base_params, z0 * z))]

    @staticmethod
    def tracking_name(base_class, base_params, z) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_class.tracking_name(**base_params)
        return f"Pow({base_name}, {z})"


class ResourceProd(ResourceOperator):
    r"""Resource class for the symbolic Prod operation.

    A symbolic class used to represent a product of some base operations.

    Args:
        *factors (tuple[~.operation.Operator]): a tuple of operators which will be multiplied together.

    Resource Parameters:
        * cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed representation, corresponding to the factors in the product.

    Resources:
        This symbolic class represents a product of operations. The resources are defined trivially as the counts for each operation in the product.

    .. seealso:: :class:`~.ops.op_math.prod.Prod`

    **Example**

    The product of operations can be constructed as follows. Note, each operation in the
    product must be a valid :class:`~.ResourceOperator`

    >>> prod_op = re.ResourceProd(
    ...     re.ResourceQFT(range(3)),
    ...     re.ResourceZ(0),
    ...     re.ResourceGlobalPhase(1.23, wires=[1])
    ... )
    >>> prod_op
    ResourceQFT(wires=[0, 1, 2]) @ Z(0) @ ResourceGlobalPhase(1.23, wires=[1])
    >>> prod_op.resources(**prod_op.resource_params)
    defaultdict(<class 'int'>, {QFT(3): 1, Z: 1, GlobalPhase: 1})

    """

    @staticmethod
    def _resource_decomp(cmpr_factors, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed
                representation, corresponding to the factors in the product.

        Resources:
            This symbolic class represents a product of operations. The resources are defined
                trivially as the counts for each operation in the product.

        .. seealso:: :class:`~.ops.op_math.prod.Prod`

        **Example**

        The product of operations can be constructed as follows. Note, each operation in the
        product must be a valid :class:`~.ResourceOperator`

        >>> prod_op = re.ResourceProd(
        ...     re.ResourceQFT(range(3)),
        ...     re.ResourceZ(0),
        ...     re.ResourceGlobalPhase(1.23, wires=[1])
        ... )
        >>> prod_op
        ResourceQFT(wires=[0, 1, 2]) @ Z(0) @ ResourceGlobalPhase(1.23, wires=[1])
        >>> prod_op.resources(**prod_op.resource_params)
        defaultdict(<class 'int'>, {QFT(3): 1, Z: 1, GlobalPhase: 1})

        """
        res = defaultdict(int)
        for factor in cmpr_factors:
            res[factor] += 1
        return res

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed representation, corresponding to the factors in the product.
        """
        try:
            cmpr_factors = tuple(factor.resource_rep_from_op() for factor in self.operands)
        except AttributeError as error:
            raise ValueError(
                "All factors of the Product must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        return {"cmpr_factors": cmpr_factors}

    @classmethod
    def resource_rep(cls, cmpr_factors) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed
                representation, corresponding to the factors in the product.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"cmpr_factors": cmpr_factors})


@singledispatch
def _apply_adj(action):
    raise TypeError(f"Unsupported type {action}")


@_apply_adj.register
def _(action: GateCount):
    gate = action.gate
    return GateCount(ResourceAdjoint.resource_rep(gate.op_type, gate.params), action.count)


@_apply_adj.register
def _(action: GrabWires):
    return FreeWires(action.n)


@_apply_adj.register
def _(action: FreeWires):
    return GrabWires(action.n)
