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
from typing import Dict, List, Union

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import GrabWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    GateCount,
    resource_rep,
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
                * base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
                * base_params (dict): the resource parameters required to extract the cost of the base operator

        """
        return {"base_cmpr_op": self.base_op}

    @classmethod
    def resource_rep(cls, base_cmpr_op) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
            base_params (dict): the resource parameters required to extract the cost of the base operator

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"base_cmpr_op": base_cmpr_op})

    @classmethod
    def default_resource_decomp(cls, base_cmpr_op: CompressedResourceOp, **kwargs):
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
        return [GateCount(base_cmpr_op)]

    @staticmethod
    def tracking_name(base_cmpr_op: CompressedResourceOp) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Adjoint({base_name})"


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
            self.wires = None or base_op.wires
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
                * base_class (Type[~.ResourceOperator]): the class type of the base operator to be controlled
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
        cls, base_cmpr_op, num_ctrl_wires, num_ctrl_values,
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
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
            },
        )

    @classmethod
    def default_resource_decomp(
        cls, base_cmpr_op, num_ctrl_wires, num_ctrl_values, **kwargs
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
        
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        try:
            return base_class.controlled_resource_decomp(
                ctrl_num_ctrl_wires = num_ctrl_wires, ctrl_num_ctrl_values = num_ctrl_values, **base_params
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
                    gate, num_ctrl_wires, num_ctrl_values,
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
                    base_cmpr_op,
                    ctrl_num_ctrl_wires + num_ctrl_wires,
                    ctrl_num_ctrl_values + num_ctrl_values,
                )
            ),
        ]

    @staticmethod
    def tracking_name(
        base_cmpr_op: CompressedResourceOp, num_ctrl_wires: int, num_ctrl_values: int,
    ):
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"C({base_name}, num_ctrl_wires={num_ctrl_wires},num_ctrl_values={num_ctrl_values})"


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
                * base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
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
            base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            z (int): the power that the operator is being raised to

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls, {"base_cmpr_op": base_cmpr_op, "z": z}
        )

    @classmethod
    def default_resource_decomp(
        cls, base_cmpr_op, z, **kwargs
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
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        
        if z == 0:
            return [GateCount(resource_rep(re.ResourceIdentity))]

        if z == 1:
            return [GateCount(base_cmpr_op)]

        try:
            return base_class.pow_resource_decomp(pow_z=z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            gate_lst = []
            decomp = base_class.resources(**base_params, **kwargs)
            for action in decomp:
                if isinstance(action, GateCount):
                    gate = action.gate
                    pow_gate = cls.resource_rep(gate, z)
                    gate_lst.append(pow_gate, action.count)
                else:
                    gate_lst.append(action)

            return gate_lst

        except re.ResourcesNotDefined:
            pass

        return [GateCount(base_cmpr_op, z)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, base_cmpr_op, z):
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
    resource_keys = {"cmpr_factors"}
    
    def __init__(self, res_ops: List[ResourceOperator], wires=None) -> None:
        self.queue(res_ops)

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in res_ops)  
            self.cmpr_ops = cmpr_ops
        except AttributeError as error:
            raise ValueError(
                "All factors of the Product must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        if wires:
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)
        else:
            ops_wires = [op.wires for op in res_ops if op.wires is not None]
            if len(ops_wires) == 0:
                self.wires = None
                self.num_wires = max((op.num_wires for op in res_ops))
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
                * cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed representation, corresponding to the factors in the product.
        """
        return {"cmpr_factors": self.cmpr_ops}

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

    @classmethod
    def default_resource_decomp(cls, cmpr_factors, **kwargs):
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
        return [GateCount(cmpr_op) for cmpr_op in cmpr_factors]


class ResourceChangeBasisOp(ResourceOperator):
    """Change of Basis resource operator """
    resource_keys = {"cmpr_compute_op", "cmpr_base_op", "cmpr_uncompute_op"}
    
    def __init__(
        self, 
        compute_op: ResourceOperator, 
        base_op: ResourceOperator, 
        uncompute_op: Union[None, ResourceOperator] = None, 
        wires = None,
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
                * cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed representation, corresponding to the factors in the product.
        """
        return {
            "cmpr_compute_op": self.cmpr_compute_op,
            "cmpr_base_op": self.cmpr_base_op,
            "cmpr_uncompute_op": self.cmpr_uncompute_op,
        }

    @classmethod
    def resource_rep(cls, cmpr_compute_op, cmpr_base_op, cmpr_uncompute_op=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed
                representation, corresponding to the factors in the product.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        cmpr_uncompute_op = cmpr_uncompute_op or resource_rep(ResourceAdjoint, {"base_cmpr_op": cmpr_compute_op})
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
def _(action: GrabWires):
    return FreeWires(action.n)


@_apply_adj.register
def _(action: FreeWires):
    return GrabWires(action.n)
