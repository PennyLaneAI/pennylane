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
from typing import Dict

import pennylane.labs.resource_estimation as re
from pennylane import math
from pennylane.labs.resource_estimation.resource_container import _scale_dict
from pennylane.operation import Operation
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.exp import Exp
from pennylane.ops.op_math.pow import PowOperation
from pennylane.ops.op_math.prod import Prod
from pennylane.pauli import PauliSentence

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments,too-many-positional-arguments


class ResourceAdjoint(AdjointOperation, re.ResourceOperator):
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

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
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
        try:
            return base_class.adjoint_resource_decomp(**base_params)
        except re.ResourcesNotDefined:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params)
                gate_types[rep] = count

            return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
                * base_params (dict): the resource parameters required to extract the cost of the base operator

        """
        return {"base_class": type(self.base), "base_params": self.base.resource_params}

    @classmethod
    def resource_rep(cls, base_class, base_params) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): the class type of the base operator that we want the adjoint of
            base_params (dict): the resource parameters required to extract the cost of the base operator

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return re.CompressedResourceOp(cls, {"base_class": base_class, "base_params": base_params})

    @staticmethod
    def adjoint_resource_decomp(base_class, base_params) -> Dict[re.CompressedResourceOp, int]:
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
        return {base_class.resource_rep(**base_params): 1}

    @staticmethod
    def tracking_name(base_class, base_params) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_class.tracking_name(**base_params)
        return f"Adjoint({base_name})"


class ResourceControlled(ControlledOp, re.ResourceOperator):
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

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
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
        try:
            return base_class.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **base_params
            )
        except re.ResourcesNotDefined:
            pass

        gate_types = defaultdict(int)

        if num_ctrl_values == 0:
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params, num_ctrl_wires, 0, num_work_wires)
                gate_types[rep] = count

            return gate_types

        no_control = cls.resource_rep(base_class, base_params, num_ctrl_wires, 0, num_work_wires)
        x = re.ResourceX.resource_rep()
        gate_types[no_control] = 1
        gate_types[x] = 2 * num_ctrl_values

        return gate_types

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
            "base_class": type(self.base),
            "base_params": self.base.resource_params,
            "num_ctrl_wires": len(self.control_wires),
            "num_ctrl_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> re.CompressedResourceOp:
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
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "num_work_wires": num_work_wires,
            },
        )

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
    ) -> Dict[re.CompressedResourceOp, int]:
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
        return {
            cls.resource_rep(
                base_class,
                base_params,
                outer_num_ctrl_wires + num_ctrl_wires,
                outer_num_ctrl_values + num_ctrl_values,
                outer_num_work_wires + num_work_wires,
            ): 1
        }

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires):
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_class.tracking_name(**base_params)
        return f"C({base_name},{num_ctrl_wires},{num_ctrl_values},{num_work_wires})"


class ResourcePow(PowOperation, re.ResourceOperator):
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

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, z, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
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
            return {re.ResourceIdentity.resource_rep(): 1}

        if z == 1:
            return {base_class.resource_rep(**base_params): 1}

        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params, z)
                gate_types[rep] = count

            return gate_types
        except re.ResourcesNotDefined:
            pass

        return {base_class.resource_rep(**base_params): z}

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
            "base_class": type(self.base),
            "base_params": self.base.resource_params,
            "z": self.z,
        }

    @classmethod
    def resource_rep(cls, base_class, base_params, z) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            z (int): the power that the operator is being raised to

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "base_params": base_params, "z": z}
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, base_params, z
    ) -> Dict[re.CompressedResourceOp, int]:
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
        return {cls.resource_rep(base_class, base_params, z0 * z): 1}

    @staticmethod
    def tracking_name(base_class, base_params, z) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_class.tracking_name(**base_params)
        return f"Pow({base_name}, {z})"


class ResourceExp(Exp, re.ResourceOperator):
    r"""Resource class for the symbolic Exp operation.

    A symbolic class used to represent the exponential of some base operation.

    Args:
        base (~.operation.Operator): The operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    Resource Parameters:
        * base_class (Type[~.ResourceOperator]): The class type of the base operator that is exponentiated.
        * base_params (dict): the resource parameters required to extract the cost of the base operator
        * base_pauli_rep (Union[PauliSentence, None]): The base operator represented as a linear combination of Pauli words. If such a representation is not applicable, then :code:`None`.
        * coeff (complex): a scalar value which multiplies the base operator in the exponent
        * num_steps (int): the number of trotter steps to use in approximating the exponential

    Resources:
        This symbolic operation represents the exponential of some base operation. The resources
        are determined as follows. If the base operation class :code:`base_class` implements the
        :code:`.exp_resource_decomp()` method, then the resources are obtained from this.

        Otherwise, the exponetiated operator's resources are computed using the linear combination
        of Pauli words representation (:code:`base_pauli_rep`). The exponential is approximated by
        the product of the exponential of each Pauli word in the sum. This product is repeated
        :code:`num_steps` many times. Specifically, the cost for the exponential of each Pauli word
        is given by an associated :class:`~.ResourcePauliRot`.

    .. seealso:: :class:`~.ops.op_math.exp.Exp`

    **Example**

    The exponentiated operation can be constructed like this:

    >>> hamiltonian = qml.dot([0.1, -2.3], [qml.X(0)@qml.Y(1), qml.Z(0)])
    >>> hamiltonian
    0.1 * (X(0) @ Y(1)) + -2.3 * Z(0)
    >>> exp_hamiltonian = re.ResourceExp(hamiltonian, 0.1*1j, num_steps=2)
    >>> exp_hamiltonian.resources(**exp_hamiltonian.resource_params)
    defaultdict(<class 'int'>, {PauliRot: 2, PauliRot: 2})

    Alternatively, we can call the resources method on from the class:

    >>> re.ResourceExp.resources(
    ...     base_class = qml.ops.Sum,
    ...     base_params = {},
    ...     base_pauli_rep = hamiltonian.pauli_rep,
    ...     coeff = 0.1*1j,
    ...     num_steps = 2,
    ... )
    defaultdict(<class 'int'>, {PauliRot: 2, PauliRot: 2})

    .. details::
        :title: Usage Details

        We can configure the resources for the exponential of a base operation by modifying
        its :code:`.exp_resource_decomp(scalar, num_steps, **resource_params)` method. Consider
        for example this custom PauliZ class, where the exponentiated resources are not defined
        (this is the default for a general :class:`~.ResourceOperator`).

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def exp_resource_decomp(cls, scalar, num_steps):
                    raise re.ResourcesNotDefined

        When this method is not defined, the resources are computed from the linear combination
        of Pauli words representation.

        >>> pauli_rep = CustomZ(wires=0).pauli_rep
        >>> pauli_rep
        1.0 * Z(0)
        >>> re.ResourceExp.resources(CustomZ, {}, base_pauli_rep=pauli_rep, coeff=0.1*1j, num_steps=3)
        defaultdict(<class 'int'>, {PauliRot: 3})

        We can update the exponential resources with the observation that the PauliZ gate, when
        exponentiated, produces an RZ rotation:

        .. code-block:: python

            class CustomZ(re.ResourceZ):

                @classmethod
                def exp_resource_decomp(cls, scalar, num_steps):
                    return {re.ResourceRZ.resource_rep(): num_steps}

        >>> re.ResourceExp.resources(CustomZ, {}, base_pauli_rep=pauli_rep, coeff=0.1*1j, num_steps=3)
        {RZ: 3}

    """

    @staticmethod
    def _resource_decomp(
        base_class: Operation,
        base_params: Dict,
        base_pauli_rep: PauliSentence,
        coeff: complex,
        num_steps: int,
        **kwargs,
    ):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            base_class (Type[~.ResourceOperator]): The class type of the base operator that is
                exponentiated.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            base_pauli_rep (Union[PauliSentence, None]): The base operator represented as a linear
                combination of Pauli words. If such a representation is not applicable, then :code:`None`.
            coeff (complex): a scalar value which multiplies the base operator in the exponent
            num_steps (int): the number of trotter steps to use in approximating the exponential

        Resources:
            This symbolic operation represents the exponential of some base operation. The resources
            are determined as follows. If the base operation class :code:`base_class` implements the
            :code:`.exp_resource_decomp()` method, then the resources are obtained from this.

            Otherwise, the exponetiated operator's resources are computed using the linear combination
            of Pauli words representation (:code:`base_pauli_rep`). The exponential is approximated by
            the product of the exponential of each Pauli word in the sum. This product is repeated
            :code:`num_steps` many times. Specifically, the cost for the exponential of each Pauli word
            is given by an associated :class:`~.ResourcePauliRot`.

        **Example**

        The exponentiated operation can be constructed like this:

        >>> hamiltonian = qml.dot([0.1, -2.3], [qml.X(0)@qml.Y(1), qml.Z(0)])
        >>> hamiltonian
        0.1 * (X(0) @ Y(1)) + -2.3 * Z(0)
        >>> exp_hamiltonian = re.ResourceExp(hamiltonian, 0.1*1j, num_steps=2)
        >>> exp_hamiltonian.resources(**exp_hamiltonian.resource_params)
        defaultdict(<class 'int'>, {PauliRot: 2, PauliRot: 2})

        Alternatively, we can call the resources method on from the class:

        >>> re.ResourceExp.resources(
        ...     base_class = qml.ops.Sum,
        ...     base_params = {},
        ...     base_pauli_rep = hamiltonian.pauli_rep,
        ...     coeff = 0.1*1j,
        ...     num_steps = 2,
        ... )
        defaultdict(<class 'int'>, {PauliRot: 2, PauliRot: 2})

        .. details::
            :title: Usage Details

            We can configure the resources for the exponential of a base operation by modifying
            its :code:`.exp_resource_decomp(scalar, num_steps, **resource_params)` method. Consider
            for example this custom PauliZ class, where the exponentiated resources are not defined
            (this is the default for a general :class:`~.ResourceOperator`).

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def exp_resource_decomp(cls, scalar, num_steps):
                        raise re.ResourcesNotDefined

            When this method is not defined, the resources are computed from the linear combination
            of Pauli words representation.

            >>> pauli_rep = CustomZ(wires=0).pauli_rep
            >>> pauli_rep
            1.0 * Z(0)
            >>> re.ResourceExp.resources(CustomZ, {}, base_pauli_rep=pauli_rep, coeff=0.1*1j, num_steps=3)
            defaultdict(<class 'int'>, {PauliRot: 3})

            We can update the exponential resources with the observation that the PauliZ gate, when
            exponentiated, produces an RZ rotation:

            .. code-block:: python

                class CustomZ(re.ResourceZ):

                    @classmethod
                    def exp_resource_decomp(cls, scalar, num_steps):
                        return {re.ResourceRZ.resource_rep(): num_steps}

            >>> re.ResourceExp.resources(CustomZ, {}, base_pauli_rep=pauli_rep, coeff=0.1*1j, num_steps=3)
            {RZ: 3}

        """
        # Custom exponential operator resources:
        if issubclass(base_class, re.ResourceOperator):
            try:
                return base_class.exp_resource_decomp(coeff, num_steps, **base_params)
            except re.ResourcesNotDefined:
                pass

        if base_pauli_rep and math.real(coeff) == 0:
            scalar = num_steps or 1  # 1st-order Trotter-Suzuki with 'num_steps' trotter steps:
            return _scale_dict(
                _resources_from_pauli_sentence(base_pauli_rep), scalar=scalar, in_place=True
            )

        raise re.ResourcesNotDefined

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[ResourceOperator]): The class type of the base operator that is exponentiated.
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * base_pauli_rep (Union[PauliSentence, None]): The base operator represented as a linear combination of Pauli words. If such a representation is not applicable, then :code:`None`.
                * coeff (complex): a scalar value which multiplies the base operator in the exponent
                * num_steps (int): the number of trotter steps to use in approximating the exponential
        """
        return _extract_exp_params(self.base, self.scalar, self.num_steps)

    @classmethod
    def resource_rep(cls, base_class, base_params, base_pauli_rep, coeff, num_steps):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[~.ResourceOperator]): The class type of the base operator that is
                exponentiated.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            base_pauli_rep (Union[PauliSentence, None]): The base operator represented as a linear
                combination of Pauli words. If such a representation is not applicable, then :code:`None`.
            coeff (complex): a scalar value which multiplies the base operator in the exponent
            num_steps (int): the number of trotter steps to use in approximating the exponential

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        name = cls.tracking_name(base_class, base_params, base_pauli_rep, coeff, num_steps)
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "base_pauli_rep": base_pauli_rep,
                "coeff": coeff,
                "num_steps": num_steps,
            },
            name=name,
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, base_params, base_pauli_rep, coeff, num_steps
    ) -> Dict[re.CompressedResourceOp, int]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            z0 (int): the power that the operator is being raised to
            base_class (Type[~.ResourceOperator]): The class type of the base operator that is
                exponentiated.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            base_pauli_rep (Union[PauliSentence, None]): The base operator represented as a linear
                combination of Pauli words. If such a representation is not applicable, then :code:`None`.
            coeff (complex): a scalar value which multiplies the base operator in the exponent
            num_steps (int): the number of trotter steps to use in approximating the exponential

        Resources:
            The resources are derived by simply multiplying together the :math:`z0` exponent and the
            :code:`coeff` coefficient into a single instance of :class:`~.ResourceExp` gate with
            coefficient :code:`z0 * coeff`.

        Returns:
            Dict[CompressedResourceOp, int]: The keys are the operators and the associated
                values are the counts.
        """
        return {cls.resource_rep(base_class, base_params, base_pauli_rep, z0 * coeff, num_steps): 1}

    @staticmethod
    def tracking_name(
        base_class: Operation,
        base_params: Dict,
        base_pauli_rep: PauliSentence,
        coeff: complex,
        num_steps: int,
    ):  # pylint: disable=unused-argument
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = (
            base_class.tracking_name(**base_params)
            if issubclass(base_class, re.ResourceOperator)
            else base_class.__name__
        )

        return f"Exp({base_name}, {coeff}, num_steps={num_steps})".replace("Resource", "")


class ResourceProd(Prod, re.ResourceOperator):
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
    def _resource_decomp(cmpr_factors, **kwargs) -> Dict[re.CompressedResourceOp, int]:
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
    def resource_rep(cls, cmpr_factors) -> re.CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_factors (list[CompressedResourceOp]): A list of operations, in the compressed
                representation, corresponding to the factors in the product.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return re.CompressedResourceOp(cls, {"cmpr_factors": cmpr_factors})


def _extract_exp_params(base_op, scalar, num_steps):
    pauli_rep = base_op.pauli_rep
    isinstance_resource_op = isinstance(base_op, re.ResourceOperator)

    if (not isinstance_resource_op) and (pauli_rep is None):
        raise ValueError(
            f"Cannot obtain resources for the exponential of {base_op}, if it is not a ResourceOperator and it doesn't have a Pauli decomposition."
        )

    base_class = type(base_op)
    base_params = base_op.resource_params if isinstance_resource_op else {}

    return {
        "base_class": base_class,
        "base_params": base_params,
        "base_pauli_rep": pauli_rep,
        "coeff": scalar,
        "num_steps": num_steps,
    }


def _resources_from_pauli_sentence(pauli_sentence):
    gate_types = defaultdict(int)

    for pauli_word in iter(pauli_sentence.keys()):
        pauli_string = "".join((str(v) for v in pauli_word.values()))
        pauli_rot_gate = re.ResourcePauliRot.resource_rep(pauli_string)
        gate_types[pauli_rot_gate] = 1

    return gate_types
