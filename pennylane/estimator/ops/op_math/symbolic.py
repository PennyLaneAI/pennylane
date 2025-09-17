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

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.wires import Wires

# pylint: disable=arguments-differ,super-init-not-called, signature-differs


class Adjoint(ResourceOperator):
    r"""Resource class for the symbolic Adjoint operation.

    A symbolic class used to represent the adjoint of some base operation.

    Args:
        base_op (:class:`~.pennylane.estimator.ResourceOperator`): The operator that we
            want the adjoint of.

    Resources:
        This symbolic operation represents the adjoint of some base operation. The resources are
        determined as follows. If the base operation implements the
        :code:`.adjoint_resource_decomp()` method, then the resources are obtained from
        this.

        Otherwise, the adjoint resources are given as the adjoint of each operation in the
        base operation's resources.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Adjoint`.

    **Example**

    The adjoint operation can be constructed like this:

        >>> qft = qml.estimator.QFT(num_wires=3)
        >>> adj_qft = qml.estimator.Adjoint(qft)

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
    >>> print(qml.estimator.estimate(qft, gate_set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 7
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
    {'Hadamard': 3, 'SWAP': 1, 'ControlledPhaseShift': 3}
    >>>
    >>> print(qml.estimator.estimate(adj_qft, gate_set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 7
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
    {'Adjoint(ControlledPhaseShift)': 3, 'Adjoint(SWAP)': 1, 'Adjoint(Hadamard)': 3}
    """

    resource_keys = {"base_cmpr_op"}

    def __init__(self, base_op: ResourceOperator) -> None:
        _dequeue(op_to_remove=base_op)
        self.queue()
        base_cmpr_op = base_op.resource_rep_from_op()

        self.base_op = base_cmpr_op
        self.wires = base_op.wires
        self.num_wires = base_cmpr_op.num_wires

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * base_cmpr_op (:class:`~.pennylane.estimator.ResourceOperator`): The operator
            that we want the adjoint of.

        """
        return {"base_cmpr_op": self.base_op}

    @classmethod
    def resource_rep(cls, base_cmpr_op: CompressedResourceOp) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.ResourceOperator`): The operator
                that we want the adjoint of.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, {"base_cmpr_op": base_cmpr_op})

    @classmethod
    def resource_decomp(cls, base_cmpr_op: CompressedResourceOp, **kwargs):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A
                compressed resource representation for the operator we want the adjoint of.
            wires (Sequence[int], optional): the wires the operation acts on

        Resources:
            This symbolic operation represents the adjoint of some base operation. The resources are
            determined as follows. If the base operation implements the
            :code:`.adjoint_resource_decomp()` method, then the resources are obtained from
            this.

            Otherwise, the adjoint resources are given as the adjoint of each operation in the
            base operation's resources.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: :class:`~.ops.op_math.adjoint.AdjointOperation`

        **Example**

        The adjoint operation can be constructed like this:

        >>> qft = qml.estimator.QFT(num_wires=3)
        >>> adj_qft = qml.estimator.Adjoint(qft)

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
        >>> print(qml.estimator.estimate(qft, gate_set))
        --- Resources: ---
        Total qubits: 3
        Total gates : 7
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'Hadamard': 3, 'SWAP': 1, 'ControlledPhaseShift': 3}
        >>>
        >>> print(qml.estimator.estimate(adj_qft, gate_set))
        --- Resources: ---
        Total qubits: 3
        Total gates : 7
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'Adjoint(ControlledPhaseShift)': 3, 'Adjoint(SWAP)': 1, 'Adjoint(Hadamard)': 3}

        """
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        base_params = {key: value for key, value in base_params.items() if value is not None}
        kwargs = {key: value for key, value in kwargs.items() if key not in base_params}

        try:
            return base_class.adjoint_resource_decomp(**base_params, **kwargs)
        except ResourcesUndefinedError:
            gate_lst = []
            decomp = base_class.resource_decomp(**base_params, **kwargs)

            for gate in decomp[::-1]:  # reverse the order
                gate_lst.append(_apply_adj(gate))
            return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, base_cmpr_op: CompressedResourceOp):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A
                compressed resource representation for the operator we want the adjoint of.

        Resources:
            The adjoint of an adjointed operation is just the original operation. The resources
            are given as one instance of the base operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(base_cmpr_op)]

    @staticmethod
    def tracking_name(base_cmpr_op: CompressedResourceOp) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Adjoint({base_name})"


class Controlled(ResourceOperator):
    r"""Resource class for the symbolic Controlled operation.

    A symbolic class used to represent the application of some base operation controlled on the
    state of some control qubits.

    Args:
        base_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The base operator to be
            controlled.
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the
            :math:`|0\rangle` state

    Resources:
        The resources are determined as follows. If the base operator implements the
        :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
        this.

        Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
        which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
        to an additional cost of two ``X`` gates per :code:`num_zero_ctrl`.
        Secondly, the base operation resources are extracted and we add to the cost the controlled
        variant of each operation in the resources.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Controlled`.

    **Example**

    The controlled operation can be constructed like this:

    >>> x = qml.estimator.X()
    >>> cx = qml.estimator.Controlled(x, num_ctrl_wires=1, num_zero_ctrl=0)
    >>> ccx = qml.estimator.Controlled(x, num_ctrl_wires=2, num_zero_ctrl=2)

    We can observe the expected gates when we estimate the resources.

    >>> print(qml.estimator.estimate(cx))
    --- Resources: ---
    Total qubits: 2
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
    {'CNOT': 1}
    >>>
    >>> print(qml.estimator.estimate(ccx))
    --- Resources: ---
    Total qubits: 3
    Total gates : 5
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
    {'X': 4, 'Toffoli': 1}

    """

    resource_keys = {"base_cmpr_op", "num_ctrl_wires", "num_zero_ctrl"}

    def __init__(
        self,
        base_op: ResourceOperator,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        wires=None,
    ) -> None:
        _dequeue(op_to_remove=base_op)
        self.queue()
        base_cmpr_op = base_op.resource_rep_from_op()

        self.base_op = base_cmpr_op
        self.num_ctrl_wires = num_ctrl_wires
        self.num_zero_ctrl = num_zero_ctrl

        self.num_wires = num_ctrl_wires + base_cmpr_op.num_wires
        if wires:
            self.wires = Wires(wires)
            if base_wires := base_op.wires:
                self.wires = Wires.all_wires([self.wires, base_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {wires}.")
        else:
            self.wires = None

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The base
            operator to be controlled.
            * num_ctrl_wires (int): the number of qubits the operation is controlled on
            * num_zero_ctrl (int): the number of control qubits, that are controlled when in the
            :math:`|0\rangle` state
        """

        return {
            "base_cmpr_op": self.base_op,
            "num_ctrl_wires": self.num_ctrl_wires,
            "num_zero_ctrl": self.num_zero_ctrl,
        }

    @classmethod
    def resource_rep(
        cls,
        base_cmpr_op,
        num_ctrl_wires,
        num_zero_ctrl,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The base
                operator to be controlled.
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the
                :math:`|0\rangle` state

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + base_cmpr_op.num_wires
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

    @classmethod
    def resource_decomp(
        cls, base_cmpr_op, num_ctrl_wires, num_zero_ctrl, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The base
                operator to be controlled.
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the
                :math:`|0\rangle` state

        Resources:
            The resources are determined as follows. If the base operator implements the
            :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
            this.

            Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
            which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
            to an additional cost of two ``X`` gates per :code:`num_zero_ctrl`.
            Secondly, the base operation resources are extracted and we add to the cost the controlled
            variant of each operation in the resources.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Controlled`.

        **Example**

        The controlled operation can be constructed like this:

        >>> x = qml.estimator.X()
        >>> cx = qml.estimator.Controlled(x, num_ctrl_wires=1, num_zero_ctrl=0)
        >>> ccx = qml.estimator.Controlled(x, num_ctrl_wires=2, num_zero_ctrl=2)

        We can observe the expected gates when we estimate the resources.

        >>> print(qml.estimator.estimate(cx))
        --- Resources: ---
        Total qubits: 2
        Total gates : 1
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
        Gate breakdown:
        {'CNOT': 1}
        >>>
        >>> print(qml.estimator.estimate(ccx))
        --- Resources: ---
        Total qubits: 3
        Total gates : 5
        Qubit breakdown:
        clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
        Gate breakdown:
        {'X': 4, 'Toffoli': 1}

        """

        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        base_params = {key: value for key, value in base_params.items() if value is not None}
        kwargs = {key: value for key, value in kwargs.items() if key not in base_params}
        try:
            return base_class.controlled_resource_decomp(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                target_resource_params=base_params,
            )
        except ResourcesUndefinedError:
            pass

        gate_lst = []
        if num_zero_ctrl != 0:
            x = resource_rep(qre.X)
            gate_lst.append(GateCount(x, 2 * num_zero_ctrl))

        decomp = base_class.resource_decomp(**base_params, **kwargs)
        for action in decomp:
            if isinstance(action, GateCount):
                gate = action.gate
                c_gate = cls.resource_rep(
                    gate,
                    num_ctrl_wires,
                    num_zero_ctrl=0,  # we flipped already and added the X gates above
                )
                gate_lst.append(GateCount(c_gate, action.count))

            else:  # pragma: no cover
                gate_lst.append(action)

        return gate_lst

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires,
        num_zero_ctrl,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            num_zero_ctrl (int): The subset of those control qubits, which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            target_resource_params (dict): The resource parameters of the base controlled operation.

        Resources:
            The resources are derived by simply combining the control qubits, control-values and
            work qubits into a single instance of ``Controlled`` gate, controlled
            on the whole set of control-qubits.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        inner_ctrl_wires = target_resource_params.get("num_ctrl_wires")
        inner_zero_ctrl = target_resource_params.get("num_zero_ctrl")
        base_cmpr_op = target_resource_params.get("base_cmpr_op")

        return [
            GateCount(
                cls.resource_rep(
                    base_cmpr_op,
                    inner_ctrl_wires + num_ctrl_wires,
                    inner_zero_ctrl + num_zero_ctrl,
                )
            ),
        ]

    @staticmethod
    def tracking_name(
        base_cmpr_op: CompressedResourceOp,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
    ):
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"C({base_name}, num_ctrl_wires={num_ctrl_wires},num_zero_ctrl={num_zero_ctrl})"


@singledispatch
def _apply_adj(action):
    raise TypeError(f"Unsupported type {action}")


@_apply_adj.register
def _(action: GateCount):
    gate = action.gate
    return GateCount(resource_rep(Adjoint, {"base_cmpr_op": gate}), action.count)


@_apply_adj.register
def _(action: Allocate):
    return Deallocate(action.num_wires)


@_apply_adj.register
def _(action: Deallocate):
    return Allocate(action.num_wires)
