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
from collections.abc import Iterable
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
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,super-init-not-called, signature-differs


class Adjoint(ResourceOperator):
    r"""Resource class for the symbolic Adjoint operation.

    Args:
        base_op (:class:`~.pennylane.estimator.ResourceOperator`): The operator for which
            to retrieve the adjoint.

    Resources:
        This symbolic operation represents the adjoint of some base operation. If the base operation implements the
        :code:`.adjoint_resource_decomp()` method, then the resources are obtained from
        this object. Otherwise, the adjoint resources are given as the adjoint of each operation in the
        base operation's resources.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Adjoint`.

    **Example**

    The adjoint operation can be constructed like this:

        >>> qft = qml.estimator.QFT(num_wires=3)
        >>> adj_qft = qml.estimator.Adjoint(qft)

    We can see how the resources differ by choosing a suitable gateset and estimating resources:

    >>> import pennylane.estimator as qre
    >>> gate_set = {
    ...     "SWAP",
    ...     "Adjoint(SWAP)",
    ...     "Hadamard",
    ...     "Adjoint(Hadamard)",
    ...     "ControlledPhaseShift",
    ...     "Adjoint(ControlledPhaseShift)",
    ... }
    >>>
    >>> print(qre.estimate(qft, gate_set))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 7
      'SWAP': 1,
      'ControlledPhaseShift': 3,
      'Hadamard': 3
    >>>
    >>> print(qre.estimate(adj_qft, gate_set))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 7
      'Adjoint(ControlledPhaseShift)': 3,
      'Adjoint(SWAP)': 1,
      'Adjoint(Hadamard)': 3

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

        Resources:
            This symbolic operation represents the adjoint of some base operation. The resources are
            determined as follows. If the base operation implements the
            :code:`.adjoint_resource_decomp()` method, then the resources are obtained from
            this method. Otherwise, the adjoint resources are given as the adjoint of each operation in the
            base operation's resources.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)

        base_params.update(
            (key, value)
            for key, value in kwargs.items()
            if key in base_params and base_params[key] is None
        )

        try:
            return base_class.adjoint_resource_decomp(base_params)
        except ResourcesUndefinedError:
            gate_lst = []
            decomp = base_class.resource_decomp(**base_params)

            for gate in decomp[::-1]:  # reverse the order
                gate_lst.append(_apply_adj(gate))
            return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the
                target operator.

        Resources:
            The adjoint of an adjointed operation is just the original operation. The resources
            are given as one instance of the base operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        base_cmpr_op = target_resource_params.get("base_cmpr_op")
        return [GateCount(base_cmpr_op)]

    @staticmethod
    # pylint: disable=arguments-renamed
    def tracking_name(base_cmpr_op: CompressedResourceOp) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Adjoint({base_name})"


class Controlled(ResourceOperator):
    r"""Resource class for the symbolic Controlled operation.

    Args:
        base_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The base operator to be
            controlled.
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the
            :math:`|0\rangle` state

    Resources:
        The resources are determined as follows. If the base operator implements the
        :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
        this object. Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
        which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
        to an additional cost of two ``X`` gates per :code:`num_zero_ctrl`.
        Secondly, the base operation resources are extracted and we add to the cost the controlled
        variant of each operation in the resources.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Controlled`.

    **Example**

    The controlled operation can be constructed like this:

    >>> import pennylane.estimator as qre
    >>> x = qre.X()
    >>> cx = qre.Controlled(x, num_ctrl_wires=1, num_zero_ctrl=0)
    >>> ccx = qre.Controlled(x, num_ctrl_wires=2, num_zero_ctrl=2)

    We can observe the expected gates when we estimate the resources.

    >>> print(qre.estimate(cx))
    --- Resources: ---
     Total wires: 2
        algorithmic wires: 2
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 1
      'CNOT': 1
    >>>
    >>> print(qre.estimate(ccx))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 5
      'Toffoli': 1,
      'X': 4

    """

    resource_keys = {"base_cmpr_op", "num_ctrl_wires", "num_zero_ctrl"}

    def __init__(
        self,
        base_op: ResourceOperator,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        wires: WiresLike = None,
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
        base_cmpr_op: CompressedResourceOp,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
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
        cls, base_cmpr_op: CompressedResourceOp, num_ctrl_wires: int, num_zero_ctrl: int, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): The base
                operator to be controlled.
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits that are controlled when in the
                :math:`|0\rangle` state

        Resources:
            The resources are determined as follows. If the base operator implements the
            :code:`.controlled_resource_decomp()` method, then the resources are obtained directly from
            this method. Otherwise, the controlled resources are given in two steps. Firstly, any control qubits
            which should be triggered when in the :math:`|0\rangle` state, are flipped. This corresponds
            to an additional cost of two ``X`` gates per :code:`num_zero_ctrl`.
            Secondly, the base operation resources are extracted and we add to the cost the controlled
            variant of each operation in the resources.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """

        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        base_params.update(
            (key, value)
            for key, value in kwargs.items()
            if key in base_params and base_params[key] is None
        )

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

        decomp = base_class.resource_decomp(**base_params)
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
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            num_zero_ctrl (int): The subset of those control qubits which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            target_resource_params (dict): A dictionary containing the resource parameters of the
                target operator.

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


class Pow(ResourceOperator):
    r"""Resource class for the symbolic Pow operation.

    This symbolic class can be used to represent some base operation raised to a power.

    Args:
        base_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The operator to exponentiate.
        pow_z (int): the exponent (default value is 1)

    Resources:
        The resources are determined as follows. If the power :math:`z = 0`, this corresponds to the identity
        gate which requires no resources. If the base operation class :code:`base_class` implements the
        :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
        the resources of the operation raised to the power :math:`z` are given by extracting the base
        operation's resources (via :class:`~.pennylane.estimator.resources_base.Resources`) and raising each operation to the same power.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Pow`.

    **Example**

    The operation raised to a power :math:`z` can be constructed like this:

    >>> import pennylane.estimator as qre
    >>> z = qre.Z()
    >>> z_2 = qre.Pow(z, 2)
    >>> z_5 = qre.Pow(z, 5)

    We obtain the expected resources.

    >>> print(qre.estimate(z_2, gate_set={"Identity", "Z"}))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 1
      'Identity': 1
    >>>
    >>> print(qre.estimate(z_5, gate_set={"Identity", "Z"}))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 1
      'Z': 1

    """

    resource_keys = {"base_cmpr_op", "z"}

    def __init__(self, base_op: ResourceOperator, pow_z: int) -> None:
        _dequeue(op_to_remove=base_op)
        self.queue()
        base_cmpr_op = base_op.resource_rep_from_op()

        self.pow_z = pow_z
        self.base_op = base_cmpr_op
        self.wires = base_op.wires
        self.num_wires = base_cmpr_op.num_wires

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * base_class (Type[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): The class type of the base operator to be raised to some power.
                * base_params (dict): the resource parameters required to extract the cost of the base operator
                * z (int): the power that the operator is being raised to
        """
        return {
            "base_cmpr_op": self.base_op,
            "pow_z": self.pow_z,
        }

    @classmethod
    def resource_rep(cls, base_cmpr_op: CompressedResourceOp, pow_z: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute a resource estimation.

        Args:
            base_class (Type[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): The class type of the base operator to be raised to some power.
            base_params (dict): the resource parameters required to extract the cost of the base operator
            pow_z (int): the power that the operator is being raised to

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = base_cmpr_op.num_wires
        return CompressedResourceOp(cls, num_wires, {"base_cmpr_op": base_cmpr_op, "pow_z": pow_z})

    @classmethod
    def resource_decomp(
        cls, base_cmpr_op: CompressedResourceOp, pow_z: int, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            base_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A
                compressed resource representation for the operator we want to exponentiate.
            pow_z (float): the exponent (default value is 1)

        Resources:
            The resources are determined as follows. If the power :math:`z = 0`, this corresponds to the identity
            gate which requires no resources. If the base operation class :code:`base_class` implements the
            :code:`.pow_resource_decomp()` method, then the resources are obtained from this. Otherwise,
            the resources of the operation raised to the power :math:`z` are given by extracting the base
            operation's resources (via :class:`~.pennylane.estimator.resources_base.Resources`) and
            raising each operation to the same power.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        base_class, base_params = (base_cmpr_op.op_type, base_cmpr_op.params)
        base_params.update(
            (key, value)
            for key, value in kwargs.items()
            if key in base_params and base_params[key] is None
        )

        if pow_z == 0:
            return [GateCount(resource_rep(qre.Identity))]

        if pow_z == 1:
            return [GateCount(base_cmpr_op)]

        try:
            return base_class.pow_resource_decomp(pow_z=pow_z, target_resource_params=base_params)
        except ResourcesUndefinedError:
            return [GateCount(base_cmpr_op, pow_z)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            pow_z (int): The exponent that the base operator is being raised to. Default value is 1.
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are derived by simply adding together the :math:`z` exponent and the
            :math:`z_{0}` exponent into a single instance of :class:`~.Pow` gate, raising
            the base operator to the power :math:`z + z_{0}`.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        z = target_resource_params.get("pow_z", 1)
        base_cmpr_op = target_resource_params.get("base_cmpr_op")
        return [GateCount(cls.resource_rep(base_cmpr_op, pow_z * z))]

    @staticmethod
    def tracking_name(base_cmpr_op: CompressedResourceOp, pow_z: int) -> str:
        r"""Returns the tracking name built with the operator's parameters."""
        base_name = base_cmpr_op.name
        return f"Pow({base_name}, {pow_z})"


class Prod(ResourceOperator):
    r"""Resource class for the symbolic Prod operation.

    This symbolic class can be used to represent a product of some base operations.

    Args:
        res_ops (tuple[:class:`~.pennylane.estimator.resource_operator.ResourceOperator`]): A tuple of
            resource operators or a nested tuple of resource operators and counts.
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        This symbolic class represents a product of operations. The resources are defined trivially
        as the counts for each operation in the product.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.Prod`.

    **Example**

    The product of operations can be constructed from a list of operations or
    a nested tuple where each operator is accompanied by its count.
    Each operation in the product must be a valid :class:`~.estimator.ResourceOperator`.

    We can construct a product operator as follows:

    >>> import pennylane.estimator as qre
    >>> factors = [qre.X(), qre.Y(), qre.Z()]
    >>> prod_xyz = qre.Prod(factors)
    >>>
    >>> print(qre.estimate(prod_xyz))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 3
      'X': 1,
      'Y': 1,
      'Z': 1

    We can also specify the factors as a tuple with

    >>> factors = [(qre.X(), 2), (qre.Z(), 3)]
    >>> prod_x2z3 = qre.Prod(factors)
    >>>
    >>> print(qre.estimate(prod_x2z3))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
                 zero state: 0
                 any state: 0
     Total gates : 5
      'X': 2,
      'Z': 3

    """

    resource_keys = {"num_wires", "cmpr_factors_and_counts"}

    def __init__(
        self,
        res_ops: Iterable[ResourceOperator | tuple[ResourceOperator, int]],
        wires: WiresLike = None,
    ) -> None:

        ops = []
        counts = []

        ops, counts = zip(
            *(item if isinstance(item, (list, tuple)) else (item, 1) for item in res_ops)
        )

        _dequeue(op_to_remove=ops)
        self.queue()

        try:
            cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)
        except AttributeError as error:
            raise ValueError(
                "All factors of the Product must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        self.cmpr_factors_and_counts = tuple(zip(cmpr_ops, counts))

        if wires:  # User defined wires take precedent
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)

        else:  # Otherwise determine the wires from the factors in the product
            ops_wires = Wires.all_wires([op.wires for op in ops if op.wires is not None])
            num_unique_wires_required = max(op.num_wires for op in cmpr_ops)

            if (
                len(ops_wires) < num_unique_wires_required
            ):  # If factors didn't provide enough wire labels
                self.wires = None  # we assume they all act on the same set
                self.num_wires = num_unique_wires_required

            else:  # If there are more wire labels, use that as the operator wires
                self.wires = ops_wires
                self.num_wires = len(self.wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires the operator acts upon
                * cmpr_factors_and_counts (Tuple[Tuple[:class:`~.estimator.CompressedResourceOp`, int]]):
                  A sequence of tuples containing the operations, in the compressed representation, and
                  a count for how many times they are repeated corresponding to the factors in the product.

        """
        return {
            "num_wires": self.num_wires,
            "cmpr_factors_and_counts": self.cmpr_factors_and_counts,
        }

    @classmethod
    def resource_rep(
        cls, cmpr_factors_and_counts, num_wires: WiresLike = None
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute a resource estimation.

        Args:
            cmpr_factors_and_counts (Tuple[Tuple[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`, int]]):
                A sequence of tuples containing the operations, in the compressed representation, and
                a count for how many times they are repeated corresponding to the factors in the product.
            num_wires (int): an optional integer representing the number of wires this operator acts upon

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = num_wires or max(cmpr_op.num_wires for cmpr_op, _ in cmpr_factors_and_counts)
        return CompressedResourceOp(
            cls,
            num_wires,
            {"num_wires": num_wires, "cmpr_factors_and_counts": cmpr_factors_and_counts},
        )

    @classmethod
    def resource_decomp(
        cls, cmpr_factors_and_counts, num_wires: int
    ):  # pylint: disable=unused-argument
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            cmpr_factors_and_counts (Tuple[Tuple[:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`, int]]):
                A sequence of tuples containing the operations, in the compressed representation, and
                a count for how many times they are repeated corresponding to the factors in the product.
            num_wires (int): the number of wires this operator acts upon

        Resources:
            This symbolic class represents a product of operations. The resources are defined
            trivially as the counts for each operation in the product.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        return [GateCount(cmpr_op, count) for cmpr_op, count in cmpr_factors_and_counts]


class ChangeOpBasis(ResourceOperator):
    r"""Change of Basis resource operator.

    This symbolic class can be used to represent a change of basis operation with a compute-uncompute pattern.
    This is a special type of operator which can be expressed as
    :math:`\hat{U}_{compute} \cdot \hat{V} \cdot \hat{U}_{uncompute}`. If no :code:`uncompute_op` is
    provided then the adjoint of the :code:`compute_op` is used by default.

    Args:
        compute_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): A resource operator
            representing the basis change operation.
        target_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): A resource operator
            representing the base operation.
        uncompute_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): An optional
            resource operator representing the inverse of the basis change operation. If no
            :code:`uncompute_op` is provided then the adjoint of the :code:`compute_op` is used by default.
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        This symbolic class represents a product of the three provided operations. The resources are
        defined trivially as the sum of the costs of each.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.ChangeOpBasis`.

    **Example**

    The change of basis operation can be constructed as follows with each operation defining the
    compute-uncompute pattern being a valid :class:`~.pennylane.estimator.resource_operator.ResourceOperator`:

    >>> import pennylane.estimator as qre
    >>> compute_u = qre.Hadamard()
    >>> base_v = qre.Z()
    >>> cb_op = qre.ChangeOpBasis(compute_u, base_v)
    >>> print(qre.estimate(cb_op, gate_set={"Z", "Hadamard", "Adjoint(Hadamard)"}))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 3
      'Adjoint(Hadamard)': 1,
      'Z': 1,
      'Hadamard': 1

    We can also set the :code:`uncompute_op` directly.

    >>> uncompute_u = qre.Hadamard()
    >>> cb_op = qre.ChangeOpBasis(compute_u, base_v, uncompute_u)
    >>> print(qre.estimate(cb_op, gate_set={"Z", "Hadamard", "Adjoint(Hadamard)"}))
    --- Resources: ---
     Total wires: 1
        algorithmic wires: 1
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 3
      'Z': 1,
      'Hadamard': 2

    """

    resource_keys = {"num_wires", "cmpr_compute_op", "cmpr_target_op", "cmpr_uncompute_op"}

    def __init__(
        self,
        compute_op: ResourceOperator,
        target_op: ResourceOperator,
        uncompute_op: None | ResourceOperator = None,
        wires: WiresLike = None,
    ) -> None:
        ops_to_remove = (
            [compute_op, target_op, uncompute_op] if uncompute_op else [compute_op, target_op]
        )
        _dequeue(op_to_remove=ops_to_remove)
        self.queue()

        try:
            self.cmpr_compute_op = compute_op.resource_rep_from_op()
            self.cmpr_target_op = target_op.resource_rep_from_op()
        except AttributeError as error:
            raise ValueError(
                "All ops of the ChangeOpBasis must be instances of `ResourceOperator` in order to obtain resources."
            ) from error

        self.cmpr_uncompute_op = (
            uncompute_op.resource_rep_from_op()
            if uncompute_op
            else Adjoint.resource_rep(base_cmpr_op=self.cmpr_compute_op)
        )

        if wires:  # User defined wires take precedent
            self.wires = Wires(wires)
            self.num_wires = len(self.wires)

        else:  # Otherwise determine the wires from the compute, base & uncompute ops
            ops_wires = Wires.all_wires([op.wires for op in ops_to_remove if op.wires is not None])
            num_unique_wires_required = max(
                op.num_wires
                for op in [self.cmpr_target_op, self.cmpr_compute_op, self.cmpr_uncompute_op]
            )

            if (
                len(ops_wires) < num_unique_wires_required
            ):  # If factors didn't provide enough wire labels
                self.wires = None
                self.num_wires = num_unique_wires_required

            else:  # If there are more wire labels, use that as the operator wires
                self.wires = ops_wires
                self.num_wires = len(self.wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_compute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the compute operation.
                * cmpr_target_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the base operation.
                * cmpr_uncompute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                  to the uncompute operation.
                * num_wires (int): the number of wires this operator acts upon

        """
        return {
            "cmpr_compute_op": self.cmpr_compute_op,
            "cmpr_target_op": self.cmpr_target_op,
            "cmpr_uncompute_op": self.cmpr_uncompute_op,
            "num_wires": self.num_wires,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_compute_op: CompressedResourceOp,
        cmpr_target_op: CompressedResourceOp,
        cmpr_uncompute_op: CompressedResourceOp | None = None,
        num_wires: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to estimate the resources.

        Args:
            cmpr_compute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the compute operation.
            cmpr_target_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the base operation.
            cmpr_uncompute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): An optional compressed resource operator, corresponding
                to the uncompute operation. The adjoint of the :code:`cmpr_compute_op` is used by default.
            num_wires (int): an optional integer representing the number of wires this operator acts upon

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        cmpr_uncompute_op = cmpr_uncompute_op or resource_rep(
            Adjoint, {"base_cmpr_op": cmpr_compute_op}
        )
        num_wires = num_wires or max(
            cmpr_compute_op.num_wires, cmpr_target_op.num_wires, cmpr_uncompute_op.num_wires
        )
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "cmpr_compute_op": cmpr_compute_op,
                "cmpr_target_op": cmpr_target_op,
                "cmpr_uncompute_op": cmpr_uncompute_op,
                "num_wires": num_wires,
            },
        )

    @classmethod
    def resource_decomp(
        cls,
        cmpr_compute_op: CompressedResourceOp,
        cmpr_target_op: CompressedResourceOp,
        cmpr_uncompute_op: CompressedResourceOp,
        num_wires: int,  # pylint: disable=unused-argument
    ):
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            cmpr_compute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the compute operation.
            cmpr_target_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): A compressed resource operator, corresponding
                to the base operation.
            cmpr_uncompute_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`): An optional compressed resource operator, corresponding
                to the uncompute operation. The adjoint of the :code:`cmpr_compute_op` is used by default.

        Resources:
            This symbolic class represents a product of the three provided operations. The resources are
            defined trivially as the sum of the costs of each.

        .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ops.op_math.ChangeOpBasis`.

        **Example**

        The change of basis operation can be constructed as follows with each operation defining the
        compute-uncompute pattern being a valid :class:`~.pennylane.estimator.resource_operator.ResourceOperator`:

        >>> import pennylane.estimator as qre
        >>> compute_u = qre.Hadamard()
        >>> base_v = qre.Z()
        >>> cb_op = qre.ChangeOpBasis(compute_u, base_v)
        >>> print(qre.estimate(cb_op, gate_set={"Z", "Hadamard", "Adjoint(Hadamard)"}))
        --- Resources: ---
         Total wires: 1
            algorithmic wires: 1
            allocated wires: 0
                     zero state: 0
                     any state: 0
         Total gates : 3
          'Adjoint(Hadamard)': 1,
          'Z': 1,
          'Hadamard': 1

        We can also set the :code:`uncompute_op` directly.

        >>> uncompute_u = qre.Hadamard()
        >>> cb_op = qre.ChangeOpBasis(compute_u, base_v, uncompute_u)
        >>> print(qre.estimate(cb_op, gate_set={"Z", "Hadamard", "Adjoint(Hadamard)"}))
        --- Resources: ---
         Total wires: 1
            algorithmic wires: 1
            allocated wires: 0
                 zero state: 0
                 any state: 0
         Total gates : 3
          'Z': 1,
          'Hadamard': 2
        """
        return [
            GateCount(cmpr_compute_op),
            GateCount(cmpr_target_op),
            GateCount(cmpr_uncompute_op),
        ]


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
