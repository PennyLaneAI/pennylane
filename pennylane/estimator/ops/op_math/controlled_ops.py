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
r"""Resource operators for controlled operations."""

from pennylane.estimator.ops import RY, RZ, Hadamard, Identity, S, T, X, Z
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    ResourcesNotDefined,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate

# pylint: disable=arguments-differ,too-many-ancestors,too-many-arguments,too-many-positional-arguments,unused-argument


class CH(ResourceOperator):
    r"""Resource class for the CH gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are derived from the following identities (as presented in this
        `blog post <https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control>`_):

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are defined as two :class:`~.RY`, two
        :class:`~.Hadamard` and one :class:`~.CNOT` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CH`.

    **Example**

    The resources for this operation are computed using:

    >>> re.CH.resource_decomp()
    [(2 x Hadamard), (2 x RY), (1 x CNOT)]

    """

    num_wires = 2

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are derived from the following identities (as presented in this
            `blog post <https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control>`_):

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are defined as two :class:`~.ResourceRY`, two
            :class:`~.ResourceHadamard` and one :class:`~.CNOT` gates.
        """
        ry = resource_rep(RY)
        h = resource_rep(Hadamard)
        cnot = resource_rep(CNOT)
        return [GateCount(h, 2), GateCount(ry, 2), GateCount(cnot, 1)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceHadamard` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CY(ResourceOperator):
    r"""Resource class for the CY gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

        By replacing the :class:`~.X` gate with a :class:`~.CNOT` we
        obtain the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.CNOT` gate conjugated by a pair of :class:`~.S` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CY.resource_decomp()
    [(1 x CNOT), (1 x S), (1 x Adjoint(S))]
    """

    num_wires = 2

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

            By replacing the :class:`~.X` gate with a :class:`~.CNOT` we
            obtain the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.CNOT` gate conjugated by a pair of :class:`~.S` gates.
        """
        cnot = resource_rep(CNOT)
        s = resource_rep(S)
        z = resource_rep(Z)

        return [GateCount(cnot), GateCount(s, 2), GateCount(z)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.Y` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CZ(ResourceOperator):
    r"""Resource class for the CZ gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the :class:`~.X` gate with a :class:`~.CNOT` we obtain
        the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.CNOT` gate conjugated by a pair of :class:`~.Hadamard` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CZ.resource_decomp()
    [(1 x CNOT), (2 x Hadamard)]
    """

    num_wires = 2

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the :class:`~.ResourceX` gate with a :class:`~.CNOT` we obtain
            the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.CNOT` gate conjugated by a pair of :class:`~.ResourceHadamard` gates.
        """
        cnot = resource_rep(CNOT)
        h = resource_rep(Hadamard)

        return [GateCount(cnot), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceZ` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1 and ctrl_num_ctrl_values == 0:
            return [GateCount(resource_rep(CCZ))]

        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CSWAP(ResourceOperator):
    r"""Resource class for the CSWAP gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/pdf/2305.18128>`_.

        The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
        defined as:

        .. code-block:: bash

            0: ────╭●────┤
            1: ─╭X─├●─╭X─┤
            2: ─╰●─╰X─╰●─┤

    .. seealso:: The corresponding PennyLane operation :class:`~.CSWAP`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CSWAP.resource_decomp()
    [(1 x Toffoli), (2 x CNOT)]
    """

    num_wires = 3

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/pdf/2305.18128>`_.

            The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
            defined as:

            .. code-block:: bash

                0: ────╭●────┤
                1: ─╭X─├●─╭X─┤
                2: ─╰●─╰X─╰●─┤
        """
        tof = resource_rep(Toffoli)
        cnot = resource_rep(CNOT)
        return [GateCount(tof), GateCount(cnot, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.SWAP` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CCZ(ResourceOperator):
    r"""Resource class for the CCZ gate.

    Args:
        wires (Sequence[int], optional): the subsystem the gate acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the :class:`~.X` gate with a :class:`~.Toffoli` we obtain
        the controlled decomposition. Specifically, the resources are defined as a
        :class:`~.Toffoli` gate conjugated by a pair of :class:`~.Hadamard` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CCZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CCZ.resource_decomp()
    [(1 x Toffoli), (2 x Hadamard)]
    """

    num_wires = 3

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the :class:`~.X` gate with a :class:`~.Toffoli` we obtain
            the controlled decomposition. Specifically, the resources are defined as a
            :class:`~.Toffoli` gate conjugated by a pair of :class:`~.Hadamard` gates.
        """
        toffoli = resource_rep(Toffoli)
        h = resource_rep(Hadamard)
        return [GateCount(toffoli), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.Z` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CNOT(ResourceOperator):
    r"""Resource class for the CNOT gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: The corresponding PennyLane operation :class:`~.CNOT`.

    """

    num_wires = 2

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

        Raises:
            ResourcesNotDefined: This gate is fundamental, no further decomposition defined.
        """
        raise ResourcesNotDefined

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed as one general :class:`~.MultiControlledX` gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1 and ctrl_num_ctrl_values == 0:
            return [GateCount(resource_rep(Toffoli))]

        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class TempAND(ResourceOperator):
    r"""Resource class representing a temporary `AND`-gate.

    Args:
        wires (Sequence[int], optional): the wires the operation acts on

    This gate was introduced in Fig 4 of `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_ along
    with it's adjoint (uncompute).

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.TempAND.resource_decomp()
    [(1 x Toffoli)]
    """

    num_wires = 3

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            The resources are obtained from Figure 4 of `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        tof = resource_rep(Toffoli, {"elbow": "left"})
        return [GateCount(tof)]

    @classmethod
    def adjoint_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The resources are obtained from Figure 4 of `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = resource_rep(Hadamard)
        cz = resource_rep(CZ)
        return [GateCount(h), GateCount(cz)]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are expressed as one general :class:`~.MultiControlledX` gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined


class Toffoli(ResourceOperator):
    r"""Resource class for the Toffoli gate.

    Args:
        wires (Sequence[int], optional): the subsystem the gate acts on
        elbow (Union[str, None]): String identifier to determine if this is a special type of
            Toffoli gate (left or right elbow). Default value is `None`.

    Resources:
        If `elbow` is provided, resources are obtained from Figure 4 of
        `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_.

        If `elbow` is `None`, the resources are obtained from Figure 1 of
        `Jones 2012 <https://arxiv.org/pdf/1212.5069>`_.

        The circuit which applies the Toffoli gate on target wire 'target' with control wires
        ('c1', 'c2') is defined as:

        .. code-block:: bash

                c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
              aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
              aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
            target: ─────────────────────────────────╰X──────║───║─┤
                                                             ╚═══╝

        Specifically, the resources are defined as nine :class:`~.CNOT` gates, three
        :class:`~.Hadamard` gates, one :class:`~.CZ` gate, one :class:`~.S`
        gate, two :class:`~.T` gates and two adjoint :class:`~.T` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.Toffoli`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.Toffoli.resource_decomp()
    [Allocate(2), (9 x CNOT), (3 x Hadamard), (1 x S), (1 x CZ), (2 x T), (2 x Adjoint(T)), Deallocate(2)]
    """

    num_wires = 3
    resource_keys = {"elbow"}

    def __init__(self, elbow=None, wires=None) -> None:
        self.elbow = elbow
        super().__init__(wires=wires)

    @staticmethod
    def elbow_decomp(elbow="left"):
        """A function that prepares the resource decomposition obtained from Figure 4 of
        `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_.

        Args:
            elbow (str, optional): One of "left" or "right". Defaults to "left".

        Returns:
            list[GateCount]: The resources of decomposing the elbow gates.
        """
        gate_types = []
        t = resource_rep(T)
        h = resource_rep(Hadamard)
        cnot = resource_rep(CNOT)
        cz = resource_rep(CZ)
        s = resource_rep(S)
        z = resource_rep(Z)

        if elbow == "left":
            gate_types.append(GateCount(t, 4))
            gate_types.append(GateCount(cnot, 3))
            gate_types.append(GateCount(s, 3))
            gate_types.append(GateCount(z, 3))

        if elbow == "right":
            gate_types.append(GateCount(h))
            gate_types.append(GateCount(cz))

        return gate_types

    @classmethod
    def resource_decomp(cls, elbow=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            If `elbow` is provided, resources are obtained from Figure 4 of
            `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.

            If `elbow` is `None`, the resources are obtained from Figure 1 of
            `Jones 2012 <https://arxiv.org/pdf/1212.5069>`_.

            The circuit which applies the Toffoli gate on target wire 'target' with control wires
            ('c1', 'c2') is defined as:

            .. code-block:: bash

                  c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                  c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
                aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
                aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
              target: ─────────────────────────────────╰X──────║───║─┤
                                                               ╚═══╝

            Specifically, the resources are defined as nine :class:`~.CNOT` gates, three
            :class:`~.Hadamard` gates, one :class:`~.CZ` gate, one :class:`~.S`
            gate, two :class:`~.T` gates and two adjoint :class:`~.T` gates.
        """
        if elbow:
            return Toffoli.elbow_decomp(elbow)

        cnot = resource_rep(CNOT)
        t = resource_rep(T)
        h = resource_rep(Hadamard)
        s = resource_rep(S)
        cz = resource_rep(CZ)
        z = resource_rep(Z)

        return [
            Allocate(2),
            GateCount(cnot, 9),
            GateCount(h, 3),
            GateCount(s, 3),
            GateCount(cz),
            GateCount(t, 4),
            GateCount(z, 2),
            Deallocate(2),
        ]

    @classmethod
    def textbook_resource_decomp(cls, elbow=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Resources:
            If `elbow` is provided, resources are obtained from Figure 4 of
            `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.

            If `elbow` is `None`, the resources are taken from Figure 4.9 of `Nielsen, M. A., & Chuang, I. L. (2010)
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

            The circuit is defined as:

            .. code-block:: bash

                0: ───────────╭●───────────╭●────╭●──T──╭●─┤
                1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
                2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

            Specifically, the resources are defined as six :class:`~.CNOT` gates, two
            :class:`~.Hadamard` gates, four :class:`~.T` gates and three adjoint
            :class:`~.T` gates.
        """
        if elbow:
            return Toffoli.elbow_decomp(elbow)

        cnot = resource_rep(CNOT)
        t = resource_rep(T)
        h = resource_rep(Hadamard)
        z = resource_rep(Z)
        s = resource_rep(S)

        return [
            GateCount(cnot, 6),
            GateCount(h, 2),
            GateCount(t, 7),
            GateCount(z, 3),
            GateCount(s, 3),
        ]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * elbow (Union[str, None]): String identifier to determine if this is a special type of Toffoli gate (left or right elbow).

        """
        return {"elbow": self.elbow}

    @classmethod
    def resource_rep(cls, elbow=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"elbow": elbow})

    @classmethod
    def adjoint_resource_decomp(cls, elbow=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if elbow is None:
            return [GateCount(cls.resource_rep())]

        adj_elbow = "left" if elbow == "right" else "right"
        return [GateCount(cls.resource_rep(elbow=adj_elbow))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        elbow=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            elbow (Union[str, None]): String identifier to determine if this is a special type of Toffoli gate (left or right elbow).
                Default value is `None`.
        Resources:
            The resources are expressed as one general :class:`~.MultiControlledX` gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, elbow=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep(elbow=elbow))]
        )


class MultiControlledX(ResourceOperator):
    r"""Resource class for the MultiControlledX gate.

    Args:
        num_ctrl_wires (int): the number of qubits the operation is controlled on
        num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        wires (Sequence[int], optional): the wires this operation acts on

    Resources:
        The resources are obtained based on the unary iteration technique described in
        `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_. Specifically, the
        resources are defined as the following rules:

        * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

        * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

        * If there are three or more control qubits (:math:`n`), the resources obtained based on the unary iteration technique described in `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_. Specifically, it requires :math:`n - 2` clean qubits, and produces :math:`n - 2` elbow gates and a single :class:`~.pennylane.estimator.ops.Toffoli`.

    .. seealso:: :class:`~.MultiControlledX`

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.MultiControlledX.resource_decomp(num_ctrl_wires=5, num_ctrl_values=2)
    [(4 x X), AllocWires(3), (3 x TempAND), (3 x Toffoli), (1 x Toffoli), FreeWires(3)]
    """

    resource_keys = {"num_ctrl_wires", "num_ctrl_values"}

    def __init__(self, num_ctrl_wires, num_ctrl_values, wires=None) -> None:
        self.num_ctrl_wires = num_ctrl_wires
        self.num_ctrl_values = num_ctrl_values

        self.num_wires = num_ctrl_wires + 1
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_ctrl_wires (int): the number of qubits the operation is controlled on
                * num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        """

        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "num_ctrl_values": self.num_ctrl_values,
        }

    @classmethod
    def resource_rep(cls, num_ctrl_wires, num_ctrl_values) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
            },
        )

    @classmethod
    def resource_decomp(
        cls,
        num_ctrl_wires,
        num_ctrl_values,
        **kwargs,  # pylint: disable=unused-argument
    ) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are obtained based on the unary iteration technique described in
            `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_. Specifically, the
            resources are defined as the following rules:

            * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

            * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

            * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

            * If there are three or more control qubits (:math:`n`), the resources obtained based on the unary iteration technique described in `Babbush 2018 <https://arxiv.org/pdf/1805.03662>`_. Specifically, it requires :math:`n - 2` clean qubits, and produces :math:`n - 2` elbow gates and a single :class:`~.pennylane.estimator.ops.Toffoli`.

        """
        gate_lst = []

        x = resource_rep(X)
        if num_ctrl_wires == 0:
            if num_ctrl_values:
                return []

            return [GateCount(x)]

        if num_ctrl_values:
            gate_lst.append(GateCount(x, num_ctrl_values * 2))

        cnot = resource_rep(CNOT)
        if num_ctrl_wires == 1:
            gate_lst.append(GateCount(cnot))
            return gate_lst

        toffoli = resource_rep(Toffoli)
        if num_ctrl_wires == 2:
            gate_lst.append(GateCount(toffoli))
            return gate_lst

        raise ResourcesNotDefined

    @classmethod
    def adjoint_resource_decomp(cls, num_ctrl_wires, num_ctrl_values, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(num_ctrl_wires, num_ctrl_values))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        num_ctrl_wires,
        num_ctrl_values,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): The number of control qubits to further control the base
                controlled operation upon.
            ctrl_num_ctrl_values (int): The subset of those control qubits, which further control
                the base controlled operation, which are controlled when in the :math:`|0\rangle` state.
            num_ctrl_wires (int): the number of control qubits of the operation
            num_ctrl_values (int): The subset of control qubits of the operation, that are controlled
                when in the :math:`|0\rangle` state.

        Resources:
            The resources are derived by combining the control qubits, control-values and
            into a single instance of :class:`~.MultiControlledX` gate, controlled
            on the whole set of control-qubits.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [
            GateCount(
                cls.resource_rep(
                    ctrl_num_ctrl_wires + num_ctrl_wires,
                    ctrl_num_ctrl_values + num_ctrl_values,
                )
            )
        ]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z, num_ctrl_wires, num_ctrl_values, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep(num_ctrl_wires, num_ctrl_values))]
        )


class CRX(ResourceOperator):
    r"""Resource class for the CRX gate.

    Args:
        wires (Sequence[int], optional): the wire the operation acts on
        eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

        we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
        gates. The expression for controlled-RZ gates is used as defined in the reference above.
        Specifically, the resources are defined as two :class:`~.CNOT` gates, two
        :class:`~.Hadamard` gates and two :class:`~.RZ` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CRX`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRX.resource_decomp()
    [(2 x CNOT), (2 x RZ), (2 x Hadamard)]
    """

    resource_keys = {"eps"}
    num_wires = 2

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * eps (Union[float, None]): error threshold for the approximation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""

        return CompressedResourceOp(cls, cls.num_wires, {"eps": eps})

    @classmethod
    def resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Args:
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
            gates. The expression for controlled-RZ gates is used as defined in the reference above.
            Specifically, the resources are defined as two :class:`~.CNOT` gates, two
            :class:`~.Hadamard` gates and two :class:`~.RZ` gates.
        """
        h = resource_rep(Hadamard)
        rz = resource_rep(RZ, {"eps": eps})
        cnot = resource_rep(CNOT)

        return [GateCount(cnot, 2), GateCount(rz, 2), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.RX` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]


class CRY(ResourceOperator):
    r"""Resource class for the CRY gate.

    Args:
        wires (Sequence[int], optional): the wire the operation acts on
        eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

        By replacing the :class:`~.X` gates with :class:`~.CNOT` gates, we obtain a
        controlled-version of this identity. Thus we are able to constructively or destructively
        interfere the gates based on the value of the control qubit. Specifically, the resources are
        defined as two :class:`~.CNOT` gates and two :class:`~.RY` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CRY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRY.resource_decomp()
    [(2 x CNOT), (2 x RY)]
    """

    resource_keys = {"eps"}
    num_wires = 2

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * eps (Union[float, None]): error threshold for the approximation

        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"eps": eps})

    @classmethod
    def resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Args:
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two :class:`~.CNOT` gates
            and two :class:`~.RY` gates.
        """
        cnot = resource_rep(CNOT)
        ry = resource_rep(RY, {"eps": eps})
        return [GateCount(cnot, 2), GateCount(ry, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.RY` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]


class CRZ(ResourceOperator):
    r"""Resource class for the CRZ gate.

    Args:
        wires (Sequence[int], optional): the wire the operation acts on
        eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

        By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
        identity. Thus we are able to constructively or destructively interfere the gates based on the value
        of the control qubit. Specifically, the resources are defined as two :class:`~.CNOT` gates
        and two :class:`~.RZ` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CRZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRZ.resource_decomp()
    [(2 x CNOT), (2 x RZ)]
    """

    resource_keys = {"eps"}
    num_wires = 2

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * eps (Union[float, None]): error threshold for the approximation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""

        return CompressedResourceOp(cls, cls.num_wires, {"eps": eps})

    @classmethod
    def resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Args:
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with :code:`CNOT` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two :class:`~.CNOT` gates
            and two :class:`~.ResourceRZ` gates.
        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(RZ, {"eps": eps})
        return [GateCount(cnot, 2), GateCount(rz, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are expressed using the symbolic :class:`~.ResourceControlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.ResourceRZ` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]


class CRot(ResourceOperator):
    r"""Resource class for the CRot gate.

    Args:
        wires (Sequence[int], optional): the wire the operation acts on
        eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
        <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

        .. math::

            \begin{align}
                \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
            \end{align}

        This identity is applied along with some clever choices for the angle values to combine rotation;
        the final circuit takes the form:

        .. code-block:: bash

            ctrl: ─────╭●─────────╭●─────────┤
            trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

    .. seealso:: The corresponding PennyLane operation :class:`~.CRot`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRot.resource_decomp()
    [(2 x CNOT), (3 x RZ), (2 x RY)]
    """

    resource_keys = {"eps"}
    num_wires = 2

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * eps (Union[float, None]): error threshold for the approximation

        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"eps": eps})

    @classmethod
    def resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.
        Each GateCount object specifies a gate type and its total occurrence count.

        Args:
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `Gheorghiu, V., Mosca, M. & Mukhopadhyay
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math::

                \begin{align}
                    \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                    \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
                \end{align}

            This identity is applied along with some clever choices for the angle values to combine rotation;
            the final circuit takes the form:

            .. code-block:: bash

                ctrl: ─────╭●─────────╭●─────────┤
                trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(RZ, {"eps": eps})
        ry = resource_rep(RY, {"eps": eps})

        return [GateCount(cnot, 2), GateCount(rz, 3), GateCount(ry, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general rotation flips the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.Rot` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            Taking arbitrary powers of a general single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]


class ControlledPhaseShift(ResourceOperator):
    r"""Resource class for the ControlledPhaseShift gate.

    Args:
        wires (Sequence[int], optional): the wire the operation acts on
        eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are derived using the fact that a :class:`~.PhaseShift` gate is
        identical to the :class:`~.RZ` gate up to some global phase. Furthermore, a controlled
        global phase simplifies to a :class:`~.ResourcePhaseShift` gate. This gives rise to the
        following identity:

        .. math:: CR_\phi(\phi) = (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

        Specifically, the resources are defined as two :class:`~.CNOT` gates and three
        :class:`~.RZ` gates.

    .. seealso:: :class:`~.ControlledPhaseShift`

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.ControlledPhaseShift.resource_decomp()
    [(2 x CNOT), (3 x RZ)]
    """

    resource_keys = {"eps"}
    num_wires = 2

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * eps (Union[float, None]): error threshold for the approximation

        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"eps": eps})

    @classmethod
    def resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.

        Args:
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are derived using the fact that a :class:`~.ResourcePhaseShift` gate is
            identical to the :class:`~.ResourceRZ` gate up to some global phase. Furthermore, a controlled
            global phase simplifies to a :class:`~.ResourcePhaseShift` gate. This gives rise to the
            following identity:

            .. math:: CR_\phi(\phi) = (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

            Specifically, the resources are defined as two :class:`~.CNOT` gates and three
            :class:`~.ResourceRZ` gates.

        .. seealso:: :class:`~.ControlledPhaseShift`

        **Example**

        The resources for this operation are computed using:

        >>> qml.estimator.ControlledPhaseShift.resource_decomp()
        [(2 x CNOT), (3 x RZ)]
        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(RZ, {"eps": eps})
        return [GateCount(cnot, 2), GateCount(rz, 3)]

    @classmethod
    def adjoint_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a phase shift just flips the sign of the phase angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are expressed using the symbolic :class:`~.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.PhaseShift` class.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        raise ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, pow_z, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            Taking arbitrary powers of a phase shift produces a sum of shifts.
            The resources simplify to just one total phase shift operator.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]
