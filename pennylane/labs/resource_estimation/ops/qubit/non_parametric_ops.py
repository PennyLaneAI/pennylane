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
r"""Resource operators for non parametric single qubit operations."""
import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ


class ResourceHadamard(ResourceOperator):
    r"""Resource class for the Hadamard gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: :class:`~.Hadamard`

    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

        Raises:
            ResourcesNotDefined: This gate is fundamental, no further decomposition defined.
        """
        raise plre.ResourcesNotDefined

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep(), 1)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Hadamard gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(plre.resource_rep(plre.ResourceIdentity))]
        return [GateCount(cls.resource_rep())]


class ResourceS(ResourceOperator):
    r"""Resource class for the S-gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The S-gate decomposes into two T-gates.

    .. seealso:: :class:`~.S`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceS.resource_decomp()
    [(2 x T)]
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The S-gate decomposes into two T-gates.
        """
        t = resource_rep(ResourceT)
        return [GateCount(t, 2)]

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the S-gate is equivalent to :math:`\hat{Z} \dot \hat{S}`
            The resources are defined as one instance of Z-gate, and one instance of S-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        z = resource_rep(ResourceZ)
        return [GateCount(z, 1), GateCount(cls.resource_rep(), 1)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            - The S-gate, when raised to a power which is a multiple of four, produces identity.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to one, means one instance of the S-gate.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to two, means one instance of the Z-gate.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to three, means one instance of the Z-gate and one instance of S-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        mod_4 = pow_z % 4
        if mod_4 == 0:
            return [GateCount(resource_rep(plre.ResourceIdentity))]
        if mod_4 == 1:
            return [GateCount(cls.resource_rep())]
        if mod_4 == 2:
            return [GateCount(resource_rep(ResourceZ))]

        return [GateCount(resource_rep(ResourceZ)), GateCount(cls.resource_rep())]


class ResourceT(ResourceOperator):
    r"""Resource class for the T-gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The T-gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

    .. seealso:: :class:`~.T`

    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The T-gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :code:`ResourcesNotDefined` error.

        Raises:
            ResourcesNotDefined: This gate is fundamental, no further decomposition defined.
        """
        raise plre.ResourcesNotDefined

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of the T-gate is equivalent to the T-gate raised to the 7th power.
            The resources are defined as one Z-gate (:math:`Z = T^{4}`), one S-gate (:math:`S = T^{2}`) and one T-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        z = resource_rep(ResourceZ)
        s = resource_rep(ResourceS)
        return [GateCount(cls.resource_rep()), GateCount(s), GateCount(z)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The T-gate, when raised to a power which is a multiple of eight, produces identity.
            Consequently, for any integer power `z`, the effective quantum operation :math:`T^{z}` is equivalent
            to :math:`T^{z \pmod 8}`.

        The decomposition for :math:`T^{z}` (where :math:`z \pmod 8` is denoted as `z'`) is as follows:
            - If `z' = 0`: The operation is equivalent to the Identity gate (`I`).
            - If `z' = 1`: The operation is equivalent to the T-gate (`T`).
            - If `z' = 2`: The operation is equivalent to the S-gate (`S`).
            - If `z' = 3`: The operation is equivalent to a composition of an S-gate and a T-gate (:math:`S \cdot T`).
            - If `z' = 4` : The operation is equivalent to the Z-gate (`Z`).
            - If `z' = 5`: The operation is equivalent to a composition of a Z-gate and a T-gate (:math:`Z \cdot T`).
            - If `z' = 6`: The operation is equivalent to a composition of a Z-gate and an S-gate (:math:`Z \cdot S`).
            - If `z' = 7`: The operation is equivalent to a composition of a Z-gate, an S-gate and a T-gate.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """

        if (mod_8 := pow_z % 8) == 0:
            return [GateCount(resource_rep(plre.ResourceIdentity))]

        gate_lst = []
        if mod_8 >= 4:
            gate_lst.append(GateCount(resource_rep(ResourceZ)))
            mod_8 -= 4

        if mod_8 >= 2:
            gate_lst.append(GateCount(resource_rep(ResourceS)))
            mod_8 -= 2

        if mod_8 >= 1:
            gate_lst.append(GateCount(cls.resource_rep()))

        return gate_lst


class ResourceX(ResourceOperator):
    r"""Resource class for the X-gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The X-gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}.
            \end{align}

        Thus the resources for an X-gate are two :class:`~.ResourceS` gates and
        two :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.X`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceX.resource_decomp()
    [(2 x Hadamard), (2 x S)]
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The X-gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Z} &= \hat{S}^{2}.
                \end{align}

            Thus the resources for an X-gate are two :class:`~.ResourceS` gates and
            two :class:`~.ResourceHadamard` gates.
        """
        s = resource_rep(ResourceS)
        h = resource_rep(ResourceHadamard)

        return [GateCount(h, 2), GateCount(s, 2)]

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
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
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The X-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(plre.ResourceIdentity))]
        return [GateCount(cls.resource_rep())]


class ResourceY(ResourceOperator):
    r"""Resource class for the Y-gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The Y-gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}, \\
                \hat{S}^{\dagger} &= 3 \hat{S}.
            \end{align}

        Thus the resources for a Y-gate are six :class:`~.ResourceS` gates and
        two :class:`~.ResourceHadamard` gates.

    .. seealso:: :class:`~.Y`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceY.resource_decomp()
    [(6 x S), (2 x Hadamard)]
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The Y-gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Z} &= \hat{S}^{2}, \\
                    \hat{S}^{\dagger} &= 3 \hat{S}.
                \end{align}

            Thus the resources for a Y-gate are six :class:`~.ResourceS` gates and
            two :class:`~.ResourceHadamard` gates.
        """
        s = resource_rep(ResourceS)
        h = resource_rep(ResourceHadamard)

        return [GateCount(s, 6), GateCount(h, 2)]

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
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
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Y-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(plre.ResourceIdentity))]
        return [GateCount(cls.resource_rep())]


class ResourceZ(ResourceOperator):
    r"""Resource class for the Z-gate.

    Args:
        wires (Sequence[int] or int, optional): the wire the operation acts on

    Resources:
        The Z-gate can be decomposed according to the following identities:

        .. math:: \hat{Z} = \hat{S}^{2},

        thus the resources for a Z-gate are two :class:`~.ResourceS` gates.

    .. seealso:: :class:`~.Z`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceZ.resource_decomp()
    [(2 x S)]
    """

    num_wires = 1

    @classmethod
    def default_resource_decomp(cls, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The Z-gate can be decomposed according to the following identities:

            .. math:: \hat{Z} = \hat{S}^{2},

            thus the resources for a Z-gate are two :class:`~.ResourceS` gates.
        """
        s = resource_rep(ResourceS)
        return [GateCount(s, 2)]

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
        return CompressedResourceOp(cls, {})

    @classmethod
    def default_adjoint_resource_decomp(cls) -> list[GateCount]:
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
    def default_pow_resource_decomp(cls, pow_z) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            The Z-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(plre.ResourceIdentity))]
        return [GateCount(cls.resource_rep())]
