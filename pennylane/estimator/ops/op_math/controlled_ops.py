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

from typing import Literal

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.wires import Wires, WiresLike

# pylint: disable= arguments-differ, signature-differs


class CH(ResourceOperator):
    r"""Resource class for the CH gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources are derived from the following identities:

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
            \end{align}

        Specifically, the resources are defined as two ``RY``, two ``Hadamard`` and one ``CNOT`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CH`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CH.resource_decomp()
    [(2 x Hadamard), (2 x RY), (1 x CNOT)]

    """

    num_wires = 2

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list of ``GateCount`` objects representing the resources of the operator..

        Resources:
            The resources are derived from the following identities:

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are defined as two ``RY``, two ``Hadamard`` and one ``CNOT`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        ry = resource_rep(qre.RY)
        h = resource_rep(qre.Hadamard)
        cnot = resource_rep(CNOT)
        return [GateCount(h, 2), GateCount(ry, 2), GateCount(cnot, 1)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Hadamard` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        ctrl_h = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Hadamard),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_h)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CY(ResourceOperator):
    r"""Resource class for the CY gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

        By replacing the ``X`` gate with a ``CNOT`` we obtain the controlled decomposition.
        Specifically, the resources are defined as a ``CNOT`` gate conjugated by a pair of
        ``S`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CY.resource_decomp()
    [(1 x CNOT), (1 x S), (1 x Adjoint(S))]
    """

    num_wires = 2

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

            By replacing the ``X`` gate with a ``CNOT`` we obtain the controlled decomposition.
            Specifically, the resources are defined as a ``CNOT`` gate conjugated by a pair of
            ``S`` gates.
        """
        cnot = resource_rep(CNOT)
        s = resource_rep(qre.S)
        s_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        return [GateCount(cnot), GateCount(s), GateCount(s_dag)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Y` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        ctrl_y = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Y),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_y)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CZ(ResourceOperator):
    r"""Resource class for the CZ gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the ``X`` gate with a ``CNOT`` we obtain the controlled decomposition.
        Specifically, the resources are defined as a ``CNOT`` gate conjugated by a pair of
        ``Hadamard`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CZ.resource_decomp()
    [(1 x CNOT), (2 x Hadamard)]
    """

    num_wires = 2

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the ``X`` gate with a ``CNOT`` we obtain the controlled decomposition.
            Specifically, the resources are defined as a ``CNOT`` gate conjugated by a pair of
            ``Hadamard`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(CNOT)
        h = resource_rep(qre.Hadamard)

        return [GateCount(cnot), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the original operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Z` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1 and num_zero_ctrl == 0:
            return [GateCount(resource_rep(CCZ))]

        ctrl_z = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Z),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_z)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CSWAP(ResourceOperator):
    r"""Resource class for the CSWAP gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/abs/2305.18128>`_.

        The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
        defined as:

        .. code-block:: bash

            0: ────╭●────┤
            1: ─╭X─├●─╭X─┤
            2: ─╰●─╰X─╰●─┤

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CSWAP`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CSWAP.resource_decomp()
    [(1 x Toffoli), (2 x CNOT)]
    """

    num_wires = 3

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The resources are taken from Figure 1d of `arXiv:2305.18128 <https://arxiv.org/abs/2305.18128>`_.

            The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
            defined as:

            .. code-block:: bash

                0: ────╭●────┤
                1: ─╭X─├●─╭X─┤
                2: ─╰●─╰X─╰●─┤

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        tof = resource_rep(Toffoli)
        cnot = resource_rep(CNOT)
        return [GateCount(tof), GateCount(cnot, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.SWAP` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        ctrl_swap = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.SWAP),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_swap)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CCZ(ResourceOperator):
    r"""Resource class for the CCZ gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

        By replacing the ``X`` gate with a ``Toffoli`` we obtain the controlled decomposition.
        Specifically, the resources are defined as a ``Toffoli`` gate conjugated by a pair of
        ``Hadamard`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CCZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CCZ.resource_decomp()
    [(1 x Toffoli), (2 x Hadamard)]
    """

    num_wires = 3

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The resources are derived from the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            By replacing the ``X`` gate with a ``Toffoli`` we obtain the controlled decomposition.
            Specifically, the resources are defined as a ``Toffoli`` gate conjugated by a pair of
            ``Hadamard`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        toffoli = resource_rep(Toffoli)
        h = resource_rep(qre.Hadamard)
        return [GateCount(toffoli), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Z` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        ctrl_z = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Z),
                "num_ctrl_wires": num_ctrl_wires + 2,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(ctrl_z)]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class CNOT(ResourceOperator):
    r"""Resource class for the CNOT gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed further.
        Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CNOT`.

    """

    num_wires = 2

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The CNOT gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

        Raises:
            ResourcesUndefinedError: This gate is fundamental, no further decomposition defined.
        """
        raise ResourcesUndefinedError

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed as one general :class:`~.pennylane.estimator.ops.MultiControlledX` gate.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        if num_ctrl_wires == 1 and num_zero_ctrl == 0:
            return [GateCount(resource_rep(Toffoli))]

        mcx = resource_rep(
            MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [
            GateCount(mcx),
        ]

    @classmethod
    def pow_resource_decomp(
        cls, pow_z: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep())]
        )


class TemporaryAND(ResourceOperator):
    r"""Resource class representing a `TemporaryAND` gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    This gate was introduced in Fig 4 of `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ along
    with its adjoint.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.TemporaryAND`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.TemporaryAND.resource_decomp()
    [(1 x Toffoli)]
    """

    num_wires = 3

    def __init__(self, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

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
        the operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            The resources are obtained from Figure 4 of `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        tof = resource_rep(Toffoli, {"elbow": "left"})
        return [GateCount(tof)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are obtained from Figure 4 of `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = resource_rep(qre.Hadamard)
        cz = resource_rep(CZ)
        return [GateCount(h), GateCount(cz)]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed as one general :class:`~.pennylane.estimator.ops.MultiControlledX` gate.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires + 2,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(mcx)]


class Toffoli(ResourceOperator):
    r"""Resource class for the Toffoli gate.

    Args:
        wires (Sequence[int] | None): the subsystem the gate acts on
        elbow (str | None): String identifier to determine if this is a special type of
            Toffoli gate. Available options are `left`, `right`, and `None`.

    Resources:
        If `elbow` is provided, resources are obtained from Figure 4 of
        `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.

        If `elbow` is `None`, the resources are obtained from Figure 1 of
        `Jones (2012) <https://arxiv.org/pdf/1212.5069>`_.

        The circuit which applies the Toffoli gate on target wire 'target' with control wires
        ('c1', 'c2') is defined as:

        .. code-block:: bash

                c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
              aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
              aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
            target: ─────────────────────────────────╰X──────║───║─┤
                                                             ╚═══╝

        Specifically, the resources are defined as nine ``CNOT`` gates, three ``Hadamard`` gates,
        one ``CZ`` gate, one ``S`` gate, two ``T`` gates and two adjoint ``T`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.Toffoli`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.Toffoli.resource_decomp()
    [Allocate(2), (9 x CNOT), (3 x Hadamard), (1 x S), (1 x CZ), (2 x T), (2 x Adjoint(T)), Deallocate(2)]
    """

    num_wires = 3
    resource_keys = {"elbow"}

    def __init__(
        self, elbow: Literal["left", "right"] | None = None, wires: WiresLike = None
    ) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.elbow = elbow
        super().__init__(wires=wires)

    @staticmethod
    def elbow_decomp(elbow: Literal["left", "right"] | None = "left"):
        """A function that prepares the resource decomposition obtained from Figure 4 of
        `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_.

        Args:
            elbow (str | None): One of "left" or "right". Defaults to "left".

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: The resources of decomposing the elbow gates.
        """
        gate_types = []
        t = resource_rep(qre.T)
        t_dag = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": t},
        )
        h = resource_rep(qre.Hadamard)
        cnot = resource_rep(CNOT)
        s_dag = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": resource_rep(qre.S)},
        )
        cz = resource_rep(CZ)

        if elbow == "left":
            gate_types.append(GateCount(t, 2))
            gate_types.append(GateCount(t_dag, 2))
            gate_types.append(GateCount(cnot, 3))
            gate_types.append(GateCount(s_dag))

        if elbow == "right":
            gate_types.append(GateCount(h))
            gate_types.append(GateCount(cz))

        return gate_types

    @classmethod
    def resource_decomp(cls, elbow: Literal["left", "right"] | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Resources:
            If `elbow` is provided, resources are obtained from Figure 4 of
            `arXiv:1805.03662 <https://arxiv.org/pdf/1805.03662>`_.

            If `elbow` is `None`, the resources are obtained from Figure 1 of
            `Jones (2012) <https://arxiv.org/pdf/1212.5069>`_.

            The circuit which applies the Toffoli gate on target wire 'target' with control wires
            ('c1', 'c2') is defined as:

            .. code-block:: bash

                  c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                  c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
                aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
                aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
              target: ─────────────────────────────────╰X──────║───║─┤
                                                               ╚═══╝

            Specifically, the resources are defined as nine ``CNOT`` gates, three ``Hadamard`` gates,
            one ``CZ`` gate, one ``S`` gate, two ``T`` gates and two adjoint ``T`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if elbow:
            return Toffoli.elbow_decomp(elbow)

        cnot = resource_rep(CNOT)
        t = resource_rep(qre.T)
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        cz = resource_rep(CZ)
        t_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": t})

        return [
            Allocate(2),
            GateCount(cnot, 9),
            GateCount(h, 3),
            GateCount(s),
            GateCount(cz),
            GateCount(t, 2),
            GateCount(t_dag, 2),
            Deallocate(2),
        ]

    @classmethod
    def textbook_resource_decomp(
        cls, elbow: Literal["left", "right"] | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            elbow (str | None): String identifier to determine if this is a special type of
                Toffoli gate (left or right elbow). Default value is `None`.

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

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if elbow:
            return Toffoli.elbow_decomp(elbow)

        cnot = resource_rep(CNOT)
        t = resource_rep(qre.T)
        h = resource_rep(qre.Hadamard)
        t_dag = resource_rep(qre.Adjoint, {"base_cmpr_op": t})

        return [GateCount(cnot, 6), GateCount(h, 2), GateCount(t, 4), GateCount(t_dag, 3)]

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            A dictionary containing the resource parameters:
                * elbow (str | None): String identifier to determine if this is a special type of Toffoli gate (left or right elbow).

        """
        return {"elbow": self.elbow}

    @classmethod
    def resource_rep(cls, elbow: Literal["left", "right"] | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            elbow (str | None): String identifier to determine if this is a special type of Toffoli gate (left or right elbow).

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {"elbow": elbow})

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        elbow = target_resource_params.get("elbow")
        if elbow is None:
            return [GateCount(cls.resource_rep())]

        adj_elbow = "left" if elbow == "right" else "right"
        return [GateCount(cls.resource_rep(elbow=adj_elbow))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are expressed as one general :class:`~.pennylane.estimator.ops.MultiControlledX` gate.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires + 2,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(mcx)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        elbow = target_resource_params.get("elbow")
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep(elbow=elbow))]
        )


class MultiControlledX(ResourceOperator):
    r"""Resource class for the MultiControlledX gate.

    Args:
        num_ctrl_wires (int | None): the number of qubits the operation is controlled on
        num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        wires (Sequence[int] | None): the wires this operation acts on

    Resources:
        The resources are obtained based on the unary iteration technique described in
        `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_. Specifically, the
        resources are defined as the following rules:

        * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

        * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

        * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

        * If there are three or more control qubits (:math:`n`), the resources obtained based on the unary iteration technique described in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_. Specifically, it requires :math:`n - 2` clean qubits, and produces :math:`n - 2` elbow gates and a single :class:`~.pennylane.estimator.ops.Toffoli`.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.MultiControlledX`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.MultiControlledX.resource_decomp(num_ctrl_wires=5, num_zero_ctrl=2)
    [(4 x X), Allocate(3), (3 x TemporaryAND), (3 x Adjoint(TemporaryAND)), (1 x Toffoli), Deallocate(3)]
    """

    resource_keys = {"num_ctrl_wires", "num_zero_ctrl"}

    def __init__(
        self, num_ctrl_wires: int | None = None, num_zero_ctrl: int = 0, wires: WiresLike = None
    ) -> None:

        if num_ctrl_wires is None:
            if wires is None:
                raise ValueError("Must provide atleast one of `num_ctrl_wires` and `wires`.")

            num_ctrl_wires = len(wires) - 1

        self.num_ctrl_wires = num_ctrl_wires
        self.num_zero_ctrl = num_zero_ctrl

        self.num_wires = num_ctrl_wires + 1
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_ctrl_wires (int): the number of qubits the operation is controlled on
                * num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
        """

        return {
            "num_ctrl_wires": self.num_ctrl_wires,
            "num_zero_ctrl": self.num_zero_ctrl,
        }

    @classmethod
    def resource_rep(cls, num_ctrl_wires: int, num_zero_ctrl: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = num_ctrl_wires + 1
        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

    @classmethod
    def resource_decomp(cls, num_ctrl_wires: int, num_zero_ctrl: int) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state

        Resources:
            The resources are obtained based on the unary iteration technique described in
            `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_. Specifically, the
            resources are defined as the following rules:

            * If there are no control qubits, treat the operation as a :class:`~.pennylane.estimator.ops.X` gate.

            * If there is only one control qubit, treat the resources as a :class:`~.pennylane.estimator.ops.CNOT` gate.

            * If there are two control qubits, treat the resources as a :class:`~.pennylane.estimator.ops.Toffoli` gate.

            * If there are three or more control qubits (:math:`n`), the resources obtained based on the unary iteration technique described in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_. Specifically, it requires :math:`n - 2` clean qubits, and produces :math:`n - 2` elbow gates and a single :class:`~.pennylane.estimator.ops.Toffoli`.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_lst = []

        x = resource_rep(qre.X)
        if num_ctrl_wires == 0:
            if num_zero_ctrl:
                return []

            return [GateCount(x)]

        if num_zero_ctrl:
            gate_lst.append(GateCount(x, num_zero_ctrl * 2))

        cnot = resource_rep(CNOT)
        if num_ctrl_wires == 1:
            gate_lst.append(GateCount(cnot))
            return gate_lst

        toffoli = resource_rep(Toffoli)
        if num_ctrl_wires == 2:
            gate_lst.append(GateCount(toffoli))
            return gate_lst

        l_elbow = resource_rep(TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        res = [
            Allocate(num_ctrl_wires - 2),
            GateCount(l_elbow, num_ctrl_wires - 2),
            GateCount(r_elbow, num_ctrl_wires - 2),
            GateCount(toffoli, 1),
            Deallocate(num_ctrl_wires - 2),
        ]
        gate_lst.extend(res)
        return gate_lst

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation results
            are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_ctrl_wires = target_resource_params.get("num_ctrl_wires")
        num_zero_ctrl = target_resource_params.get("num_zero_ctrl")
        return [GateCount(cls.resource_rep(num_ctrl_wires, num_zero_ctrl))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of control qubits of the operation
            num_zero_ctrl (int): The subset of control qubits of the operation, that are controlled
                when in the :math:`|0\rangle` state.
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are derived by combining the control qubits, control-values and
            into a single instance of ``MultiControlledX`` gate, controlled
            on the whole set of control-qubits.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        base_ctrl_wires = target_resource_params.get("num_ctrl_wires")
        base_ctrl_zero = target_resource_params.get("num_zero_ctrl")
        return [
            GateCount(
                cls.resource_rep(
                    num_ctrl_wires + base_ctrl_wires,
                    num_zero_ctrl + base_ctrl_zero,
                )
            )
        ]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            This operation is self-inverse, thus when raised to even integer powers acts like
            the identity operator and raised to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_ctrl_wires = target_resource_params.get("num_ctrl_wires")
        num_zero_ctrl = target_resource_params.get("num_zero_ctrl")
        return (
            [GateCount(resource_rep(qre.Identity))]
            if pow_z % 2 == 0
            else [GateCount(cls.resource_rep(num_ctrl_wires, num_zero_ctrl))]
        )


class CRX(ResourceOperator):
    r"""Resource class for the CRX gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on
        precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the precision stated in the config.

    Resources:
        The resources are taken from Figure 1b of `arXiv:2110.10292
        <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

        we can express the ``CRX`` gate as a ``CRZ`` gate conjugated by ``Hadamard``
        gates. The expression for controlled-RZ gates is used as defined in the reference above.
        Specifically, the resources are defined as two ``CNOT`` gates, two ``Hadamard`` gates
        and two ``RZ`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CRX`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRX.resource_decomp()
    [(2 x CNOT), (2 x RZ), (2 x Hadamard)]
    """

    resource_keys = {"precision"}
    num_wires = 2

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float | None): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            precision (float | None): The error threshold for the Clifford + T decomposition of this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """

        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `arXiv:2110.10292
            <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the ``CRX`` gate as a ``CRZ`` gate conjugated by ``Hadamard``
            gates. The expression for controlled-RZ gates is used as defined in the reference above.
            Specifically, the resources are defined as two ``CNOT`` gates, two ``Hadamard`` gates and
            two ``RZ`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ, {"precision": precision})

        cnot = resource_rep(CNOT)

        return [GateCount(cnot, 2), GateCount(rz, 2), GateCount(h, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.RX` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        ctrl_rx = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.RX, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_rx)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class CRY(ResourceOperator):
    r"""Resource class for the CRY gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on
        precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `arXiv:2110.10292
        <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

        By replacing the ``X`` gates with ``CNOT`` gates, we obtain a
        controlled-version of this identity. Thus we are able to constructively or destructively
        interfere the gates based on the value of the control qubit. Specifically, the resources are
        defined as two ``CNOT`` gates and two ``RY`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.CRY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRY.resource_decomp()
    [(2 x CNOT), (2 x RY)]
    """

    resource_keys = {"precision"}
    num_wires = 2

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float | None): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            precision (float | None): The error threshold for the Clifford + T decomposition of this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `arXiv:2110.10292
            <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

            By replacing the ``X`` gates with ``CNOT`` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two ``CNOT`` gates
            and two ``RY`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(CNOT)
        ry = resource_rep(qre.RY, {"precision": precision})
        return [GateCount(cnot, 2), GateCount(ry, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.RY` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        ctrl_ry = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.RY, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_ry)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class CRZ(ResourceOperator):
    r"""Resource class for the CRZ gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on
        precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `arXiv:2110.10292
        <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

        .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

        By replacing the ``X`` gates with ``CNOT`` gates, we obtain a controlled-version of this
        identity. Thus we are able to constructively or destructively interfere the gates based on the value
        of the control qubit. Specifically, the resources are defined as two ``CNOT`` gates
        and two ``RZ`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.CRZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.CRZ.resource_decomp()
    [(2 x CNOT), (2 x RZ)]
    """

    resource_keys = {"precision"}
    num_wires = 2

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float | None): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            precision (float | None): The error threshold for the Clifford + T decomposition of this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """

        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `arXiv:2110.10292
            <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

            By replacing the ``X`` gates with ``CNOT`` gates, we obtain a controlled-version of this
            identity. Thus we are able to constructively or destructively interfere the gates based on the value
            of the control qubit. Specifically, the resources are defined as two ``CNOT`` gates
            and two ``RZ`` gates.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(qre.RZ, {"precision": precision})
        return [GateCount(cnot, 2), GateCount(rz, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.RZ` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        ctrl_rz = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.RZ, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_rz)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class CRot(ResourceOperator):
    r"""Resource class for the CRot gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on
        precision (float | None): The error threshold for Clifford + T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are taken from Figure 1b of `arXiv:2110.10292
        <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

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

    resource_keys = {"precision"}
    num_wires = 2

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float | None): the number of qubits the operation is controlled on

        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            precision (float | None): The error threshold for the Clifford + T decomposition of this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator.

        Args:
            precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are taken from Figure 1b of `arXiv:2110.10292
            <https://arxiv.org/abs/2110.10292>`_. In combination with the following identity:

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

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(qre.RZ, {"precision": precision})
        ry = resource_rep(qre.RY, {"precision": precision})

        return [GateCount(cnot, 2), GateCount(rz, 3), GateCount(ry, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a general rotation flips the sign of the rotation angle,
            thus the resources of the adjoint operation result are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.Rot` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        ctrl_rot = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.Rot, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_rot)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of a general single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class ControlledPhaseShift(ResourceOperator):
    r"""Resource class for the ControlledPhaseShift gate.

    Args:
        wires (Sequence[int] | None): the wire the operation acts on
        precision (float | None): The error threshold for Clifford + T decomposition of the rotation gate.
            The default value is `None` which corresponds to using the epsilon stated in the config.

    Resources:
        The resources are derived using the fact that a ``PhaseShift`` gate is
        identical to the ``RZ`` gate up to some global phase. Furthermore, a controlled
        global phase simplifies to a ``PhaseShift`` gate. This gives rise to the
        following identity:

        .. math:: CR_\phi(\phi) = (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

        Specifically, the resources are defined as two ``CNOT`` gates and three ``RZ`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ControlledPhaseShift`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.ControlledPhaseShift.resource_decomp()
    [(2 x CNOT), (3 x RZ)]
    """

    resource_keys = {"precision"}
    num_wires = 2

    def __init__(self, precision: float | None = None, wires: WiresLike = None) -> None:
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (float | None): the number of qubits the operation is controlled on

        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute the resources.

        Args:
            precision (float | None): The error threshold for the Clifford + T decomposition of this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: A compressed representation of the operator.
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the resources of the operator.

        Args:
            precision (float | None): The error threshold for clifford plus T decomposition of the rotation gate.
                The default value is `None` which corresponds to using the epsilon stated in the config.

        Resources:
            The resources are derived using the fact that a ``PhaseShift`` gate is
            identical to the ``RZ`` gate up to some global phase. Furthermore, a controlled
            global phase simplifies to a ``PhaseShift`` gate. This gives rise to the
            following identity:

            .. math:: CR_\phi(\phi) = (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

            Specifically, the resources are defined as two ``CNOT`` gates and three ``RZ`` gates.

        .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.ControlledPhaseShift`.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.

        **Example**

        The resources for this operation are computed using:

        >>> qml.estimator.ControlledPhaseShift.resource_decomp()
        [(2 x CNOT), (3 x RZ)]
        """
        cnot = resource_rep(CNOT)
        rz = resource_rep(qre.RZ, {"precision": precision})

        return [GateCount(cnot, 2), GateCount(rz, 3)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a phase shift just flips the sign of the phase angle,
            thus the resources of the adjoint operation result are same as the originial operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are expressed using the symbolic :class:`~.pennylane.estimator.ops.Controlled`. The resources
            are computed according to the :code:`controlled_resource_decomp()` of the base
            :class:`~.pennylane.estimator.ops.PhaseShift` class.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        ctrl_ps = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(qre.PhaseShift, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ctrl_ps)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Taking arbitrary powers of a phase shift produces a sum of shifts.
            The resources simplify to just one total phase shift operator.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]
