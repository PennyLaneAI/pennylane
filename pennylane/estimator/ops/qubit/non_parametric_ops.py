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

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.wires import Wires

# pylint: disable=arguments-differ


class Hadamard(ResourceOperator):
    r"""Resource class for the Hadamard gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.Hadamard`.

    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``Hadamard`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The ``Hadamard`` gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

        Raises:
            ResourcesUndefinedError: This gate is fundamental, no further decomposition defined.
        """
        raise ResourcesUndefinedError

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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

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
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(), 1)]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            For a single control wire, the cost is a single instance of ``CH``.
            Two additional ``X`` gates are used to flip the control qubit if it is zero-controlled.
            In the case where multiple controlled wires are provided, the resources are derived from
            the following identities:

            .. math::

                \begin{align}
                    \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \cdot \hat{Z}  \cdot \hat{R}_{y}(\frac{-\pi}{4}), \\
                    \hat{Z} &= \hat{H} \cdot \hat{X}  \cdot \hat{H}.
                \end{align}

            Specifically, the resources are given by two ``RY`` gates, two
            ``Hadamard`` gates and a ``X`` gate. By replacing the
            ``X`` gate with ``MultiControlledX`` gate, we obtain a
            controlled-version of this identity.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_lst = [GateCount(resource_rep(qre.CH))]

            if num_zero_ctrl:
                gate_lst.append(GateCount(resource_rep(X), 2))

            return gate_lst

        gate_lst = []

        ry = resource_rep(qre.RY)
        h = cls.resource_rep()
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        gate_lst.append(GateCount(h, 2))
        gate_lst.append(GateCount(ry, 2))
        gate_lst.append(GateCount(mcx))
        return gate_lst

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
            The Hadamard gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep())]


class S(ResourceOperator):
    r"""Resource class for the S-gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The ``S`` gate decomposes into two ``T`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.S`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.S.resource_decomp()
    [(2 x T)]
    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``S`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The S-gate decomposes into two T-gates.
        """
        t = resource_rep(T)
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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of the ``S`` gate is equivalent to :math:`\hat{Z} \cdot \hat{S}`.
            The resources are defined as one instance of Z-gate, and one instance of S-gate.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        z = resource_rep(Z)
        return [GateCount(z, 1), GateCount(cls.resource_rep(), 1)]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The controlled-S gate decomposition is presented in (Fig. 5)
            `arXiv:1803.04933 <https://arxiv.org/pdf/1803.04933>`_. Given a single control wire, the
            cost is therefore two ``CNOT`` gates and three ``T`` gates.
            Two additional ``X`` gates are used to flip the control qubit if it is
            zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one auxiliary qubit in a `zeroed` state, which is reset at the end.
            In this case the cost increases by two additional ``MultiControlledX`` gates,
            as described in (Lemma 7.11) `Barenco et al. arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_lst = [
                GateCount(resource_rep(qre.CNOT), 2),
                GateCount(resource_rep(T), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(T)})),
            ]

            if num_zero_ctrl:
                gate_lst.append(GateCount(resource_rep(X), 2))

            return gate_lst

        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [
            GateCount(mcx, 2),
            GateCount(resource_rep(qre.CNOT), 2),
            GateCount(resource_rep(T), 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(T)})),
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
            - The S-gate, when raised to a power which is a multiple of four, produces identity.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to one, means one instance of the S-gate.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to two, means one instance of the Z-gate.
            - The cost of raising to an arbitrary integer power :math:`z`, when :math:`z \mod 4`
              is equal to three, means one instance of the Z-gate and one instance of S-gate.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        mod_4 = pow_z % 4
        if mod_4 == 0:
            return []
        if mod_4 == 1:
            return [GateCount(cls.resource_rep())]
        if mod_4 == 2:
            return [GateCount(resource_rep(Z))]

        return [GateCount(resource_rep(Z)), GateCount(cls.resource_rep())]


class SWAP(ResourceOperator):
    r"""Resource class for the SWAP gate.

    Args:
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources come from the following identity expressing SWAP as the product of
        three ``CNOT`` gates:

        .. math::

            SWAP = \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0\\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1
                    \end{bmatrix}
            =  \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
                \end{bmatrix}
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0\\
                    0 & 1 & 0 & 0
                \end{bmatrix}
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 1\\
                    0 & 0 & 1 & 0
            \end{bmatrix}.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.SWAP`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.SWAP.resource_decomp()
    [(3 x CNOT)]

    """

    num_wires = 2

    def __init__(self, wires=None):
        """Initializes the ``SWAP`` operator."""
        if wires is not None and len(Wires(wires)) != 2:
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
        the operator that are needed to compute a resource estimation."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The resources come from the following identity expressing SWAP as the product of
            three ``CNOT`` gates:

            .. math::

                SWAP = \begin{bmatrix}
                            1 & 0 & 0 & 0 \\
                            0 & 0 & 1 & 0\\
                            0 & 1 & 0 & 0\\
                            0 & 0 & 0 & 1
                        \end{bmatrix}
                =  \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0
                    \end{bmatrix}
                    \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0\\
                        0 & 1 & 0 & 0
                    \end{bmatrix}
                    \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0\\
                        0 & 0 & 0 & 1\\
                        0 & 0 & 1 & 0
                \end{bmatrix}.
        """
        return [GateCount(resource_rep(qre.CNOT), 3)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            This operation is self-adjoint, so the resources of the adjoint operation
            are same as the original operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            For a single control wire, the cost is a single instance of ``CSWAP``.
            Two additional ``X`` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are given by
            two ``CNOT`` gates and one ``MultiControlledX`` gate. This
            is because of the symmetric resource decomposition of the SWAP gate. By controlling on
            the middle CNOT gate, we obtain the required controlled operation.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.CSWAP))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(X), 2))

            return gate_types

        cnot = resource_rep(qre.CNOT)
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires + 1,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(cnot, 2), GateCount(mcx)]

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
            The ``SWAP`` gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep())]


class T(ResourceOperator):
    r"""Resource class for the T-gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The ``T`` gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.T`.

    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``T`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The ``T`` gate is treated as a fundamental gate and thus it cannot be decomposed
            further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

        Raises:
            ResourcesUndefinedError: This gate is fundamental, no further decomposition defined.
        """
        raise ResourcesUndefinedError

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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of the T-gate is equivalent to the T-gate raised to the 7th power.
            The resources are defined as one Z-gate (:math:`Z = T^{4}`), one S-gate (:math:`S = T^{2}`) and one T-gate.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        z = resource_rep(Z)
        s = resource_rep(S)
        return [GateCount(cls.resource_rep()), GateCount(s), GateCount(z)]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict | None = None
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The T-gate is equivalent to the PhaseShift gate for some fixed phase. Given a single
            control wire, the cost is therefore a single instance of
            ``ControlledPhaseShift``. Two additional ``X`` gates are
            used to flip the control qubit if it is zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one auxiliary qubit in a `zeroed` state, which is reset at the end.
            In this case the cost increases by two additional ``MultiControlledX`` gates,
            as described in (Lemma 7.11) `Barenco et al. arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.ControlledPhaseShift))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(X), 2))

            return gate_types

        ct = resource_rep(qre.ControlledPhaseShift)
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(ct), GateCount(mcx, 2)]

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
            The ``T`` gate, when raised to a power which is a multiple of eight, produces identity.
            Consequently, for any integer power `z`, the effective quantum operation :math:`T^{z}` is equivalent
            to :math:`T^{z \pmod 8}`.

            The decomposition for :math:`T^{z}`, where :math:`z \pmod 8` is denoted as `z'`, is as follows:

            - If `z' = 0`: The operation is equivalent to the Identity gate (:math:`I`).
            - If `z' = 1`: The operation is equivalent to the T-gate (:math:`T`).
            - If `z' = 2`: The operation is equivalent to the S-gate (:math:`S`).
            - If `z' = 3`: The operation is equivalent to a composition of an S-gate and a T-gate (:math:`S \cdot T`).
            - If `z' = 4` : The operation is equivalent to the Z-gate (:math:`Z`).
            - If `z' = 5`: The operation is equivalent to a composition of a Z-gate and a T-gate (:math:`Z \cdot T`).
            - If `z' = 6`: The operation is equivalent to a composition of a Z-gate and an S-gate (:math:`Z \cdot S`).
            - If `z' = 7`: The operation is equivalent to a composition of a Z-gate, an S-gate and a T-gate.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        if (mod_8 := pow_z % 8) == 0:
            return [GateCount(resource_rep(qre.Identity))]

        gate_lst = []
        if mod_8 >= 4:
            gate_lst.append(GateCount(resource_rep(Z)))
            mod_8 -= 4

        if mod_8 >= 2:
            gate_lst.append(GateCount(resource_rep(S)))
            mod_8 -= 2

        if mod_8 >= 1:
            gate_lst.append(GateCount(cls.resource_rep()))

        return gate_lst


class X(ResourceOperator):
    r"""Resource class for the X-gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The ``X`` gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Z} &= \hat{S}^{2}.
            \end{align}

        Thus the resources for an X-gate are two ``S`` gates and two ``Hadamard`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PauliX`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.X.resource_decomp()
    [(2 x Hadamard), (2 x S)]
    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``X`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The ``X`` gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Z} &= \hat{S}^{2}.
                \end{align}

            Thus the resources for an X-gate are two ``S`` gates and two ``Hadamard`` gates.
        """
        s = resource_rep(S)
        h = resource_rep(Hadamard)

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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

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
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep())]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict | None = None,
    ):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            For one or two control wires, the cost is one of ``CNOT`` or ``Toffoli`` respectively.
            Two additional ``X`` gates per control qubit are used to flip the control qubits
            if they are zero-controlled. In the case where multiple controlled wires are provided,
            the cost is one general ``MultiControlledX`` gate.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        if num_ctrl_wires > 2:
            mcx = resource_rep(
                qre.MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            return [GateCount(mcx)]

        gate_lst = []
        if num_zero_ctrl:
            gate_lst.append(GateCount(resource_rep(X), 2 * num_zero_ctrl))

        if num_ctrl_wires == 0:
            gate_lst.append(GateCount(resource_rep(X)))

        elif num_ctrl_wires == 1:
            gate_lst.append(GateCount(resource_rep(qre.CNOT)))

        else:
            gate_lst.append(GateCount(resource_rep(qre.Toffoli)))

        return gate_lst

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
            The X-gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep())]


class Y(ResourceOperator):
    r"""Resource class for the Y-gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The ``Y`` gate can be decomposed according to the following identities:

        .. math::

            \begin{align}
                \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{S}^{\dagger} &= \hat{Z} \cdot \hat{S}. \\
            \end{align}

        Thus the resources for a Y-gate are two ``S`` gates, two ``Z`` gates
        and two ``Hadamard`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PauliY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.Y.resource_decomp()
    [(1 x S), (1 x Z), (1 x Adjoint(S)), (2 x Hadamard)]
    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``Y`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The ``Y`` gate can be decomposed according to the following identities:

            .. math::

                \begin{align}
                    \hat{Y} &= \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}, \\
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{S}^{\dagger} &= \hat{Z} \cdot \hat{S}. \\
                \end{align}

            Thus the resources for a Y-gate are one S-gate, one Adjoint(S)-gate,
            one Z-gate and two Hadamard gates.
        """
        z = resource_rep(Z)
        s = resource_rep(S)
        s_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": s})
        h = resource_rep(Hadamard)

        return [GateCount(s), GateCount(z), GateCount(s_adj), GateCount(h, 2)]

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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

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
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            For a single control wire, the cost is a single instance of ``CY``.
            Two additional ``X`` gates are used to flip the control qubit if
            it is zero-controlled. In the case where multiple controlled wires
            are provided, the resources are derived from the following identity:

            .. math:: \hat{Y} = \hat{S} \cdot \hat{X} \cdot \hat{S}^{\dagger}.

            Specifically, the resources are given by a ``X`` gate conjugated with
            a pair of ``S`` gates. By replacing the ``X`` gate with a ``MultiControlledX``
            gate, we obtain a controlled-version of this identity.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.CY))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(X), 2))

            return gate_types

        s = resource_rep(S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(s), GateCount(s_dagg), GateCount(mcx)]

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
            The ``Y`` gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep())]


class Z(ResourceOperator):
    r"""Resource class for the Z-gate.

    Args:
        wires (Sequence[int] | int | None): the wire the operation acts on

    Resources:
        The ``Z`` gate can be decomposed according to the following identities:

        .. math:: \hat{Z} = \hat{S}^{2},

        thus the resources for a Z-gate are two ``S`` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PauliZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.Z.resource_decomp()
    [(2 x S)]
    """

    num_wires = 1

    def __init__(self, wires=None):
        """Initializes the ``Z`` operator."""
        if wires is not None and len(Wires(wires)) != 1:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The ``Z`` gate can be decomposed according to the following identities:

            .. math:: \hat{Z} = \hat{S}^{2},

            thus the resources for a Z-gate are two ``S`` gates.
        """
        s = resource_rep(S)
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
        the operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, num_wires=cls.num_wires, params={})

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
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            For one or two control wires, the cost is one of ``CZ``
            or ``CCZ`` respectively. Two additional ``X`` gates
            per control qubit are used to flip the control qubits if they are zero-controlled.
            In the case where multiple controlled wires are provided, the resources are derived from
            the following identity:

            .. math:: \hat{Z} = \hat{H} \cdot \hat{X} \cdot \hat{H}.

            Specifically, the resources are given by a ``X`` gate conjugated with
            a pair of ``Hadamard`` gates. By replacing the ``X`` gate with a
            ``MultiControlledX`` gate, we obtain a controlled-version of this identity.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of GateCount objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires > 2:
            h = resource_rep(Hadamard)
            mcx = resource_rep(
                qre.MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            return [GateCount(h, 2), GateCount(mcx)]

        gate_list = []
        if num_ctrl_wires == 1:
            gate_list.append(GateCount(resource_rep(qre.CZ)))

        if num_ctrl_wires == 2:
            gate_list.append(GateCount(resource_rep(qre.CCZ)))

        if num_zero_ctrl:
            gate_list.append(GateCount(resource_rep(X), 2 * num_zero_ctrl))

        return gate_list

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
            The ``Z`` gate raised to even powers produces identity and raised
            to odd powers it produces itself.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if pow_z % 2 == 0:
            return [GateCount(resource_rep(qre.Identity))]
        return [GateCount(cls.resource_rep())]
