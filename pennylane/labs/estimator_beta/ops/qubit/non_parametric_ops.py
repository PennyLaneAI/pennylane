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

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.wires import Wires, WiresLike


# pylint: disable=arguments-differ, unused-argument
class Hadamard(ResourceOperator):
    r"""Resource class for the Hadamard gate.

    Args:
        wires (WiresLike | None): the wire the operation acts on

    Resources:
        The Hadamard gate is treated as a fundamental gate and thus it cannot be decomposed
        further. Requesting the resources of this gate raises a :class:`~.pennylane.exceptions.ResourcesUndefinedError` error.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.Hadamard`.

    """

    num_wires = 1

    def __init__(self, wires: WiresLike | None = None):
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

            The ``RY(:math:`\frac{\pi}{4}`)`` and ``RY(:math:`\frac{-\pi}{4}`)`` gates are further decomposed as
            :math:`e^{-i\frac{\pi}{8}}SHTHS^{\dagger}` and :math:`e^{i\frac{\pi}{8}}}SHT^{\dagger}HS^{\dagger}`, respectively.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_lst = [GateCount(resource_rep(qre.CH))]

            if num_zero_ctrl:
                gate_lst.append(GateCount(resource_rep(qre.X), 2))

            return gate_lst

        gate_lst = []

        h = cls.resource_rep()
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        gate_lst.append(GateCount(h, 4))
        gate_lst.append(GateCount(resource_rep(qre.T), 1))
        gate_lst.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1)
        )
        gate_lst.append(GateCount(resource_rep(qre.S), 2))
        gate_lst.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 2)
        )
        gate_lst.append(GateCount(mcx))
        return gate_lst

    @classmethod
    def toffoli_based_controlled_decomp(
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

            The ``RY(:math:`\frac{\pi}{4}`)`` and ``RY(:math:`\frac{-\pi}{4}`)`` gates are decomposed as :math:`e^{-i\frac{\pi}{8}}SHTHS^{\dagger}` and :math:`e^{i\frac{\pi}{8}}}SHT^{\dagger}HS^{\dagger}`, respectively.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if num_ctrl_wires == 1:
            gate_lst = [GateCount(resource_rep(qre.CH))]

            if num_zero_ctrl:
                gate_lst.append(GateCount(resource_rep(qre.X), 2))

            return gate_lst

        gate_lst = []

        h = cls.resource_rep()
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        gate_lst.append(GateCount(h, 5))
        gate_lst.append(GateCount(resource_rep(qre.Toffoli), 1))
        gate_lst.append(GateCount(resource_rep(qre.S), 2))
        gate_lst.append(GateCount(resource_rep(qre.Adjoint, {"base_op": qre.S}), 1))

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
