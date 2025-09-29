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
r"""Resource operators for parametric single qubit operations."""

import math

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ, signature-differs


def _rotation_resources(precision=1e-9):
    r"""Estimates the number of T gates needed to implement a Pauli rotation to a given precision.

    The expected T-count is taken from the "Simulation Results" section of `Efficient
    Synthesis of Universal Repeat-Until-Success Circuits <https://arxiv.org/abs/1404.5320>`_.
    The cost is given as:

        .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2,

    where :math:`\epsilon` is the provided ``precision``.

    Args:
        precision (float): The acceptable error threshold for the approximation.

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    num_gates = round(1.149 * math.log2(1 / precision) + 9.2)
    t = resource_rep(qre.T)
    return [GateCount(t, num_gates)]


class PhaseShift(ResourceOperator):
    r"""Resource class for the PhaseShift gate.

    Args:
        precision (float | None): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires | None): The wires the operation acts on.

    Resources:
        The phase shift gate is equivalent to a Z-rotation up to some global phase,
        as defined from the following identity:

        .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\phi}
                \end{bmatrix}.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PhaseShift`.

    **Example**

    The resources for this operation are computed as:

    >>> qml.estimator.PhaseShift.resource_decomp()
    [(1 x RZ), (1 x GlobalPhase)]
    """

    num_wires = 1
    resource_keys = {"precision"}

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
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Keyword Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            The phase shift gate is equivalent to a Z-rotation upto some global phase,
            as defined in the following identity:

            .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\phi}
                    \end{bmatrix}.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rz = resource_rep(RZ, {"precision": precision})
        global_phase = resource_rep(qre.GlobalPhase)
        return [GateCount(rz), GateCount(global_phase)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a phase shift operator just changes the sign of the phase, thus
            the resources of the adjoint operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision=precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.
        """
        precision = target_resource_params.get("precision")
        if num_ctrl_wires == 1:
            gate_types = [
                GateCount(resource_rep(qre.ControlledPhaseShift, {"precision": precision}))
            ]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        c_ps = resource_rep(qre.ControlledPhaseShift, {"precision": precision})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [qre.Allocate(1), GateCount(c_ps), GateCount(mcx, 2), qre.Deallocate(1)]

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision=precision))]


class RX(ResourceOperator):
    r"""Resource class for the RX gate.

    Args:
        precision (float | None): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires | None): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from the "Simulation Results" section of `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RX`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.RX.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

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
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Keyword Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from the "Simulation Results" section of `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.
        """
        precision = target_resource_params.get("precision")
        if num_ctrl_wires == 1:

            gate_types = [GateCount(resource_rep(qre.CRX, {"precision": precision}))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        h = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ, {"precision": precision})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )
        return [GateCount(h, 2), GateCount(rz, 2), GateCount(mcx, 2)]

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class RY(ResourceOperator):
    r"""Resource class for the RY gate.

    Args:
        precision (float | None): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires | None): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from the "Simulation Results" section of `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.RY.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

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
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Keyword Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from the "Simulation Results" section of `Efficient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
        """
        precision = target_resource_params.get("precision")
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.CRY, {"precision": precision}))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        ry = resource_rep(qre.RY, {"precision": precision})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(ry, 2), GateCount(mcx, 2)]

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class RZ(ResourceOperator):
    r"""Resource class for the RZ gate.

    Args:
        precision (float | None): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires | None): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from the "Simulation Results" section of `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.RZ.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

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
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from the "Simulation Results" section of `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} \approx 1.149 \times log_{2}(\frac{1}{\epsilon}) + 9.2

        Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
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
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.
        """
        precision = target_resource_params.get("precision")
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.CRZ, {"precision": precision}))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        rz = resource_rep(qre.RZ, {"precision": precision})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(rz, 2), GateCount(mcx, 2)]

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]


class Rot(ResourceOperator):
    r"""Resource class for the Rot gate.

    Args:
        precision (float | None): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires | None): The wires the operation acts on.

    Resources:
        The resources are obtained according to the definition of the ``Rot`` gate:

        .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.Rot`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.Rot.resource_decomp()
    [(1 x RY), (2 x RZ)]
    """

    num_wires = 1
    resource_keys = {"precision"}

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
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            The resources are obtained according to the definition of the ``Rot`` gate:

            .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

        """
        ry = resource_rep(RY, {"precision": precision})
        rz = resource_rep(RZ, {"precision": precision})

        return [GateCount(ry), GateCount(rz, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general single qubit rotation changes the sign of the rotation angles,
            thus the resources of the adjoint operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.
        """
        precision = target_resource_params.get("precision")
        if num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(qre.CRot, {"precision": precision}))]

            if num_zero_ctrl:
                gate_types.append(GateCount(resource_rep(qre.X), 2))

            return gate_types

        rz = resource_rep(qre.RZ, {"precision": precision})
        ry = resource_rep(qre.RY, {"precision": precision})
        mcx = resource_rep(
            qre.MultiControlledX,
            {
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(mcx, 2), GateCount(rz, 3), GateCount(ry, 2)]

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params.get("precision")
        return [GateCount(cls.resource_rep(precision))]
