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

import numpy as np

from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.exceptions import ResourcesUndefinedError

from ..identity import GlobalPhase
from .non_parametric_ops import T

# pylint: disable=arguments-differ


def _rotation_resources(precision=10e-3):
    r"""An estimate on the number of T gates needed to implement a Pauli rotation.

    The expected T-count is taken from (the 'Simulation Results' section) `Efficient
    Synthesis of Universal Repeat-Until-Success Circuits <https://arxiv.org/abs/1404.5320>`_.
    The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil,

    where :math:`\epsilon` is the provided ``precision``.

    Args:
        precision (float): the acceptable error threshold for the approximation

    Returns:
        list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
        where each object represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    num_gates = round(1.149 * np.log2(1 / precision) + 9.2)
    t = resource_rep(T)
    return [GateCount(t, num_gates)]


class PhaseShift(ResourceOperator):
    r"""Resource class for the PhaseShift gate.

    Args:
        precision (float, optional): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires, optional): The wires the operation acts on.

    Resources:
        The phase shift gate is equivalent to a Z-rotation upto some global phase,
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

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""

        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
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
        global_phase = resource_rep(GlobalPhase)
        return [GateCount(rz), GateCount(global_phase)]

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a phase shift operator just changes the sign of the phase, thus
            the resources of the adjoint operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision=precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        precision=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator."""
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a phase shift produces a sum of shifts.
            The resources simplify to just one total phase shift operator.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision=precision))]


class RX(ResourceOperator):
    r"""Resource class for the RX gate.

    Args:
        precision (float, optional): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires, optional): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RX`.

    **Example**

    The resources for this operation are computed as:

    >>> qml.estimator.RX.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""

        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Keyword Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        precision=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator."""
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]


class RY(ResourceOperator):
    r"""Resource class for the RY gate.

    Args:
        precision (float, optional): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires, optional): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RY`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.RY.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Keyword Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        precision=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator."""
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]


class RZ(ResourceOperator):
    r"""Resource class for the RZ gate.

    Args:
        precision (float, optional): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires, optional): The wires the operation acts on.

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.RZ`.

    **Example**

    The resources for this operation are computed using:

    >>> qml.estimator.RZ.resource_decomp(precision=1e-4)
    [(24 x T)]
    """

    num_wires = 1
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

        Args:
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        precision=None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator."""
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]


class Rot(ResourceOperator):
    r"""Resource class for the Rot gate.

    Args:
        precision (float, optional): The error threshold for the Clifford + T decomposition
            of this operation. The default value is ``None`` which corresponds to using the
            ``precision`` stated in the ``ResourceConfig``.
        wires (Any or Wires, optional): The wires the operation acts on.

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

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * precision (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources."""
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
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
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general single qubit rotation changes the sign of the rotation angles,
            thus the resources of the adjoint operation are same as the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, precision=None, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator."""
        raise ResourcesUndefinedError

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a general single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]
