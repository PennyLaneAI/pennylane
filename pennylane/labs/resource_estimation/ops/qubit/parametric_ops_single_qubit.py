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

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ


def _rotation_resources(epsilon=10e-3):
    r"""An estimate on the number of T gates needed to implement a Pauli rotation.

    The expected T-count is taken from (the 'Simulation Results' section) `Efficient
    Synthesis of Universal Repeat-Until-Success Circuits <https://arxiv.org/abs/1404.5320>`_.
    The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    Args:
        epsilon (float): the acceptable error threshold for the approximation

    Returns:
        list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
    """
    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = resource_rep(re.ResourceT)
    return [GateCount(t, num_gates)]


class ResourcePhaseShift(ResourceOperator):
    r"""Resource class for the PhaseShift gate.

    Keyword Args:
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        The phase shift gate is equivalent to a Z-rotation upto some global phase,
        as defined from the following identity:

        .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\phi}
                \end{bmatrix}.

    .. seealso:: :class:`~.PhaseShift`

    **Example**

    The resources for this operation are computed as:

    >>> re.ResourcePhaseShift.resources()
    {RZ: 1, GlobalPhase: 1}
    """

    num_wires = 1
    resource_keys = {"eps"}

    def __init__(self, epsilon=None, wires=None) -> None:
        self.eps = epsilon
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
        the Operator that are needed to compute a resource estimation."""

        params = {"eps": eps} if eps is not None else {}
        return CompressedResourceOp(cls, params)        

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The phase shift gate is equivalent to a Z-rotation upto some global phase,
            as defined from the following identity:

            .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\phi}
                    \end{bmatrix}.
        """
        rz = resource_rep(ResourceRZ, {"eps": eps})
        global_phase = resource_rep(re.ResourceGlobalPhase)
        return [GateCount(rz), GateCount(global_phase)]

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a phase shift operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a phase shift produces a sum of shifts.
            The resources simplify to just one total phase shift operator.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep(eps=eps), 1)]
        return [GateCount(cls.resource_rep(eps=eps))]


class ResourceRX(ResourceOperator):
    r"""Resource class for the RX gate.

    Keyword Args:
        eps (float): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RX`

    **Example**

    The resources for this operation are computed as:

    >>> re.estimate_resources(re.ResourceRX)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rx": 1e-3}
    >>> re.estimate_resources(re.ResourceRX, config=config)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}
    """

    num_wires = 1
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * eps (Union[float, None]): the number of qubits the operation is controlled on
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""

        params = {"eps": eps} if eps is not None else {}
        return CompressedResourceOp(cls, params)        

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Keyword Args:
            eps (float): the error threshold

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        """
        eps = eps or kwargs["config"]["error_rx"]
        return _rotation_resources(epsilon=eps)

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

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
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep(eps=eps), 1)]

        return [GateCount(cls.resource_rep(eps))]


class ResourceRY(ResourceOperator):
    r"""Resource class for the RY gate.

    Keyword Args:
        eps (float): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RY`

    **Example**

    The resources for this operation are computed using:

    >>> re.estimate_resources(re.ResourceRY)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_ry": 1e-3}
    >>> re.estimate_resources(re.ResourceRY, config=config)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}

    """

    num_wires = 1
    resource_keys = {"eps"}

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
        the Operator that are needed to compute a resource estimation."""

        params = {"eps": eps} if eps is not None else {}
        return CompressedResourceOp(cls, params)        

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Keyword Args:
            eps (float): the error threshold

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        Args:
            config (dict): a dictionary containing the error threshold
        """
        eps = eps or kwargs["config"]["error_ry"]
        return _rotation_resources(epsilon=eps)

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

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
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep(eps=eps), 1)]

        return [GateCount(cls.resource_rep(eps))]


class ResourceRZ(ResourceOperator):
    r"""Resource class for the RZ gate.

    Keyword Args:
        eps (float): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int] or int): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

    .. seealso:: :class:`~.RZ`

    **Example**

    The resources for this operation are computed using:

    >>> re.estimate_resources(re.ResourceRZ)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rz": 1e-3}
    >>> re.estimate_resources(re.ResourceRZ, config=config)
    --- Resources: ---
    Total qubits: 1
    Total gates : 1
    Qubit breakdown:
    clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
    Gate breakdown:
    {'T': 17}

    """

    num_wires = 1
    resource_keys = {"eps"}

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
        the Operator that are needed to compute a resource estimation."""

        params = {"eps": eps} if eps is not None else {}
        return CompressedResourceOp(cls, params)        
        
    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \ceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)

        Args:
            config (dict): a dictionary containing the error threshold
        """
        eps = eps or kwargs["config"]["error_rz"]
        return _rotation_resources(epsilon=eps)

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

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
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep(eps=eps), 1)]

        return [GateCount(cls.resource_rep(eps))]


class ResourceRot(ResourceOperator):
    r"""Resource class for the Rot-gate.

    Args:
        wires (Any, Wires): the wire the operation acts on

    Resources:
        The resources are obtained according to the definition of the :class:`Rot` gate:

        .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

    .. seealso:: :class:`~.Rot`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceRot.resource_decomp()
    {RY: 1, RZ: 2}
    """

    num_wires = 1
    resource_keys = {"eps"}

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
        the Operator that are needed to compute a resource estimation."""

        params = {"eps": eps} if eps is not None else {}
        return CompressedResourceOp(cls, params)        

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs) -> list[GateCount]:
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained according to the definition of the :class:`Rot` gate:

            .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

        """
        ry = resource_rep(ResourceRY, {"eps": eps})
        rz = resource_rep(ResourceRZ, {"eps": eps})

        return [GateCount(ry), GateCount(rz, 2)]

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general single qubit rotation changes the sign of the rotation angles,
            thus the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps))]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a dictionary representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to

        Resources:
            Taking arbitrary powers of a general single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.
        """
        if pow_z == 0:
            return [GateCount(re.ResourceIdentity.resource_rep(eps=eps), 1)]

        return [GateCount(cls.resource_rep(eps))]
