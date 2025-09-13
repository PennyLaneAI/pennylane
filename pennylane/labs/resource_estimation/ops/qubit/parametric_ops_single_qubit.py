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

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ


def _rotation_resources(precision=10e-3):
    r"""An estimate on the number of T gates needed to implement a Pauli rotation.

    The expected T-count is taken from (the 'Simulation Results' section) `Efficient
    Synthesis of Universal Repeat-Until-Success Circuits <https://arxiv.org/abs/1404.5320>`_.
    The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    Args:
        precision (float): the acceptable error threshold for the approximation

    Returns:
        list[GateCount]: A list of GateCount objects, where each object
        represents a specific quantum gate and the number of times it appears
        in the decomposition.
    """
    num_gates = round(1.149 * np.log2(1 / precision) + 9.2)
    t = resource_rep(plre.ResourceT)
    return [GateCount(t, num_gates)]


class ResourcePhaseShift(ResourceOperator):
    r"""Resource class for the PhaseShift gate.

    Keyword Args:
        precision (float, optional): The error threshold for clifford plus T decomposition of this operation.
            The default value is `None` which corresponds to using the precision stated in the config.
        wires (Any or Wires, optional): the wire the operation acts on

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

    >>> plre.ResourcePhaseShift.resource_decomp()
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
            A dictionary containing the resource parameters:
                * precision (Union[float, None]): error threshold for the approximation
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
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            The phase shift gate is equivalent to a Z-rotation upto some global phase,
            as defined from the following identity:

            .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                        1 & 0 \\
                        0 & e^{i\phi}
                    \end{bmatrix}.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rz = resource_rep(ResourceRZ, {"precision": precision})
        global_phase = resource_rep(plre.ResourceGlobalPhase)
        return [GateCount(rz), GateCount(global_phase)]

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a phase shift operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
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
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            For a single control wire, the cost is a single instance of
            :class:`~.ResourceControlledPhaseShift`. Two additional :class:`~.ResourceX` gates are used
            to flip the control qubit if it is zero-controlled.

            In the case where multiple controlled wires are provided, we can collapse the control
            wires by introducing one 'clean' auxilliary qubit (which gets reset at the end).
            In this case the cost increases by two additional :class:`~.ResourceMultiControlledX` gates,
            as described in (lemma 7.11) `Elementary gates for quantum computation <https://arxiv.org/pdf/quant-ph/9503016>`_.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [
                GateCount(resource_rep(plre.ResourceControlledPhaseShift, {"precision": precision}))
            ]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        c_ps = resource_rep(plre.ResourceControlledPhaseShift, {"precision": precision})
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [AllocWires(1), GateCount(c_ps), GateCount(mcx, 2), FreeWires(1)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
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
        return [GateCount(cls.resource_rep(precision=precision))]


class ResourceRX(ResourceOperator):
    r"""Resource class for the RX gate.

    Keyword Args:
        precision (float): error threshold for clifford plus T decomposition of this operation
        wires (Any, Wires, optional): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: :class:`~.RX`

    **Example**

    The resources for this operation are computed as:

    >>> op = plre.estimate(plre.ResourceRX)()
    >>> print(op)
    --- Resources: ---
     Total qubits: 1
     Total gates : 21
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 21}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rx": 1e-4}
    >>> op = plre.estimate(plre.ResourceRX, config=config)()
    >>> print(op)
    --- Resources: ---
     Total qubits: 1
     Total gates : 24
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 24}
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
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
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
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRX`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are taken
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. In combination with the following identity:

            .. math:: \hat{RX} = \hat{H} \cdot \hat{RZ}  \cdot \hat{H},

            we can express the :code:`CRX` gate as a :code:`CRZ` gate conjugated by :code:`Hadamard`
            gates. The expression for controlled-RZ gates is used as defined in the reference above.
            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of that identity.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:

            gate_types = [GateCount(resource_rep(plre.ResourceCRX, {"precision": precision}))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        h = resource_rep(plre.ResourceHadamard)
        rz = resource_rep(ResourceRZ, {"precision": precision})
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )
        return [GateCount(h, 2), GateCount(rz, 2), GateCount(mcx, 2)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

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
        return [GateCount(cls.resource_rep(precision))]


class ResourceRY(ResourceOperator):
    r"""Resource class for the RY gate.

    Keyword Args:
        precision (float): error threshold for clifford plus T decomposition of this operation
        wires (Any, Wires, optional): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: :class:`~.RY`

    **Example**

    The resources for this operation are computed using:

    >>> op = plre.estimate(plre.ResourceRY)()
    >>> print(op)
    --- Resources: ---
     Total qubits: 1
     Total gates : 1
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 21}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_ry": 1e-4}
    >>> op = plre.estimate(plre.ResourceRY, config=config)()
    --- Resources: ---
     Total qubits: 1
     Total gates : 24
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 24}
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
            A dictionary containing the resource parameters:
                * precision (Union[float, None]): error threshold for the approximation
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
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
            resources are approximating the gate with a series of T gates. The expected T-count is taken
            from (the 'Simulation Results' section) `Efficient Synthesis of Universal Repeat-Until-Success
            Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

            .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
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
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRY`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are taken
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit
            unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the
            following identity:

            .. math:: \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of this identity. Thus we are able to constructively or destructively
            interfere the gates based on the value of the control qubits.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(plre.ResourceCRY, {"precision": precision}))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        ry = resource_rep(ResourceRY, {"precision": precision})
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        return [GateCount(ry, 2), GateCount(mcx, 2)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a single qubit rotation produces a sum of rotations.
            The resources simplify to just one total single qubit rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]


class ResourceRZ(ResourceOperator):
    r"""Resource class for the RZ gate.

    Keyword Args:
        precision (float): error threshold for clifford plus T decomposition of this operation
        wires (Any, Wires, optional): the wire the operation acts on

    Resources:
        A single qubit rotation gate can be approximately synthesised from Clifford and T gates. The
        resources are approximating the gate with a series of T gates. The expected T-count is taken
        from (the 'Simulation Results' section) `Eﬃcient Synthesis of Universal Repeat-Until-Success
        Circuits <https://arxiv.org/abs/1404.5320>`_. The cost is given as:

        .. math:: T_{count} = \lceil(1.149 * log_{2}(\frac{1}{\epsilon}) + 9.2)\rceil

    .. seealso:: :class:`~.RZ`

    **Example**

    The resources for this operation are computed using:

    >>> op = plre.estimate(plre.ResourceRZ)()
    >>> op
    --- Resources: ---
     Total qubits: 1
     Total gates : 21
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 21}

    The operation does not require any parameters directly, however, it will depend on the single
    qubit error threshold, which can be set using a config dictionary.

    >>> config = {"error_rz": 1e-4}
    >>> op = plre.estimate(plre.ResourceRZ, config=config)()
    >>> print(op)
    --- Resources: ---
     Total qubits: 1
     Total gates : 24
     Qubit breakdown:
      clean qubits: 0, dirty qubits: 0, algorithmic qubits: 1
     Gate breakdown:
      {'T': 24}
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
            A dictionary containing the resource parameters:
                * precision (Union[float, None]): error threshold for the approximation
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
            precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return _rotation_resources(precision=precision)

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a single qubit rotation changes the sign of the rotation angle,
            thus the resources of the adjoint operation result in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
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
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRY`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.

            In the case where multiple controlled wires are provided, the resources are obtained
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. They are derived from the following identity:

            .. math:: \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}.

            By replacing the :code:`X` gates with multi-controlled :code:`X` gates, we obtain a
            controlled-version of this identity. Thus we are able to constructively or destructively
            interfere the gates based on the value of the control qubits.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(plre.ResourceCRZ, {"precision": precision}))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        rz = resource_rep(ResourceRZ, {"precision": precision})
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        return [GateCount(rz, 2), GateCount(mcx, 2)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

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
        return [GateCount(cls.resource_rep(precision))]


class ResourceRot(ResourceOperator):
    r"""Resource class for the Rot-gate.

    Args:
        precision (float): error threshold for clifford plus T decomposition of this operation
        wires (Any, Wires, optional): the wire the operation acts on

    Resources:
        The resources are obtained according to the definition of the :class:`Rot` gate:

        .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

    .. seealso:: :class:`~.Rot`

    **Example**

    The resources for this operation are computed using:

    >>> plre.ResourceRot.resource_decomp()
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
            A dictionary containing the resource parameters:
                * precision (Union[float, None]): error threshold for the approximation
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
            The resources are obtained according to the definition of the :class:`Rot` gate:

            .. math:: \hat{R}(\omega, \theta, \phi) = \hat{RZ}(\omega) \cdot \hat{RY}(\theta) \cdot \hat{RZ}(\phi).

        """
        ry = resource_rep(ResourceRY, {"precision": precision})
        rz = resource_rep(ResourceRZ, {"precision": precision})

        return [GateCount(ry), GateCount(rz, 2)]

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Resources:
            The adjoint of a general single qubit rotation changes the sign of the rotation angles,
            thus the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision))]

    @classmethod
    def controlled_resource_decomp(
        cls, ctrl_num_ctrl_wires, ctrl_num_ctrl_values, precision=None, **kwargs
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            precision (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            For a single control wire, the cost is a single instance of :class:`~.ResourceCRot`.
            Two additional :class:`~.ResourceX` gates are used to flip the control qubit if
            it is zero-controlled.
            In the case where multiple controlled wires are provided, the resources are derived
            from Figure 1b of the paper `T-count and T-depth of any multi-qubit unitary
            <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the following
            identities:

            .. math::

                \begin{align}
                    \hat{RZ}(\theta) = \hat{X} \cdot \hat{RZ}(- \theta) \cdot \hat{X}, \\
                    \hat{RY}(\theta) = \hat{X} \cdot \hat{RY}(- \theta) \cdot \hat{X}.
                \end{align}

            This identity is applied along with some clever choices for the angle values to combine
            rotations; the final circuit takes the form:

            .. code-block:: bash

                ctrl: ─────╭●─────────╭●─────────┤
                trgt: ──RZ─╰X──RZ──RY─╰X──RY──RZ─┤

            The :code:`CNOT` gates are replaced with multi-controlled X-gates to generalize to the
            multi-controlled case.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if ctrl_num_ctrl_wires == 1:
            gate_types = [GateCount(resource_rep(plre.ResourceCRot, {"precision": precision}))]

            if ctrl_num_ctrl_values:
                gate_types.append(GateCount(resource_rep(plre.ResourceX), 2))

            return gate_types

        rz = resource_rep(ResourceRZ, {"precision": precision})
        ry = resource_rep(ResourceRY, {"precision": precision})
        mcx = resource_rep(
            plre.ResourceMultiControlledX,
            {
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        return [GateCount(mcx, 2), GateCount(rz, 3), GateCount(ry, 2)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

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
        return [GateCount(cls.resource_rep(precision))]
