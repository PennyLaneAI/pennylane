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
r"""Resource operators for parametric multi qubit operations."""

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)

# pylint: disable=arguments-differ


class ResourceMultiRZ(ResourceOperator):
    r"""Resource class for the MultiRZ gate.

    Args:
        num_wires (int): the number of qubits the operation acts upon
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources come from Section VIII (Figure 3) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper.

        Specifically, the resources are given by one :class:`~.ResourceRZ` gate and a cascade of
        :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
        the gate acts on.

    .. seealso:: :class:`~.MultiRZ`

    **Example**

    The resources for this operation are computed using:

    >>> multi_rz = plre.ResourceMultiRZ(num_wires=3)
    >>> gate_set = {"CNOT", "RZ"}
    >>>
    >>> print(plre.estimate_resources(multi_rz, gate_set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 5
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'CNOT': 4, 'RZ': 1}

    """

    resource_keys = {"num_wires", "eps"}

    def __init__(self, num_wires, eps=None, wires=None) -> None:
        self.num_wires = num_wires
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, num_wires, eps=None, **kwargs):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            eps (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources come from Section VIII (Figure 3) of `The Bravyi-Kitaev transformation for
            quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper.

            Specifically, the resources are given by one :class:`~.ResourceRZ` gate and a cascade of
            :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
            the gate acts on.
        """
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep(eps=eps)

        return [GateCount(cnot, 2 * (num_wires - 1)), GateCount(rz)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * num_wires (int): the number of qubits the operation acts upon
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"num_wires": self.num_wires, "eps": self.eps}

    @classmethod
    def resource_rep(cls, num_wires, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            eps (float): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"num_wires": num_wires, "eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, num_wires, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            eps (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(num_wires=num_wires, eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        num_wires,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            num_wires (int): the number of qubits the base operation acts upon
            eps (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a cascade of
            :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
            the gate acts on.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = re.resource_rep(re.ResourceCNOT)
        ctrl_rz = re.resource_rep(
            re.ResourceControlled,
            {
                "base_cmpr_op": re.resource_rep(re.ResourceRZ, {"eps": eps}),
                "num_ctrl_wires": ctrl_num_ctrl_wires,
                "num_ctrl_values": ctrl_num_ctrl_values,
            },
        )

        return [GateCount(cnot, 2 * (num_wires - 1)), GateCount(ctrl_rz)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, num_wires, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            num_wires (int): the number of qubits the base operation acts upon
            eps (float): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a general rotation produces a sum of rotations.
            The resources simplify to just one total multi-RZ rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(num_wires=num_wires, eps=eps))]


class ResourcePauliRot(ResourceOperator):
    r"""Resource class for the PauliRot gate.

    Args:
        pauli_string (str): a string describing the pauli operators that define the rotation
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
        the cost is the associated single qubit rotation (:code:`RX, RY, RZ, GlobalPhase`).

        The resources come from Section VIII (Figures 3 & 4) of `The Bravyi-Kitaev transformation
        for quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper,
        in combination with the following identity:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
            \end{align}

        Specifically, the resources are given by one :class:`~.ResourceRZ` gate and a cascade of
        :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
        the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
        a pair of :class:`~.ResourceHadamard` gates, and for each :code:`Y` gate in the Pauli word we
        conjugate by a pair of :class:`~.ResourceHadamard` and a pair of :class:`~.ResourceS` gates.

    .. seealso:: :class:`~.PauliRot`

    **Example**

    The resources for this operation are computed using:

    >>> pr = plre.ResourcePauliRot(pauli_string="XYZ")
    >>> print(plre.estimate_resources(pr, plre.StandardGate\
    Set))
    --- Resources: ---
    Total qubits: 3
    Total gates : 11
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 3
    Gate breakdown:
     {'Hadamard': 4, 'S': 1, 'Adjoint(S)': 1, 'RZ': 1, 'CNOT': 4}

    """

    resource_keys = {"pauli_string", "eps"}

    def __init__(self, pauli_string, eps=None, wires=None) -> None:
        self.eps = eps
        self.pauli_string = pauli_string
        self.num_wires = len(pauli_string)
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, pauli_string, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            pauli_string (str): a string describing the pauli operators that define the rotation
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
            the cost is the associated single qubit rotation (:code:`RX, RY, RZ, GlobalPhase`).

            The resources come from Section VIII (Figures 3 & 4) of `The Bravyi-Kitaev transformation
            for quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper,
            in combination with the following identity:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
                \end{align}

            Specifically, the resources are given by one :class:`~.ResourceRZ` gate and a cascade of
            :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
            the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
            a pair of :class:`~.ResourceHadamard` gates, and for each :code:`Y` gate in the Pauli word we
            conjugate by a pair of :class:`~.ResourceHadamard` and a pair of :class:`~.ResourceS` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            gp = re.resource_rep(re.ResourceGlobalPhase)
            return [GateCount(gp)]

        if pauli_string == "X":
            return [GateCount(re.resource_rep(re.ResourceRX, {"eps": eps}))]
        if pauli_string == "Y":
            return [GateCount(re.resource_rep(re.ResourceRY, {"eps": eps}))]
        if pauli_string == "Z":
            return [GateCount(re.resource_rep(re.ResourceRZ, {"eps": eps}))]

        active_wires = len(pauli_string.replace("I", ""))

        h = re.resource_rep(re.ResourceHadamard)
        s = re.resource_rep(re.ResourceS)
        rz = re.resource_rep(re.ResourceRZ, {"eps": eps})
        s_dagg = re.resource_rep(
            re.ResourceAdjoint,
            {"base_cmpr_op": re.resource_rep(re.ResourceS)},
        )
        cnot = re.resource_rep(re.ResourceCNOT)

        h_count = 0
        s_count = 0

        for gate in pauli_string:
            if gate == "X":
                h_count += 1
            if gate == "Y":
                h_count += 1
                s_count += 1

        gate_types = []
        if h_count:
            gate_types.append(GateCount(h, 2 * h_count))

        if s_count:
            gate_types.append(GateCount(s, s_count))
            gate_types.append(GateCount(s_dagg, s_count))

        gate_types.append(GateCount(rz))
        gate_types.append(GateCount(cnot, 2 * (active_wires - 1)))

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * pauli_string (str): a string describing the pauli operators that define the rotation
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {
            "pauli_string": self.pauli_string,
            "eps": self.eps,
        }

    @classmethod
    def resource_rep(cls, pauli_string, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            pauli_string (str): a string describing the pauli operators that define the rotation
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"pauli_string": pauli_string, "eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, pauli_string, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            pauli_string (str): a string describing the pauli operators that define the rotation
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(pauli_string=pauli_string, eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        pauli_string,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            pauli_string (str): a string describing the pauli operators that define the rotation
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
            the cost is the associated controlled single qubit rotation gate: (:class:`~.ResourceCRX`,
            :class:`~.ResourceCRY`, :class:`~.ResourceCRZ`, controlled-:class:`~.ResourceGlobalPhase`).

            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a cascade of
            :math:`2 * (n - 1)` :class:`~.ResourceCNOT` gates where :math:`n` is the number of qubits
            the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
            a pair of :class:`~.ResourceHadamard` gates, and for each :code:`Y` gate in the Pauli word
            we conjugate by a pair of :class:`~.ResourceHadamard` and a pair of :class:`~.ResourceS` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            ctrl_gp = re.ResourceControlled.resource_rep(
                re.resource_rep(re.ResourceGlobalPhase),
                ctrl_num_ctrl_wires,
                ctrl_num_ctrl_values,
            )
            return [GateCount(ctrl_gp)]

        if pauli_string == "X":
            return [
                GateCount(
                    re.ResourceControlled.resource_rep(
                        re.resource_rep(re.ResourceRX, {"eps": eps}),
                        ctrl_num_ctrl_wires,
                        ctrl_num_ctrl_values,
                    )
                )
            ]
        if pauli_string == "Y":
            return [
                GateCount(
                    re.ResourceControlled.resource_rep(
                        re.resource_rep(re.ResourceRY, {"eps": eps}),
                        ctrl_num_ctrl_wires,
                        ctrl_num_ctrl_values,
                    )
                )
            ]
        if pauli_string == "Z":
            return [
                GateCount(
                    re.ResourceControlled.resource_rep(
                        re.resource_rep(re.ResourceRZ, {"eps": eps}),
                        ctrl_num_ctrl_wires,
                        ctrl_num_ctrl_values,
                    )
                )
            ]

        active_wires = len(pauli_string.replace("I", ""))

        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()
        crz = re.ResourceControlled.resource_rep(
            re.resource_rep(re.ResourceRZ, {"eps": eps}),
            ctrl_num_ctrl_wires,
            ctrl_num_ctrl_values,
        )
        s_dagg = re.resource_rep(
            re.ResourceAdjoint,
            {"base_cmpr_op": re.resource_rep(re.ResourceS)},
        )
        cnot = re.ResourceCNOT.resource_rep()

        h_count = 0
        s_count = 0

        for gate in pauli_string:
            if gate == "X":
                h_count += 1
            if gate == "Y":
                h_count += 1
                s_count += 1

        gate_types = []
        if h_count:
            gate_types.append(GateCount(h, 2 * h_count))

        if s_count:
            gate_types.append(GateCount(s, s_count))
            gate_types.append(GateCount(s_dagg, s_count))

        gate_types.append(GateCount(crz))
        gate_types.append(GateCount(cnot, 2 * (active_wires - 1)))

        return gate_types

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, pauli_string, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            pauli_string (str): a string describing the pauli operators that define the rotation
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a general rotation produces a sum of rotations.
            The resources simplify to just one total pauli rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(pauli_string=pauli_string, eps=eps))]


class ResourceIsingXX(ResourceOperator):
    r"""Resource class for the IsingXX gate.

    Args:
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        Ising XX coupling gate

        .. math:: XX(\phi) = \exp\left(-i \frac{\phi}{2} (X \otimes X)\right) =
            \begin{bmatrix} =
                \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
                0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
                0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
            \end{bmatrix}.

        The circuit implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭●─────RX────╭●─┤
            1: ─╰X───────────╰X─┤

    .. seealso:: :class:`~.IsingXX`

    **Example**

    The resources for this operation are computed using:

    >>> ising_xx = plre.ResourceIsingXX()
    >>> gate_set = {"CNOT", "RX"}
    >>> print(plre.estimate_resources(ising_xx, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'CNOT': 2, 'RX': 1}

    """

    num_wires = 2
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Ising XX coupling gate

            .. math:: XX(\phi) = \exp\left(-i \frac{\phi}{2} (X \otimes X)\right) =
                \begin{bmatrix} =
                    \cos(\phi / 2) & 0 & 0 & -i \sin(\phi / 2) \\
                    0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
                    0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                    -i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
                \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ─╭●─────RX────╭●─┤
                1: ─╰X───────────╰X─┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = re.ResourceCNOT.resource_rep()
        rx = re.ResourceRX.resource_rep(eps=eps)
        return [GateCount(cnot, 2), GateCount(rx)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RX-gate and a pair of
            :class:`~.ResourceCNOT` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        cnot = re.ResourceCNOT.resource_rep()
        ctrl_rx = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRX.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(cnot, 2), GateCount(ctrl_rx)]

    @classmethod
    def default_pow_resource_decomp(
        cls,
        pow_z,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]


class ResourceIsingYY(ResourceOperator):
    r"""Resource class for the IsingYY gate.

    Args:
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        Ising YY coupling gate

        .. math:: \mathtt{YY}(\phi) = \exp\left(-i \frac{\phi}{2} (Y \otimes Y)\right) =
            \begin{bmatrix}
                \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
                0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
                0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
            \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭●─────RY────╭●─┤
            1: ─╰Y───────────╰Y─┤

    .. seealso:: :class:`~.IsingYY`

    **Example**

    The resources for this operation are computed using:

    >>> ising_yy = plre.ResourceIsingYY()
    >>> gate_set = {"CY", "RY"}
    >>> print(plre.estimate_resources(ising_yy, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'CY': 2, 'RY': 1}

    """

    num_wires = 2
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Ising YY coupling gate

            .. math:: \mathtt{YY}(\phi) = \exp\left(-i \frac{\phi}{2} (Y \otimes Y)\right) =
                \begin{bmatrix}
                    \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
                    0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
                    0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                    i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
                \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ─╭●─────RY────╭●─┤
                1: ─╰Y───────────╰Y─┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cy = re.ops.ResourceCY.resource_rep()
        ry = re.ops.ResourceRY.resource_rep(eps=eps)
        return [GateCount(cy, 2), GateCount(ry)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RY-gate and a pair of
            :class:`~.ResourceCY` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cy = re.ops.ResourceCY.resource_rep()
        ctrl_ry = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRY.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(cy, 2), GateCount(ctrl_ry)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]


class ResourceIsingXY(ResourceOperator):
    r"""Resource class for the IsingXY gate.

    Args:
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        Ising (XX + YY) coupling gate

        .. math:: \mathtt{XY}(\phi) = \exp\left(i \frac{\theta}{4} (X \otimes X + Y \otimes Y)\right) =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
                0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ──H─╭●─────RY────╭●──H─┤
            1: ────╰Y─────RX────╰Y────┤

    .. seealso:: :class:`~.IsingXY`

    **Example**

    The resources for this operation are computed using:

    >>> ising_xy = plre.ResourceIsingXY()
    >>> gate_set = {"Hadamard", "CY", "RY", "RX"}
    >>> print(plre.estimate_resources(ising_xy, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 6
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'Hadamard': 2, 'CY': 2, 'RY': 1, 'RX': 1}

    """

    num_wires = 2
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            IsingXY coupling gate

            .. math:: \mathtt{XY}(\phi) = \exp\left(i \frac{\theta}{4} (X \otimes X + Y \otimes Y)\right) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
                    0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ──H─╭●─────RY────╭●──H─┤
                1: ────╰Y─────RX────╰Y────┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = re.ResourceHadamard.resource_rep()
        cy = re.ResourceCY.resource_rep()
        ry = re.ResourceRY.resource_rep(eps=eps)
        rx = re.ResourceRX.resource_rep(eps=eps)

        return [GateCount(h, 2), GateCount(cy, 2), GateCount(ry), GateCount(rx)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RY-gate, one multi-controlled RX-gate,
            a pair of :class:`~.ResourceCY` gates and a pair of :class:`~.ResourceHadamard` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        h = re.ResourceHadamard.resource_rep()
        cy = re.ResourceCY.resource_rep()
        ctrl_rx = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRX.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )
        ctrl_ry = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRY.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(h, 2), GateCount(cy, 2), GateCount(ctrl_rx), GateCount(ctrl_ry)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]


class ResourceIsingZZ(ResourceOperator):
    r"""Resource class for the IsingZZ gate.

    Args:
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        Ising ZZ coupling gate

        .. math:: ZZ(\phi) = \exp\left(-i \frac{\phi}{2} (Z \otimes Z)\right) =
            \begin{bmatrix}
                e^{-i \phi / 2} & 0 & 0 & 0 \\
                0 & e^{i \phi / 2} & 0 & 0 \\
                0 & 0 & e^{i \phi / 2} & 0 \\
                0 & 0 & 0 & e^{-i \phi / 2}
            \end{bmatrix}.

        The cost for implmenting this transformation is given by:

        .. code-block:: bash

            0: ─╭●───────────╭●─┤
            1: ─╰X─────RZ────╰X─┤

    .. seealso:: :class:`~.IsingZZ`

    **Example**

    The resources for this operation are computed using:

    >>> ising_zz = plre.ResourceIsingZZ()
    >>> gate_set = {"CNOT", "RZ"}
    >>> print(plre.estimate_resources(ising_zz, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'CNOT': 2, 'RZ': 1}

    """

    num_wires = 2
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Ising ZZ coupling gate

            .. math:: ZZ(\phi) = \exp\left(-i \frac{\phi}{2} (Z \otimes Z)\right) =
                \begin{bmatrix}
                    e^{-i \phi / 2} & 0 & 0 & 0 \\
                    0 & e^{i \phi / 2} & 0 & 0 \\
                    0 & 0 & e^{i \phi / 2} & 0 \\
                    0 & 0 & 0 & e^{-i \phi / 2}
                \end{bmatrix}.

            The cost for implmenting this transformation is given by:

            .. code-block:: bash

                0: ─╭●───────────╭●─┤
                1: ─╰X─────RZ────╰X─┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep(eps=eps)

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rz] = 1

        return [GateCount(cnot, 2), GateCount(rz)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a pair of
            :class:`~.ResourceCNOT` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = re.ResourceCNOT.resource_rep()
        ctrl_rz = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRZ.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )
        return [GateCount(cnot, 2), GateCount(ctrl_rz)]

    @classmethod
    def default_pow_resource_decomp(cls, pow_z, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(eps=eps))]


class ResourcePSWAP(ResourceOperator):
    r"""Resource class for the PSWAP gate.

    Args:
        eps (float, optional): error threshold for clifford plus T decomposition of this operation
        wires (Sequence[int], optional): the wire the operation acts on

    Resources:
        The :code:`PSWAP` gate is defined as:

        .. math:: PSWAP(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & e^{i \phi} & 0 \\
                0 & e^{i \phi} & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭SWAP─╭●───────────╭●─┤
            1: ─╰SWAP─╰X─────Rϕ────╰X─┤

    .. seealso:: :class:`~.PSWAP`

    **Example**

    The resources for this operation are computed using:

    >>> pswap = plre.ResourcePSWAP()
    >>> gate_set = {"CNOT", "SWAP", "PhaseShift"}
    >>> print(plre.estimate_resources(pswap, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 4
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'SWAP': 1, 'PhaseShift': 1, 'CNOT': 2}

    """

    num_wires = 2
    resource_keys = {"eps"}

    def __init__(self, eps=None, wires=None) -> None:
        self.eps = eps
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(cls, eps=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The :code:`PSWAP` gate is defined as:

            .. math:: PSWAP(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & e^{i \phi} & 0 \\
                    0 & e^{i \phi} & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ─╭SWAP─╭●───────────╭●─┤
                1: ─╰SWAP─╰X─────Rϕ────╰X─┤

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        swap = re.ResourceSWAP.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        phase = re.ResourcePhaseShift.resource_rep(eps=eps)

        return [GateCount(swap), GateCount(phase), GateCount(cnot, 2)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * eps (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"eps": self.eps}

    @classmethod
    def resource_rep(cls, eps=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, {"eps": eps})

    @classmethod
    def default_adjoint_resource_decomp(cls, eps=None) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

            Resources:
                The adjoint of this operator just changes the sign of the phase angle, thus
                the resources of the adjoint operation results in the original operation.

            Returns:
                list[GateCount]: A list of GateCount objects, where each object
                represents a specific quantum gate and the number of times it appears
                in the decomposition.

        """
        return [GateCount(cls.resource_rep(eps=eps))]

    @classmethod
    def default_controlled_resource_decomp(
        cls,
        ctrl_num_ctrl_wires,
        ctrl_num_ctrl_values,
        eps=None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            ctrl_num_ctrl_wires (int): the number of qubits the operation is controlled on
            ctrl_num_ctrl_values (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            eps (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled phase shift gate, one multi-controlled
            SWAP gate and a pair of :class:`~.ResourceCNOT` gates.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = re.ResourceCNOT.resource_rep()
        ctrl_swap = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceSWAP.resource_rep(),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )
        ctrl_ps = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourcePhaseShift.resource_rep(eps=eps),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(ctrl_swap), GateCount(cnot, 2), GateCount(ctrl_ps)]
