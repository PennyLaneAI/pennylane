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

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, ResourceOperator
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ, signature-differs


class MultiRZ(ResourceOperator):
    r"""Resource class for the MultiRZ gate.

    Args:
        num_wires (int): the number of wires the operation acts upon
        precision (float | None): error threshold for Clifford + T decomposition of this operation
        wires (Sequence[int] | None): the wires the operation acts on

    Resources:
        The resources come from Section VIII (Figure 3) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/abs/1208.5986>`_ paper.

        Specifically, the resources are given by one ``RZ`` gate and a cascade of
        :math:`2 \times (n - 1)` ``CNOT`` gates where :math:`n` is the number of qubits
        the gate acts on.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.MultiRZ`.

    **Example**

    The resources for this operation are computed using:

    >>> from pennylane import estimator as qre
    >>> multi_rz = qre.MultiRZ(num_wires=3)
    >>> gate_set = {"CNOT", "RZ"}
    >>>
    >>> print(qml.estimator.estimate(multi_rz, gate_set))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 5
      'RZ': 1,
      'CNOT': 4

    """

    resource_keys = {"num_wires", "precision"}

    def __init__(
        self, num_wires: int, precision: float | None = None, wires: WiresLike = None
    ) -> None:
        self.num_wires = num_wires
        self.precision = precision
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, num_wires: int, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (float): error threshold for Clifford + T decomposition of this operation

        Resources:
            The resources come from Section VIII (Figure 3) of `The Bravyi-Kitaev transformation for
            quantum computation of electronic structure <https://arxiv.org/abs/1208.5986>`_ paper.

            Specifically, the resources are given by one ``RZ`` gate and a cascade of
            :math:`2 \times (n - 1)` ``CNOT`` gates where :math:`n` is the number of qubits
            the gate acts on.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = qre.CNOT.resource_rep()
        rz = qre.RZ.resource_rep(precision=precision)

        return [GateCount(cnot, 2 * (num_wires - 1)), GateCount(rz)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of qubits the operation acts upon
                * precision (float): error threshold for Clifford + T decomposition of this operation
        """
        return {"num_wires": self.num_wires, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_wires: int, precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (float): error threshold for Clifford + T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(
            cls, num_wires, {"num_wires": num_wires, "precision": precision}
        )

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator

        Resources:
            The adjoint of this operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_wires = target_resource_params["num_wires"]
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(num_wires=num_wires, precision=precision))]

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
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a cascade of
            :math:`2 * (n - 1)` ``CNOT`` gates where :math:`n` is the number of qubits
            the gate acts on.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_wires = target_resource_params["num_wires"]
        precision = target_resource_params["precision"]
        cnot = qre.resource_rep(qre.CNOT)
        ctrl_rz = qre.resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": qre.resource_rep(qre.RZ, {"precision": precision}),
                "num_ctrl_wires": num_ctrl_wires,
                "num_zero_ctrl": num_zero_ctrl,
            },
        )

        return [GateCount(cnot, 2 * (num_wires - 1)), GateCount(ctrl_rz)]

    @classmethod
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            Taking arbitrary powers of a general rotation produces a sum of rotations.
            The resources simplify to just one total multi-RZ rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        num_wires = target_resource_params["num_wires"]
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(num_wires=num_wires, precision=precision))]


class PauliRot(ResourceOperator):
    r"""Resource class for the PauliRot gate.

    Args:
        pauli_string (str): a string describing the Pauli operators that define the rotation
        precision (float | None): error threshold for Clifford + T decomposition of this operation
        wires (Sequence[int] | None): the wire the operation acts on

    Resources:
        When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
        the cost is the associated single qubit rotation (:code:`RX, RY, RZ, GlobalPhase`).

        The resources come from Section VIII (Figures 3 & 4) of `The Bravyi-Kitaev transformation
        for quantum computation of electronic structure <https://arxiv.org/abs/1208.5986>`_ paper,
        in combination with the following identities:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
            \end{align}

        Specifically, the resources are given by one :code:`RZ` gate and a cascade of
        :math:`2 \times (n - 1)` :code:`CNOT` gates where :math:`n` is the number of qubits
        the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
        a pair of :code:`Hadamard` gates, and for each :code:`Y` gate in the Pauli word we
        conjugate by a pair of :code:`Hadamard` and a pair of :code:`S` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PauliRot`.

    **Example**

    The resources for this operation are computed using:

    >>> from pennylane import estimator as qre
    >>> pr = qre.PauliRot(pauli_string="XYZ")
    >>> print(qre.estimate(pr))
    --- Resources: ---
     Total wires: 3
        algorithmic wires: 3
        allocated wires: 0
             zero state: 0
             any state: 0
     Total gates : 55
      'T': 44,
      'CNOT': 4,
      'Z': 1,
      'S': 2,
      'Hadamard': 4
    """

    resource_keys = {"pauli_string", "precision"}

    def __init__(
        self, pauli_string: str, precision: float | None = None, wires: WiresLike = None
    ) -> None:
        self.precision = precision
        self.pauli_string = pauli_string
        self.num_wires = len(pauli_string)

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, pauli_string: str, precision: float | None = None) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            pauli_string (str): a string describing the pauli operators that define the rotation
            precision (float | None): error threshold for Clifford + T decomposition of this operation

        Resources:
            When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
            the cost is the associated single qubit rotation (:code:`RX, RY, RZ, GlobalPhase`).

            The resources come from Section VIII (Figures 3 & 4) of `The Bravyi-Kitaev transformation
            for quantum computation of electronic structure <https://arxiv.org/abs/1208.5986>`_ paper,
            in combination with the following identity:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
                \end{align}

            Specifically, the resources are given by one :code:`RZ` gate and a cascade of
            :math:`2 \times (n - 1)` :code:`CNOT` gates where :math:`n` is the number of qubits
            the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
            a pair of :code:`Hadamard` gates, and for each :code:`Y` gate in the Pauli word we
            conjugate by a pair of :code:`Hadamard` and a pair of :code:`S` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            gp = qre.resource_rep(qre.GlobalPhase)
            return [GateCount(gp)]

        if pauli_string == "X":
            return [GateCount(qre.resource_rep(qre.RX, {"precision": precision}))]
        if pauli_string == "Y":
            return [GateCount(qre.resource_rep(qre.RY, {"precision": precision}))]
        if pauli_string == "Z":
            return [GateCount(qre.resource_rep(qre.RZ, {"precision": precision}))]

        active_wires = len(pauli_string.replace("I", ""))

        h = qre.resource_rep(qre.Hadamard)
        s = qre.resource_rep(qre.S)
        rz = qre.resource_rep(qre.RZ, {"precision": precision})
        s_dagg = qre.resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": qre.resource_rep(qre.S)},
        )
        cnot = qre.resource_rep(qre.CNOT)

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
                * precision (float): error threshold for Clifford + T decomposition of this operation
        """
        return {
            "pauli_string": self.pauli_string,
            "precision": self.precision,
        }

    @classmethod
    def resource_rep(cls, pauli_string: str, precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            pauli_string (str): a string describing the pauli operators that define the rotation
            precision (float | None): error threshold for Clifford + T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`:: the operator in a compressed representation
        """
        num_wires = len(pauli_string)
        return CompressedResourceOp(
            cls, num_wires, {"pauli_string": pauli_string, "precision": precision}
        )

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator

        Resources:
            The adjoint of this operator just changes the sign of the phase, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        pauli_string = target_resource_params["pauli_string"]
        return [GateCount(cls.resource_rep(pauli_string=pauli_string, precision=precision))]

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
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator

        Resources:
            When the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`)
            the cost is the associated controlled single qubit rotation gate: (:code:`CRX`,
            :code:`CRY`, :code:`CRZ`, controlled-\ :code:`GlobalPhase`).

            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RZ-gate and a cascade of
            :math:`2 \times (n - 1)` :code:`CNOT` gates where :math:`n` is the number of qubits
            the gate acts on. Additionally, for each :code:`X` gate in the Pauli word we conjugate by
            a pair of :code:`Hadamard` gates, and for each :code:`Y` gate in the Pauli word
            we conjugate by a pair of :code:`Hadamard` and a pair of :code:`S` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        pauli_string = target_resource_params["pauli_string"]
        precision = target_resource_params["precision"]

        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            ctrl_gp = qre.Controlled.resource_rep(
                qre.resource_rep(qre.GlobalPhase),
                num_ctrl_wires,
                num_zero_ctrl,
            )
            return [GateCount(ctrl_gp)]

        if pauli_string == "X":
            return [
                GateCount(
                    qre.Controlled.resource_rep(
                        qre.resource_rep(qre.RX, {"precision": precision}),
                        num_ctrl_wires,
                        num_zero_ctrl,
                    )
                )
            ]
        if pauli_string == "Y":
            return [
                GateCount(
                    qre.Controlled.resource_rep(
                        qre.resource_rep(qre.RY, {"precision": precision}),
                        num_ctrl_wires,
                        num_zero_ctrl,
                    )
                )
            ]
        if pauli_string == "Z":
            return [
                GateCount(
                    qre.Controlled.resource_rep(
                        qre.resource_rep(qre.RZ, {"precision": precision}),
                        num_ctrl_wires,
                        num_zero_ctrl,
                    )
                )
            ]

        active_wires = len(pauli_string.replace("I", ""))

        h = qre.Hadamard.resource_rep()
        s = qre.S.resource_rep()
        crz = qre.Controlled.resource_rep(
            qre.resource_rep(qre.RZ, {"precision": precision}),
            num_ctrl_wires,
            num_zero_ctrl,
        )
        s_dagg = qre.resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": qre.resource_rep(qre.S)},
        )
        cnot = qre.CNOT.resource_rep()

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
    def pow_resource_decomp(cls, pow_z: int, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            Taking arbitrary powers of a general rotation produces a sum of rotations.
            The resources simplify to just one total pauli rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        pauli_string = target_resource_params["pauli_string"]
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(pauli_string=pauli_string, precision=precision))]

class ResourceIsingXX(ResourceOperator):
    r"""Resource class for the IsingXX gate.

    Args:
        precision (float | None): error threshold for Clifford+T decomposition of this operation
        wires (WiresLike | None): the wire the operation acts on

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

    >>> from pennylane import estimator as qre
    >>> ising_xx = qre.IsingXX()
    >>> gate_set = {"CNOT", "RX"}
    >>> print(qre.estimate(ising_xx, gate_set))

    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision: float | None = None, wires: WiresLike | None = None) -> None:
        self.precision = precision
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision: float | None = None):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float | None): error threshold for Clifford+T decomposition of this operation

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        cnot = qre.CNOT.resource_rep()
        rx = qre.RX.resource_rep(precision=precision)
        return [GateCount(cnot, 2), GateCount(rx)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float | None): error threshold for Clifford+T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of
                the target operator.

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(precision=precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled RX-gate and a pair of
            :class:`~.pennylane.estimator.ops.CNOT` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        precision = target_resource_params["precision"]
        cnot = qre.CNOT.resource_rep()
        ctrl_rx =  qre.Controlled.resource_rep(
            base_cmpr_op=qre.RX.resource_rep(precision=precision),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )

        return [GateCount(cnot, 2), GateCount(ctrl_rx)]

    @classmethod
    def pow_resource_decomp(
        cls,
        pow_z: int,
        target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params[precision]
        return [GateCount(cls.resource_rep(precision=precision))]


class ResourceIsingYY(ResourceOperator):
    r"""Resource class for the IsingYY gate.

    Args:
        precision (float, optional): error threshold for clifford plus T decomposition of this operation
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
    >>> print(plre.estimate(ising_yy, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'CY': 2, 'RY': 1}

    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
        ry = re.ops.ResourceRY.resource_rep(precision=precision)
        return [GateCount(cy, 2), GateCount(ry)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
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
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
            base_cmpr_op=re.ResourceRY.resource_rep(precision=precision),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(cy, 2), GateCount(ctrl_ry)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision=precision))]


class ResourceIsingXY(ResourceOperator):
    r"""Resource class for the IsingXY gate.

    Args:
        precision (float, optional): error threshold for Clifford+T decomposition of this operation
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
    >>> print(plre.estimate(ising_xy, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 6
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'Hadamard': 2, 'CY': 2, 'RY': 1, 'RX': 1}

    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
        ry = re.ResourceRY.resource_rep(precision=precision)
        rx = re.ResourceRX.resource_rep(precision=precision)

        return [GateCount(h, 2), GateCount(cy, 2), GateCount(ry), GateCount(rx)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
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
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
            base_cmpr_op=re.ResourceRX.resource_rep(precision=precision),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )
        ctrl_ry = re.ResourceControlled.resource_rep(
            base_cmpr_op=re.ResourceRY.resource_rep(precision=precision),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )

        return [GateCount(h, 2), GateCount(cy, 2), GateCount(ctrl_ry), GateCount(ctrl_rx)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision=precision))]


class ResourceIsingZZ(ResourceOperator):
    r"""Resource class for the IsingZZ gate.

    Args:
        precision (float, optional): error threshold for Clifford+T decomposition of this operation
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
    >>> print(plre.estimate(ising_zz, gate_set))
    --- Resources: ---
    Total qubits: 2
    Total gates : 3
    Qubit breakdown:
     clean qubits: 0, dirty qubits: 0, algorithmic qubits: 2
    Gate breakdown:
     {'CNOT': 2, 'RZ': 1}

    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision=None, wires=None) -> None:
        self.precision = precision
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision=None, **kwargs):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
        rz = re.ResourceRZ.resource_rep(precision=precision)

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rz] = 1

        return [GateCount(cnot, 2), GateCount(rz)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * precision (float): error threshold for clifford plus T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision=None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def adjoint_resource_decomp(cls, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
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
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

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
            base_cmpr_op=re.ResourceRZ.resource_rep(precision=precision),
            num_ctrl_wires=ctrl_num_ctrl_wires,
            num_ctrl_values=ctrl_num_ctrl_values,
        )
        return [GateCount(cnot, 2), GateCount(ctrl_rz)]

    @classmethod
    def pow_resource_decomp(cls, pow_z, precision=None, **kwargs) -> list[GateCount]:
        r"""Returns a list representing the resources for an operator raised to a power.

        Args:
            pow_z (int): the power that the operator is being raised to
            precision (float, optional): error threshold for clifford plus T decomposition of this operation

        Resources:
            Taking arbitrary powers of a rotation produces a sum of rotations.
            The resources simplify to just one total Ising rotation.

        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(precision=precision))]

class PSWAP(ResourceOperator):
    r"""Resource class for the PSWAP gate.

    Args:
        precision (float | None): error threshold for Clifford+T decomposition of this operation
        wires (WiresLike | None): the wire the operation acts on

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

    >>> from pennylane import estimator as qre
    >>> pswap = qre.PWAP()
    >>> gate_set = {"CNOT", "SWAP", "PhaseShift"}
    >>> print(qre.estimate(pswap, gate_set))


    """

    num_wires = 2
    resource_keys = {"precision"}

    def __init__(self, precision: float | None = None, wires: WiresLike | None = None) -> None:
        self.precision = precision
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(cls, precision: float | None = None):
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            precision (float | None): error threshold for Clifford+T decomposition of this operation

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
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        swap = qre.SWAP.resource_rep()
        cnot = qre.CNOT.resource_rep()
        phase = qre.PhaseShift.resource_rep(precision=precision)

        return [GateCount(swap), GateCount(phase), GateCount(cnot, 2)]

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
            * precision (float | None): error threshold for Clifford+T decomposition of this operation
        """
        return {"precision": self.precision}

    @classmethod
    def resource_rep(cls, precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            precision (float | None): error threshold for Clifford+T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, cls.num_wires, {"precision": precision})

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params(dict): A dictionary containing the resource parameters of
                the target operator.

        Resources:
            The adjoint of this operator just changes the sign of the phase angle, thus
            the resources of the adjoint operation results in the original operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(precision=precision))]

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires: int,
        num_zero_ctrl: int,
        target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are controlled when in the :math:`|0\rangle` state
            target_resource_params(dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are derived from the following identity. If an operation :math:`\hat{A}`
            can be expressed as :math:`\hat{A} \ = \ \hat{U} \cdot \hat{B} \cdot \hat{U}^{\dagger}`
            then the controlled operation :math:`C\hat{A}` can be expressed as:

            .. math:: C\hat{A} \ = \ \hat{U} \cdot C\hat{B} \cdot \hat{U}^{\dagger}

            Specifically, the resources are one multi-controlled phase shift gate, one multi-controlled
            SWAP gate and a pair of :class:`~.pennylane.estimator.ops.CNOT` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = target_resource_params["precision"]
        cnot = qre.CNOT.resource_rep()
        ctrl_swap = qre.Controlled.resource_rep(
            base_cmpr_op=qre.SWAP.resource_rep(),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        ctrl_ps = qre.Controlled.resource_rep(
            base_cmpr_op=qre.PhaseShift.resource_rep(precision=precision),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )

        return [GateCount(ctrl_swap), GateCount(cnot, 2), GateCount(ctrl_ps)]
