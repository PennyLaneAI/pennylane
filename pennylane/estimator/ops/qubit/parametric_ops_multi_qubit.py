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
from collections import Counter

import pennylane.estimator as qre
from pennylane.estimator.resource_operator import GateCount, ResourceOperator
from pennylane.math.decomposition import decomp_int_to_powers_of_two
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ, signature-differs


PAULI_ROT_SPECIAL_CASES = {
    "X": lambda eps: [GateCount(qre.resource_rep(qre.RX, {"precision": eps}))],
    "Y": lambda eps: [GateCount(qre.resource_rep(qre.RY, {"precision": eps}))],
    "Z": lambda eps: [GateCount(qre.resource_rep(qre.RZ, {"precision": eps}))],
    "XX": lambda eps: [
        GateCount(qre.resource_rep(qre.RX, {"precision": eps})),
        GateCount(qre.resource_rep(qre.CNOT), count=2),
    ],
    "YY": lambda eps: [
        GateCount(qre.resource_rep(qre.RY, {"precision": eps})),
        GateCount(qre.resource_rep(qre.CY), count=2),
    ],
}


class MultiRZ(ResourceOperator):
    r"""Resource class for the MultiRZ gate.

    Args:
        num_wires (int | None): the number of wires the operation acts upon
        precision (float | None): error threshold for the Clifford + T decomposition of this operation
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources come from Section VIII (Figure 3) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/abs/1208.5986>`_ paper.

        Specifically, the resources are given by one ``RZ`` gate and a cascade of
        :math:`2 \times (n - 1)` ``CNOT`` gates where :math:`n` is the number of qubits
        the gate acts on.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.MultiRZ`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
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
        self, num_wires: int | None = None, precision: float | None = None, wires: WiresLike = None
    ) -> None:
        if num_wires is None:
            if wires is None:
                raise ValueError("Must provide at least one of `num_wires` and `wires`.")
            num_wires = len(wires)

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
            precision (float): error threshold for the Clifford + T decomposition of this operation

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
                * precision (float): error threshold for the Clifford + T decomposition of this operation
        """
        return {"num_wires": self.num_wires, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_wires: int, precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of qubits the operation acts upon
            precision (float): error threshold for the Clifford + T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        return ResourceOperator(
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
    r"""Resource class for an arbitrary Pauli word rotation operation.

    Args:
        pauli_string (str): a string describing the Pauli operators that define the rotation
        precision (float | None): error threshold for the Clifford + T decomposition of the operation
        wires (WiresLike | None): the wire the operation acts on

    Resources:
        The resources are computed based on Section VIII (Figures 3 and 4) of
        `The Bravyi-Kitaev transformation for quantum computation of electronic structure
        <https://arxiv.org/abs/1208.5986>`_, in combination with the following identities:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
            \end{align}

        Note that when the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`),
        the cost is the associated single-qubit rotation (i.e., :code:`RX, RY, RZ, GlobalPhase`). If the
        :code:`pauli_string` is :code:`XX`, the resources are one :code:`RX` gate at the specified
        precision and two :code:`CNOT` gates. If the :code:`pauli_string` is :code:`YY`, the
        resources are one :code:`RY` gate at the specified precision and two :code:`CY` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PauliRot`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
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
            pauli_string (str): a string describing the Pauli operators that define the rotation
            precision (float | None): error threshold for the Clifford + T decomposition of this operation

        Resources:
            The resources are computed based on Section VIII (Figures 3 and 4) of
            `The Bravyi-Kitaev transformation for quantum computation of electronic structure
            <https://arxiv.org/abs/1208.5986>`_, in combination with the following identities:

            .. math::

                \begin{align}
                    \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                    \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
                \end{align}

            Note that when the :code:`pauli_string` is a single Pauli operator (:code:`X, Y, Z, Identity`),
            the cost is the associated single-qubit rotation (i.e., :code:`RX, RY, RZ, GlobalPhase`). If the
            :code:`pauli_string` is :code:`XX`, the resources are one :code:`RX` gate at the specified
            precision and two :code:`CNOT` gates. If the :code:`pauli_string` is :code:`YY`, the
            resources are one :code:`RY` gate at the specified precision and two :code:`CY` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            return [GateCount(qre.resource_rep(qre.GlobalPhase))]

        # Special cases:
        if pauli_string in PAULI_ROT_SPECIAL_CASES:
            return PAULI_ROT_SPECIAL_CASES[pauli_string](eps=precision)

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
                * pauli_string (str): a string describing the Pauli operators that define the rotation
                * precision (float): error threshold for the Clifford + T decomposition of this operation
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
            pauli_string (str): a string describing the Pauli operators that define the rotation
            precision (float | None): error threshold for the Clifford + T decomposition of this operation

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`:: the operator in a compressed representation
        """
        num_wires = len(pauli_string)
        return ResourceOperator(
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
            The resources simplify to just one total Pauli rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        pauli_string = target_resource_params["pauli_string"]
        precision = target_resource_params["precision"]
        return [GateCount(cls.resource_rep(pauli_string=pauli_string, precision=precision))]


class PCPhase(ResourceOperator):
    r"""A resource operator representing a projector-controlled phase gate.

    This gate applies a complex phase :math:`e^{i\phi}` to the first ``dim``
    basis vectors of the input state while applying a complex phase :math:`e^{-i \phi}`
    to the remaining basis vectors. For example, consider the 2-qubit case where ``dim = 3``:

    .. math:: \Pi(\phi) = \begin{bmatrix}
                e^{i\phi} & 0 & 0 & 0 \\
                0 & e^{i\phi} & 0 & 0 \\
                0 & 0 & e^{i\phi} & 0 \\
                0 & 0 & 0 & e^{-i\phi}
            \end{bmatrix}.

    This can also be written as :math:`\Pi(\phi) = \exp(i\phi(2\Pi-\mathbb{I}_N))`, where
    :math:`N=2^n` is the Hilbert space dimension for :math:`n` qubits and :math:`\Pi` is
    the diagonal projector with ``dim`` ones and ``N-dim`` zeros.

    Args:
        num_wires (int): the number of wires this operator acts on
        dim (int): the dimension of the target subspace
        rotation_precision (float | None): The error threshold for the approximate Clifford + T
            decomposition of the :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates used to implement this operation.
        wires (WiresLike | None): the wire the operation acts on

    Resources:
        The resources are derived from the decomposition of the generator :math:`G` of the
        ``PCPhase`` gate into multiple projectors, which generate (multi-controlled) ``PhaseShift`` gates,
        potentially complemented with (non-controlled) ``X`` gates and/or a global phase.
        The generator is given as :math:`G = 2 \Pi - \mathbb{I}_N`, where :math:`\Pi` is a projector.
        The projector :math:`\Pi` is decomposed into sums and differences of powers of two,
        which correspond to multi-controlled :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates.

    .. seealso:: The corresponding PennyLane operation :class:`~.pennylane.PCPhase`.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> pc_phase = qre.PCPhase(num_wires=2, dim=2, rotation_precision=1e-5)
    >>> print(qre.estimate(pc_phase))
    --- Resources: ---
     Total wires: 2
       algorithmic wires: 2
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 28
       'T': 28

    """

    resource_keys = {"num_wires", "dim", "rotation_precision"}

    def __init__(
        self,
        num_wires: int,
        dim: int,
        rotation_precision: float | None = None,
        wires: WiresLike | None = None,
    ) -> None:
        self.num_wires = num_wires
        self.dim = dim
        self.rotation_precision = rotation_precision

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @classmethod
    def resource_decomp(
        cls, num_wires: int, dim: int, rotation_precision: float | None = None
    ) -> list[GateCount]:
        r"""Returns a list of GateCount objects representing the operator's resources.

        Args:
            num_wires (int): the number of wires this operator acts on
            dim (int): the dimension of the target subspace
            rotation_precision (float | None): The error threshold for the approximate Clifford + T
                decomposition of the :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates used to implement this operation.

        Resources:
            The resources are derived from the decomposition of the generator :math:`G` of the
            ``PCPhase`` gate into multiple projectors, which generate (multi-controlled) :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates,
            potentially complemented with (non-controlled) ``X`` gates and/or a global phase.
            The generator is given as :math:`G = 2 \Pi - \mathbb{I}_N`, where :math:`\Pi` is a projector.
            The projector :math:`\Pi` is decomposed into sums and differences of powers of two,
            which correspond to multi-controlled :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_count = Counter()
        flipped, *powers_of_two = decomp_int_to_powers_of_two(dim, num_wires + 1)
        sigma = (-1) ** flipped
        powers_of_two = [sigma * val for val in powers_of_two]

        n_zero_control_values = 0
        for i, c_i in enumerate(powers_of_two):

            if c_i != 0:
                subspace = int(c_i < 0)
                if flipped:
                    subspace = 1 - subspace
                gate_count.update(
                    cls._ctrl_phase_shift_resource(
                        subspace,
                        n_control_wires=i,
                        n_zero_control_values=n_zero_control_values,
                        rotation_precision=rotation_precision,
                    )
                )

            d_i = next(iter(val for val in powers_of_two[i + 1 :] if val != 0), None)
            next_cval = d_i == 1
            if c_i == 0:
                next_cval = not next_cval
            if flipped:
                next_cval = not next_cval
            if not next_cval:
                n_zero_control_values += 1

        gate_count[qre.GlobalPhase.resource_rep()] += 1
        gate_count_lst = [GateCount(op, count) for op, count in gate_count.items()]
        return gate_count_lst

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires this operator acts on
                * dim (int): the dimension of the target subspace
                * rotation_precision(float | None): The error threshold for the approximate Clifford + T
                  decomposition of the :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates used to implement this operation.
        """
        return {
            "num_wires": self.num_wires,
            "dim": self.dim,
            "rotation_precision": self.rotation_precision,
        }

    @classmethod
    def resource_rep(cls, num_wires: int, dim: int, rotation_precision: float | None = None):
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_wires (int): the number of wires this operator acts on
            dim (int): the dimension of the target subspace
            rotation_precision(float | None): The error threshold for the approximate Clifford + T
                decomposition of the :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gates used to implement this operation.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        params = {"num_wires": num_wires, "dim": dim, "rotation_precision": rotation_precision}
        return ResourceOperator(cls, num_wires, params)

    @staticmethod
    def _ctrl_phase_shift_resource(
        subspace: int,
        n_control_wires: int,
        n_zero_control_values: int,
        rotation_precision: float | None,
    ) -> dict[ResourceOperator, int]:
        """Returns the resources for a (multi-)controlled :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gate.

        Args:
            subspace (int): The subspace indicator (0 or 1) determining whether X gates are needed.
            n_control_wires (int): The number of control wires for the controlled :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift`.
            n_zero_control_values (int): The number of control qubits controlled on the |0⟩ state.
            rotation_precision (float | None): The error threshold for the approximate Clifford + T
                decomposition of the :class:`~pennylane.estimator.ops.qubit.parametric_ops_single_qubit.PhaseShift` gate.
        """
        x = qre.X.resource_rep()
        phase_shift = qre.PhaseShift.resource_rep(rotation_precision)

        if n_control_wires == 0:
            return {phase_shift: 1}
        return {
            qre.Controlled.resource_rep(
                base_cmpr_op=phase_shift,
                num_ctrl_wires=n_control_wires,
                num_zero_ctrl=n_zero_control_values,
            ): 1,
            x: 2 * (1 - subspace),
        }
