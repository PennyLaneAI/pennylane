# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceMultiRZ(qml.MultiRZ, re.ResourceOperator):
    r"""Resource class for the MultiRZ gate.

    Resources:
        The resources come from Section VIII (figure 3) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper.

        Specifically, the resources are given by one :code:`RZ` gate and a cascade of :math:`2 * (n - 1)`
        :code:`CNOT` gates where :math:`n` is the number of qubits the gate acts on.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs):
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types = {}
        gate_types[cnot] = 2 * (num_wires - 1)
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires):
        return re.CompressedResourceOp(cls, {"num_wires": num_wires})

    @classmethod
    def adjoint_resource_decomp(cls, num_wires) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(num_wires=num_wires): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
        num_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            cnot = re.ResourceCNOT.resource_rep()
            ctrl_rz = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRZ,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[cnot] = 2 * (num_wires - 1)
            gate_types[ctrl_rz] = 1

            return gate_types

        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z, num_wires) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(num_wires=num_wires): 1}


class ResourcePauliRot(qml.PauliRot, re.ResourceOperator):
    r"""Resource class for the PauliRot gate.

    Resources:
        The resources come from Section VIII (figures 3, 4) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper, in
        combination with the following identity:

        .. math::

            \begin{align}
                \hat{X} &= \hat{H} \cdot \hat{Z} \cdot \hat{H}, \\
                \hat{Y} &= \hat{S} \cdot \hat{H} \cdot \hat{Z} \cdot \hat{H} \cdot \hat{S}^{\dagger}.
            \end{align}

        Specifically, the resources are given by one :code:`RZ` gate and a cascade of :math:`2 * (n - 1)`
        :code:`CNOT` gates where :math:`n` is the number of qubits the gate acts on. Additionally, for
        each :code:`X` gate in the Pauli word we conjugate by a :code:`Hadamard` gate, and for each
        :code:`Y` gate in the Pauli word we conjugate by :code:`Hadamard` and :code:`S` gates.

    """

    @staticmethod
    def _resource_decomp(pauli_string, **kwargs):
        if (set(pauli_string) == {"I"}) or (len(pauli_string) == 0):
            gp = re.ResourceGlobalPhase.resource_rep()
            return {gp: 1}

        if pauli_string == "X":
            return {re.ResourceRX.resource_rep(): 1}
        if pauli_string == "Y":
            return {re.ResourceRY.resource_rep(): 1}
        if pauli_string == "Z":
            return {re.ResourceRZ.resource_rep(): 1}

        active_wires = len(pauli_string.replace("I", ""))

        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        s_dagg = re.ResourceAdjoint.resource_rep(re.ResourceS, {})
        cnot = re.ResourceCNOT.resource_rep()

        h_count = 0
        s_count = 0

        for gate in pauli_string:
            if gate == "X":
                h_count += 1
            if gate == "Y":
                h_count += 1
                s_count += 1

        gate_types = {}
        if h_count:
            gate_types[h] = 2 * h_count

        if s_count:
            gate_types[s] = s_count
            gate_types[s_dagg] = s_count

        gate_types[rz] = 1
        gate_types[cnot] = 2 * (active_wires - 1)

        return gate_types

    def resource_params(self):
        return {
            "pauli_string": self.hyperparameters["pauli_word"],
        }

    @classmethod
    def resource_rep(cls, pauli_string):
        return re.CompressedResourceOp(cls, {"pauli_string": pauli_string})

    @classmethod
    def adjoint_resource_decomp(cls, pauli_string) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(pauli_string=pauli_string): 1}

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
        pauli_string,
    ) -> Dict[re.CompressedResourceOp, int]:
        base_gate_types = cls.resources(pauli_string)

        pivotal_gates = (
            re.ResourceRX.resource_rep(),
            re.ResourceRY.resource_rep(),
            re.ResourceRZ.resource_rep(),
            re.ResourceGlobalPhase.resource_rep(),
        )

        for gate in pivotal_gates:
            if gate in base_gate_types:
                counts = base_gate_types.pop(gate)
                ctrl_gate = re.ResourceControlled.resource_rep(
                    gate.op_type,
                    gate.params,
                    num_ctrl_wires,
                    num_ctrl_values,
                    num_work_wires,
                )

                base_gate_types[ctrl_gate] = counts

        return base_gate_types

    @classmethod
    def pow_resource_decomp(cls, z, pauli_string) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(pauli_string=pauli_string): 1}


class ResourceIsingXX(qml.IsingXX, re.ResourceOperator):
    r"""Resource class for the IsingXX gate.

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
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        cnot = re.ResourceCNOT.resource_rep()
        rx = re.ResourceRX.resource_rep()

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rx] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            cnot = re.ResourceCNOT.resource_rep()
            ctrl_rx = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRX,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[cnot] = 2
            gate_types[ctrl_rx] = 1

            return gate_types
        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}


class ResourceIsingYY(qml.IsingYY, re.ResourceOperator):
    r"""Resource class for the IsingYY gate.

    Resources:
        Ising YY coupling gate

        .. math:: \mathtt{YY}(\phi) = \exp\left(-i \frac{\phi}{2} (Y \otimes Y)\right) =
            \begin{bmatrix}
                \cos(\phi / 2) & 0 & 0 & i \sin(\phi / 2) \\
                0 & \cos(\phi / 2) & -i \sin(\phi / 2) & 0 \\
                0 & -i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                i \sin(\phi / 2) & 0 & 0 & \cos(\phi / 2)
            \end{bmatrix}.

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ─╭●─────RY────╭●─┤
            1: ─╰Y───────────╰Y─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        cy = re.ops.ResourceCY.resource_rep()
        ry = re.ops.ResourceRY.resource_rep()

        gate_types = {}
        gate_types[cy] = 2
        gate_types[ry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            cy = re.ops.ResourceCY.resource_rep()
            ctrl_ry = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRY,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[cy] = 2
            gate_types[ctrl_ry] = 1

            return gate_types
        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}


class ResourceIsingXY(qml.IsingXY, re.ResourceOperator):
    r"""Resource class for the IsingXY gate.

    Resources:
        Ising (XX + YY) coupling gate

        .. math:: \mathtt{XY}(\phi) = \exp\left(i \frac{\theta}{4} (X \otimes X + Y \otimes Y)\right) =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos(\phi / 2) & i \sin(\phi / 2) & 0 \\
                0 & i \sin(\phi / 2) & \cos(\phi / 2) & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ──H─╭●─────RY────╭●──H─┤
            1: ────╰Y─────RX────╰Y────┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        h = re.ResourceHadamard.resource_rep()
        cy = re.ResourceCY.resource_rep()
        ry = re.ResourceRY.resource_rep()
        rx = re.ResourceRX.resource_rep()

        gate_types = {}
        gate_types[h] = 2
        gate_types[cy] = 2
        gate_types[ry] = 1
        gate_types[rx] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            h = re.ResourceHadamard.resource_rep()
            cy = re.ResourceCY.resource_rep()
            ctrl_rx = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRX,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )
            ctrl_ry = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRY,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[h] = 2
            gate_types[cy] = 2
            gate_types[ctrl_ry] = 1
            gate_types[ctrl_rx] = 1

            return gate_types
        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}


class ResourceIsingZZ(qml.IsingZZ, re.ResourceOperator):
    r"""Resource class for the IsingZZ gate.

    Resources:
        Ising ZZ coupling gate

        .. math:: ZZ(\phi) = \exp\left(-i \frac{\phi}{2} (Z \otimes Z)\right) =
            \begin{bmatrix}
                e^{-i \phi / 2} & 0 & 0 & 0 \\
                0 & e^{i \phi / 2} & 0 & 0 \\
                0 & 0 & e^{i \phi / 2} & 0 \\
                0 & 0 & 0 & e^{-i \phi / 2}
            \end{bmatrix}.

        The circuit implmenting this transformation is given by:

        .. code-block:: bash

            0: ─╭●───────────╭●─┤
            1: ─╰X─────RZ────╰X─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            cnot = re.ResourceCNOT.resource_rep()
            ctrl_rz = re.ResourceControlled.resource_rep(
                base_class=re.ResourceRZ,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[cnot] = 2
            gate_types[ctrl_rz] = 1

            return gate_types
        raise re.ResourcesNotDefined

    @classmethod
    def pow_resource_decomp(cls, z) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}


class ResourcePSWAP(qml.PSWAP, re.ResourceOperator):
    r"""Resource class for the PSWAP gate.

    Resources:
        .. math:: PSWAP(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & e^{i \phi} & 0 \\
                0 & e^{i \phi} & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

        The circuit implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭SWAP─╭●───────────╭●─┤
            1: ─╰SWAP─╰X─────Rϕ────╰X─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        swap = re.ResourceSWAP.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        phase = re.ResourcePhaseShift.resource_rep()

        gate_types = {}
        gate_types[swap] = 1
        gate_types[cnot] = 2
        gate_types[phase] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})

    @classmethod
    def adjoint_resource_decomp(cls) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(): 1}

    @staticmethod
    def controlled_resource_decomp(
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        if num_ctrl_values == 0:
            cnot = re.ResourceCNOT.resource_rep()
            ctrl_swap = re.ResourceControlled.resource_rep(
                base_class=re.ResourceSWAP,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )
            ctrl_ps = re.ResourceControlled.resource_rep(
                base_class=re.ResourcePhaseShift,
                base_params={},
                num_ctrl_wires=num_ctrl_wires,
                num_ctrl_values=num_ctrl_values,
                num_work_wires=num_work_wires,
            )

            gate_types = {}
            gate_types[ctrl_swap] = 1
            gate_types[cnot] = 2
            gate_types[ctrl_ps] = 1
            return gate_types

        raise re.ResourcesNotDefined
