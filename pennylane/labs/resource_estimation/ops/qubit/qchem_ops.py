# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Resource operators for qchem operations."""
import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceSingleExcitationMinus(qml.SingleExcitationMinus, re.ResourceOperator):
    r"""Resource class for the SingleExcitationMinus gate.

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U_-(\phi) = \begin{bmatrix}
                    e^{-i\phi/2} & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{-i\phi/2}
                \end{bmatrix}.

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ──X─╭Rϕ────X─╭●────╭●─╭RY───╭●─┤
            1: ──X─╰●─────X─╰Rϕ───╰X─╰●────╰X─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        x = re.ResourceX.resource_rep()
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        cry = re.ResourceCRY.resource_rep()

        gate_types = {}
        gate_types[x] = 4
        gate_types[ctrl_phase_shift] = 2
        gate_types[cnot] = 2
        gate_types[cry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})


class ResourceSingleExcitationPlus(qml.SingleExcitationPlus, re.ResourceOperator):
    r"""Resource class for the SingleExcitationPlus gate.

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U_+(\phi) = \begin{bmatrix}
                    e^{i\phi/2} & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{i\phi/2}
                \end{bmatrix}.

        The circuit implmementing this transformation is given by

        .. code-block:: bash

            0: ──X─╭Rϕ──X─╭●───╭●─╭RY──╭●─┤
            1: ──X─╰●───X─╰Rϕ──╰X─╰●───╰X─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        x = re.ResourceX.resource_rep()
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()
        cry = re.ResourceCRY.resource_rep()

        gate_types = {}
        gate_types[x] = 4
        gate_types[ctrl_phase_shift] = 2
        gate_types[cnot] = 2
        gate_types[cry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitation(qml.DoubleExcitation, re.ResourceOperator):
    r"""Resource class for the DoubleExcitation gate.

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle + \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle - \sin(\phi/2) |0011\rangle,

        For the source of this decomposition, see page 17 of
        `"Local, Expressive, Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ────╭●──H─╭●──RY───╭●──RY─────────────╭X──RY──────────╭●──RY───╭●─╭X──H──╭●────┤
            1: ────│─────╰X──RY───│───────╭X──RY──╭X─│───RY────╭X────│───RY───╰X─│──────│─────┤
            2: ─╭●─╰X─╭●──────────│───────│───────╰●─╰●────────│─────│───────────╰●─────╰X─╭●─┤
            3: ─╰X──H─╰X──────────╰X──H───╰●───────────────────╰●──H─╰X──H─────────────────╰X─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        h = re.ResourceHadamard.resource_rep()
        ry = re.ResourceRY.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types = {}
        gate_types[h] = 6
        gate_types[ry] = 8
        gate_types[cnot] = 14

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})


class ResourceFermionicSWAP(qml.FermionicSWAP, re.ResourceOperator):
    r"""Resource class for the FermionicSWAP gate.

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & e^{i \phi/2} \cos(\phi/2) & -ie^{i \phi/2} \sin(\phi/2) & 0 \\
                    0 & -ie^{i \phi/2} \sin(\phi/2) & e^{i \phi/2} \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{i \phi}
                \end{bmatrix}.

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ──H─╭MultiRZ──H──RX─╭MultiRZ──RX──RZ─╭Exp(I)─┤
            1: ──H─╰MultiRZ──H──RX─╰MultiRZ──RX──RZ─╰Exp(I)─┤
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        h = re.ResourceHadamard.resource_rep()
        multi_rz = re.ResourceMultiRZ.resource_rep(num_wires=2)
        rx = re.ResourceRX.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        phase = re.ResourceGlobalPhase.resource_rep()

        gate_types = {}
        gate_types[h] = 4
        gate_types[multi_rz] = 2
        gate_types[rx] = 4
        gate_types[rz] = 2
        gate_types[phase] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return re.CompressedResourceOp(cls, {})
