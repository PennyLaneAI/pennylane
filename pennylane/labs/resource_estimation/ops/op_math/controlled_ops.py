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
r"""Resource operators for controlled operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ,too-many-ancestors


class ResourceCH(qml.CH, re.ResourceOperator):
    r"""Resource class for CH gate.

    Resources:
        The resources are derived from the following identities:

        .. math::

            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \dot \hat{Z}  \dot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \dot \hat{X}  \dot \hat{H}
            \end{align}


        We can control on the Pauli-X gate to obtain our controlled Hadamard gate.

    """

    # TODO: Reference this:
    # https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        ry = re.ResourceRY.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[h] = 2
        gate_types[ry] = 2
        gate_types[cnot] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCY(qml.CY, re.ResourceOperator):
    r"""Resource class for CY gate.

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \dot \hat{X} \dot \hat{S}^{\dagger}.

        We can control on the Pauli-X gate to obtain our controlled-Y gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        s = re.ResourceS.resource_rep()

        gate_types[cnot] = 1
        gate_types[s] = 2  # Assuming S^dagg ~ S in cost!

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCZ(qml.CZ, re.ResourceOperator):
    r"""Resource class for CZ

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \dot \hat{X} \dot \hat{H}.

        We can control on the Pauli-X gate to obtain our controlled-Z gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 1
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCSWAP(qml.CSWAP, re.ResourceOperator):
    r"""Resource class for CSWAP

    Resources:
        The resources are taken (figure 1d) from the paper `Shallow unitary decompositions
        of quantum Fredkin and Toffoli gates for connectivity-aware equivalent circuit averaging
        <https://arxiv.org/pdf/2305.18128>`_.

        The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
        given by:

        .. code-block:: bash

            0: ────╭●────┤
            1: ─╭X─├●─╭X─┤
            2: ─╰●─╰X─╰●─┤

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        tof = re.ResourceToffoli.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[tof] = 1
        gate_types[cnot] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCCZ(qml.CCZ, re.ResourceOperator):
    r"""Resource class for CCZ

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \dot \hat{X} \dot \hat{H}.

        We replace the Pauli-X gate with a Toffoli gate to obtain our control-control-Z gate.
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        toffoli = re.ResourceToffoli.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[toffoli] = 1
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCNOT(qml.CNOT, re.ResourceOperator):
    r"""Resource class for CNOT

    Resources:
        There is no further decomposition provided for this gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceToffoli(qml.Toffoli, re.ResourceOperator):
    r"""Resource class for Toffoli

    Resources:
        The resources are obtained from (in figure 1.) the paper `Novel constructions for the fault-tolerant
        Toffoli gate <https://arxiv.org/pdf/1212.5069>`_.

        The circuit which applies the Toffoli gate on target wire 'target' with control wires ('c1', 'c2') is
        given by:

        .. code-block:: bash

                c1: ─╭●────╭X──T†────────╭X────╭●───────────────╭●─┤
                c2: ─│──╭X─│──╭●───T†─╭●─│──╭X─│────────────────╰Z─┤
              aux1: ─╰X─│──│──╰X───T──╰X─│──│──╰X────────────────║─┤
              aux2: ──H─╰●─╰●──T─────────╰●─╰●──H──S─╭●──H──┤↗├──║─┤
            target: ─────────────────────────────────╰X──────║───║─┤
                                                             ╚═══╝

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()
        cz = re.ResourceCZ.resource_rep()

        gate_types[cnot] = 9
        gate_types[h] = 3
        gate_types[s] = 1
        gate_types[cz] = 1
        gate_types[t] = 4

        return gate_types

    @staticmethod
    def textbook_resource_decomp() -> Dict[re.CompressedResourceOp, int]:
        r"""Resources for the Toffoli gate

        Resources:
            The resources are taken (figure 4.9) from the textbook `Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

            The circuit is given by:

            .. code-block:: bash

                0: ───────────╭●───────────╭●────╭●──T──╭●─┤
                1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
                2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 6
        gate_types[h] = 2
        gate_types[t] = 7

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceMultiControlledX(qml.MultiControlledX, re.ResourceOperator):
    r"""Resource class for MultiControlledX

    Resources:
        The resources are obtained from (table 3.) the paper `Polylogarithmic-depth controlled-NOT gates
        without ancilla qubits <https://www.nature.com/articles/s41467-024-50065-x>`_. The resources are
        estimated using the following formulas.

        If the number of control qubits is 0

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        num_control = len(self.hyperparameters["control_wires"])
        num_work_wires = len(self.hyperparameters["work_wires"])
        return {"num_ctrl_wires": num_control, "num_work_wires": num_work_wires}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRX(qml.CRX, re.ResourceOperator):
    r"""Resource class for CRX

    Resources:

        The resources are derived from the following identities:

        .. math::

            \begin{align}
                \hat{RZ}(- \theta) = \hat{X} \dot \hat{RZ}(\theta) \dot \hat{X}, \\
                \hat{X} &= \hat{H} \dot \hat{Z}  \dot \hat{H}
            \end{align}

        The expression for controlled-RZ gates is used as defined in (figure 1b.) the paper
        `T-count and T-depth of any multi-qubit unitary <https://arxiv.org/pdf/2110.10292>`_.
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        h = re.ResourceHadamard.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 2
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRY(qml.CRY, re.ResourceOperator):
    r"""Resource class for CRY

    Resources:
        The resources are derived from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
        unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived with the following identity:

        .. math:: \hat{RY}(- \theta) = \hat{X} \dot \hat{RY}(\theta) \dot \hat{X}.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        ry = re.ResourceRY.resource_rep()

        gate_types[cnot] = 2
        gate_types[ry] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRZ(qml.CRZ, re.ResourceOperator):
    r"""Resource class for CRZ

    Resources:
        The resources are obtained from (in figure 1b.) the paper `T-count and T-depth of any multi-qubit
        unitary <https://arxiv.org/pdf/2110.10292>`_. The resources are derived from the following identity:

        .. math:: \hat{RZ}(- \theta) = \hat{X} \dot \hat{RZ}(\theta) \dot \hat{X}.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRot(qml.CRot, re.ResourceOperator):
    r"""Resource class for CRot

    Resources:
        TODO: Add a source for resources!

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        ry = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3
        gate_types[ry] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, re.ResourceOperator):
    r"""Resource class for the ControlledPhaseShift gate.

        Resources:
            The resources come from the following identity expressing Controlled Phase Shift
            as a product of Phase Shifts and CNOTs.


        .. math::

            CR_\phi(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\phi}
                \end{bmatrix} =
            (R_\phi(\phi/2) \otimes I) \cdot CNOT \cdot (I \otimes R_\phi(-\phi/2)) \cdot CNOT \cdot (I \otimes R_\phi(\phi/2))

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
