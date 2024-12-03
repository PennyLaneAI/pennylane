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
import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceMultiRZ(qml.MultiRZ, re.ResourceOperator):
    r"""Resource class for the MultiRZ gate.

    Resources:
        The resources come from Section VIII of "The Bravyi-Kitaev transformation for quantum computation
        of electronic structure" (https://arxiv.org/pdf/1208.5986). See Figure 3 of that section for
        an illustration.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs):
        cnot = re.CompressedResourceOp(re.ResourceCNOT, {})
        rz = re.CompressedResourceOp(re.ResourceRZ, {})

        gate_types = {}
        gate_types[cnot] = 2 * (num_wires - 1)
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires):
        return re.CompressedResourceOp(cls, {"num_wires": num_wires})


class ResourcePauliRot(qml.PauliRot, re.ResourceOperator):
    r"""Resource class for the PauliRot gate.

    Resources:
        The resources come from Section VIII of "The Bravyi-Kitaev transformation for quantum computation
        of electronic structure" (https://arxiv.org/pdf/1208.5986). See Figure 4 of that section for
        an illustration.
    """

    @staticmethod
    def _resource_decomp(pauli_word, **kwargs):
        if set(pauli_word) == {"I"}:
            gp = re.ResourceGlobalPhase.resource_rep()
            return {gp: 1}

        active_wires = len(pauli_word.replace("I", ""))

        h = re.ResourceHadamard.resource_rep()
        rx = re.ResourceRX.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        h_count = 0
        rx_count = 0

        for gate in pauli_word:
            if gate == "X":
                h_count += 2
            if gate == "Y":
                rx_count += 2

        gate_types = {}
        gate_types[h] = h_count
        gate_types[rx] = rx_count
        gate_types[rz] = 1
        gate_types[cnot] = 2 * (active_wires - 1)

        return gate_types

    def resource_params(self):
        return {
            "pauli_word": self.hyperparameters["pauli_word"],
        }

    @classmethod
    def resource_rep(cls, pauli_word):
        return re.CompressedResourceOp(cls, {"pauli_word": pauli_word})


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

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ─╭●──RX(0.10)─╭●─┤
            1: ─╰X───────────╰X─┤
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cnot = re.ResourceCNOT.resource_rep()
        rx = re.ResourceRX.resource_rep()

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rx] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args):
        return re.CompressedResourceOp(cls, {})


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

        The circuit implementing this transoformation is given by

        .. code-block: bash

            0: ─╭●──RY(0.10)─╭●─┤
            1: ─╰Y───────────╰Y─┤
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cy = re.ops.ResourceCY.resource_rep()
        ry = re.ops.ResourceRY.resource_rep()

        gate_types = {}
        gate_types[cy] = 2
        gate_types[ry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args):
        return re.CompressedResourceOp(cls, {})


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

        The circuit implementing this gate is given by

        .. code-block:: bash

            0: ──H─╭●──RY(0.05)──╭●──H─┤
            1: ────╰Y──RX(-0.05)─╰Y────┤
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
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
    def resource_rep(cls, *args):
        return re.CompressedResourceOp(cls, {})


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

        The circuit implmenting this transformation is given by

        .. code-block:: bash

            0: ─╭●───────────╭●─┤
            1: ─╰X──RZ(0.10)─╰X─┤
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args):
        return re.CompressedResourceOp(cls, {})


class ResourcePSWAP(qml.PSWAP, re.ResourceOperator):
    r"""Resource class for the PSWAP gate.

    Resources:
        .. math:: PSWAP(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & e^{i \phi} & 0 \\
                0 & e^{i \phi} & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{bmatrix}.

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ─╭SWAP─╭●───────────╭●─┤
            1: ─╰SWAP─╰X──Rϕ(0.10)─╰X─┤
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
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
    def resource_rep(cls, *args):
        return re.CompressedResourceOp(cls, {})
