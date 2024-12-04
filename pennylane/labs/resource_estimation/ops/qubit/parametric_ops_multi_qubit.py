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
        The resources come from Section VIII (figure 3) of `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://arxiv.org/pdf/1208.5986>`_ paper.

        Specifically, the resources are given by one :code:`RZ` gate and a cascade of :math:`2 * (n - 1)`
        :code:`CNOT` gates where :math:`n` is the number of qubits the gate acts on.
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
    def _resource_decomp(pauli_word, **kwargs):
        if set(pauli_word) == {"I"}:
            gp = re.ResourceGlobalPhase.resource_rep()
            return {gp: 1}

        active_wires = len(pauli_word.replace("I", ""))

        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()  # TODO: add Adjoint(S) in the symbolic PRs
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        h_count = 0
        s_count = 0

        for gate in pauli_word:
            if gate == "X":
                h_count += 2
            if gate == "Y":
                h_count += 2
                s_count += 1

        gate_types = {}
        gate_types[h] = h_count
        gate_types[s] = s_count + (3 * s_count)  # S^dagg = 3*S in cost
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

        The circuit implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭●─────RX────╭●─┤
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

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ─╭●─────RY────╭●─┤
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

        The circuit implementing this transformation is given by

        .. code-block:: bash

            0: ──H─╭●─────RY────╭●──H─┤
            1: ────╰Y─────RX────╰Y────┤
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

        The circuit implmenting this transformation is given by:

        .. code-block:: bash

            0: ─╭●───────────╭●─┤
            1: ─╰X─────RZ────╰X─┤
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

        The circuit implementing this transformation is given by:

        .. code-block:: bash

            0: ─╭SWAP─╭●───────────╭●─┤
            1: ─╰SWAP─╰X─────Rϕ────╰X─┤
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
