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
        .. math::

            MultiRZ(\theta) = \exp\left(-i \frac{\theta}{2} Z^{\otimes n}\right)
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
    def resource_rep(cls, num_wires, **kwargs):
        return re.CompressedResourceOp(cls, {"num_wires": num_wires})


class ResourcePauliRot(qml.PauliRot, re.ResourceOperator):
    r"""Resource class for the PauliRot gate.

    Resources:
        .. math::

            RP(\theta, P) = \exp\left(-i \frac{\theta}{2} P\right)
    """

    @staticmethod
    def _resource_decomp(pauli_word, **kwargs):
        if set(pauli_word) == {"I"}:
            gp = re.ResourceGlobalPhase.resource_rep(**kwargs)
            return {gp: 1}

        active_wires = len(pauli_word.replace("I", ""))

        h = re.ResourceHadamard.resource_rep(**kwargs)
        rx = re.ResourceRX.resource_rep(**kwargs)
        multi_rz = re.ResourceMultiRZ.resource_rep(active_wires, **kwargs)

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
        gate_types[multi_rz] = 1

        return gate_types

    def resource_params(self):
        return {
            "pauli_word": self.hyperparameters["pauli_word"],
        }

    @classmethod
    def resource_rep(cls, pauli_word, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cnot = re.ResourceCNOT.resource_rep(**kwargs)
        rx = re.ResourceRX.resource_rep(**kwargs)

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rx] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cy = re.ops.ResourceCY.resource_rep(**kwargs)
        ry = re.ops.ResourceRY.resource_rep(**kwargs)

        gate_types = {}
        gate_types[cy] = 2
        gate_types[ry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        h = re.ResourceHadamard.resource_rep(**kwargs)
        cy = re.ResourceCY.resource_rep(**kwargs)
        ry = re.ResourceRY.resource_rep(**kwargs)
        rx = re.ResourceRX.resource_rep(**kwargs)

        gate_types = {}
        gate_types[h] = 2
        gate_types[cy] = 2
        gate_types[ry] = 1
        gate_types[rx] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        cnot = re.ResourceCNOT.resource_rep(**kwargs)
        rz = re.ResourceRZ.resource_rep(**kwargs)

        gate_types = {}
        gate_types[cnot] = 2
        gate_types[rz] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        swap = re.ResourceSWAP.resource_rep(**kwargs)
        cnot = re.ResourceCNOT.resource_rep(**kwargs)
        phase = re.ResourcePhaseShift.resource_rep(**kwargs)

        gate_types = {}
        gate_types[swap] = 1
        gate_types[cnot] = 2
        gate_types[phase] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, *args, **kwargs):
        return re.CompressedResourceOp(cls, {})