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
r"""Resource operators for qchem operations."""
import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceSingleExcitation(qml.SingleExcitation, re.ResourceOperator):
    r"""Resource class for the SingleExcitation gate.

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        """TODO: implement in resource_symbolic_ops branch"""
        raise re.ResourcesNotDefined

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})


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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        x = re.ResourceX.resource_rep(**kwargs)
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep(**kwargs)
        cnot = re.ResourceCNOT.resource_rep(**kwargs)
        cry = re.ResourceCRY.resource_rep(**kwargs)

        gate_types = {}
        gate_types[x] = 4
        gate_types[ctrl_phase_shift] = 2
        gate_types[cnot] = 2
        gate_types[cry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        x = re.ResourceX.resource_rep(**kwargs)
        ctrl_phase_shift = re.ResourceControlledPhaseShift.resource_rep(**kwargs)
        cnot = re.ResourceCNOT.resource_rep(**kwargs)
        cry = re.ResourceCRY.resource_rep(**kwargs)

        gate_types = {}
        gate_types[x] = 4
        gate_types[ctrl_phase_shift] = 2
        gate_types[cnot] = 2
        gate_types[cry] = 1

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitation(qml.DoubleExcitation, re.ResourceOperator):
    r"""Resource class for the DoubleExcitation gate.

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle + \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle - \sin(\phi/2) |0011\rangle,
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        """See https://arxiv.org/abs/2104.05695"""
        h = re.ResourceHadamard.resource_rep(**kwargs)
        ry = re.ResourceRY.resource_rep(**kwargs)
        cnot = re.ResourceCNOT.resource_rep(**kwargs)

        gate_types = {}
        gate_types[h] = 6
        gate_types[ry] = 8
        gate_types[cnot] = 14

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitationMinus(qml.DoubleExcitationMinus, re.ResourceOperator):
    r"""Resource class for the DoubleExcitationMinus gate.


    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
            &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        """TODO: implement in resource_symbolic_op branch"""

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitationPlus(qml.DoubleExcitationPlus, re.ResourceOperator):
    r"""Resource class for the DoubleExcitationPlus gate.

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
            &|x\rangle \rightarrow e^{i\phi/2} |x\rangle,
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        """TODO: implement in resource_symbolic_op branch"""
        raise re.ResourcesNotDefined

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})


class ResourceOrbitalRotation(qml.OrbitalRotation, re.ResourceOperator):
    r"""Resource class for the OrbitalRotation gate.

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::
            &|\Phi_{0}\rangle = \cos(\phi/2)|\Phi_{0}\rangle - \sin(\phi/2)|\Phi_{1}\rangle\\
            &|\Phi_{1}\rangle = \cos(\phi/2)|\Phi_{0}\rangle + \sin(\phi/2)|\Phi_{1}\rangle,
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        fermionic_swap = re.ResourceFermionicSWAP.resource_rep(**kwargs)
        single_excitation = re.ResourceSingleExcitation.resource_rep(**kwargs)

        gate_types = {}
        gate_types[fermionic_swap] = 2
        gate_types[single_excitation] = 2

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls, **kwargs):
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
    """

    @staticmethod
    def _resource_decomp(*args, **kwargs):
        h = re.ResourceHadamard.resource_rep(**kwargs)
        multi_rz = re.ResourceMultiRZ.resource_rep(num_wires=2, **kwargs)
        rx = re.ResourceRX.resource_rep(**kwargs)
        rz = re.ResourceRZ.resource_rep(**kwargs)
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
    def resource_rep(cls, **kwargs):
        return re.CompressedResourceOp(cls, {})
