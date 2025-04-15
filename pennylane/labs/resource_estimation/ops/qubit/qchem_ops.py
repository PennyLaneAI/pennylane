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

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ──T†──H───S─╭X──RZ-─╭X──S†──H──T─┤
            1: ──T†──S†──H─╰●──RY──╰●──H───S──T─┤

    .. seealso:: :class:`~.SingleExcitation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSingleExcitation.resources()
    {Adjoint(T): 2, Hadamard: 4, S: 2, Adjoint(S): 2, CNOT: 2, RZ: 1, RY: 1, T: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following matrix into fundamental gates.

            .. math:: U(\phi) = \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                        0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                        0 & 0 & 0 & 1
                    \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ──T†──H───S─╭X──RZ-─╭X──S†──H──T─┤
                1: ──T†──S†──H─╰●──RY──╰●──H───S──T─┤

        """
        t_dag = re.ResourceAdjoint.resource_rep(re.ResourceT, {})
        h = re.ResourceHadamard.resource_rep()
        s = re.ResourceS.resource_rep()
        s_dag = re.ResourceAdjoint.resource_rep(re.ResourceS, {})
        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        ry = re.ResourceRY.resource_rep()
        t = re.ResourceT.resource_rep()

        gate_types = {}
        gate_types[t_dag] = 2
        gate_types[h] = 4
        gate_types[s] = 2
        gate_types[s_dag] = 2
        gate_types[cnot] = 2
        gate_types[rz] = 1
        gate_types[ry] = 1
        gate_types[t] = 2

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceSingleExcitationMinus(qml.SingleExcitationMinus, re.ResourceOperator):
    r"""Resource class for the SingleExcitationMinus gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U_-(\phi) = \begin{bmatrix}
                    e^{-i\phi/2} & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{-i\phi/2}
                \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ──X─╭Rϕ────X─╭●────╭●─╭RY───╭●─┤
            1: ──X─╰●─────X─╰Rϕ───╰X─╰●────╰X─┤

    .. seealso:: :class:`~.SingleExcitationMinus`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSingleExcitationMinus.resources()
    {X: 4, ControlledPhaseShift: 2, CNOT: 2, CRY: 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following matrix into fundamental gates.

            .. math:: U_-(\phi) = \begin{bmatrix}
                        e^{-i\phi/2} & 0 & 0 & 0 \\
                        0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                        0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                        0 & 0 & 0 & e^{-i\phi/2}
                    \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ──X─╭Rϕ────X─╭●────╭●─╭RY───╭●─┤
                1: ──X─╰●─────X─╰Rϕ───╰X─╰●────╰X─┤

        """
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

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceSingleExcitationPlus(qml.SingleExcitationPlus, re.ResourceOperator):
    r"""Resource class for the SingleExcitationPlus gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U_+(\phi) = \begin{bmatrix}
                    e^{i\phi/2} & 0 & 0 & 0 \\
                    0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                    0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{i\phi/2}
                \end{bmatrix}.

        The cost for implmementing this transformation is given by:

        .. code-block:: bash

            0: ──X─╭Rϕ──X─╭●───╭●─╭RY──╭●─┤
            1: ──X─╰●───X─╰Rϕ──╰X─╰●───╰X─┤

    .. seealso:: :class:`~.SingleExcitationPlus`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceSingleExcitationPlus.resources()
    {X: 4, ControlledPhaseShift: 2, CNOT: 2, CRY: 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following matrix into fundamental gates.

            .. math:: U_+(\phi) = \begin{bmatrix}
                        e^{i\phi/2} & 0 & 0 & 0 \\
                        0 & \cos(\phi/2) & -\sin(\phi/2) & 0 \\
                        0 & \sin(\phi/2) & \cos(\phi/2) & 0 \\
                        0 & 0 & 0 & e^{i\phi/2}
                    \end{bmatrix}.

            The cost for implmementing this transformation is given by:

            .. code-block:: bash

                0: ──X─╭Rϕ──X─╭●───╭●─╭RY──╭●─┤
                1: ──X─╰●───X─╰Rϕ──╰X─╰●───╰X─┤

        """
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

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitation(qml.DoubleExcitation, re.ResourceOperator):
    r"""Resource class for the DoubleExcitation gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle + \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle - \sin(\phi/2) |0011\rangle,

        For the source of this decomposition, see page 17 of `"Local, Expressive,
        Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ────╭●──H─╭●──RY───╭●──RY─────────────╭X──RY──────────╭●──RY───╭●─╭X──H──╭●────┤
            1: ────│─────╰X──RY───│───────╭X──RY──╭X─│───RY────╭X────│───RY───╰X─│──────│─────┤
            2: ─╭●─╰X─╭●──────────│───────│───────╰●─╰●────────│─────│───────────╰●─────╰X─╭●─┤
            3: ─╰X──H─╰X──────────╰X──H───╰●───────────────────╰●──H─╰X──H─────────────────╰X─┤

    .. seealso:: :class:`~.DoubleExcitation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceDoubleExcitation.resources()
    {Hadamard: 6, RY: 8, CNOT: 14}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following mapping into fundamental gates.

            .. math::

                &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle + \sin(\phi/2) |1100\rangle\\
                &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle - \sin(\phi/2) |0011\rangle,

            For the source of this decomposition, see page 17 of `"Local, Expressive,
            Quantum-Number-Preserving VQE Ansatze for Fermionic Systems" <https://doi.org/10.1088/1367-2630/ac2cb3>`_ .

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ────╭●──H─╭●──RY───╭●──RY─────────────╭X──RY──────────╭●──RY───╭●─╭X──H──╭●────┤
                1: ────│─────╰X──RY───│───────╭X──RY──╭X─│───RY────╭X────│───RY───╰X─│──────│─────┤
                2: ─╭●─╰X─╭●──────────│───────│───────╰●─╰●────────│─────│───────────╰●─────╰X─╭●─┤
                3: ─╰X──H─╰X──────────╰X──H───╰●───────────────────╰●──H─╰X──H─────────────────╰X─┤

        """
        h = re.ResourceHadamard.resource_rep()
        ry = re.ResourceRY.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types = {}
        gate_types[h] = 6
        gate_types[ry] = 8
        gate_types[cnot] = 14

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitationMinus(qml.DoubleExcitationMinus, re.ResourceOperator):
    r"""Resource class for the DoubleExcitationMinus gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
            &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

        Specifically, the resources are given by one :class:`~.ResourceDoubleExcitation`, one
        :class:`~.ResourcePhaseShift` gate, two multi-controlled Z-gates controlled on 3 qubits,
        and two multi-controlled phase shift gates controlled on 3 qubits.

    .. seealso:: :class:`~.DoubleExcitationMinus`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceDoubleExcitationMinus.resources()
    {GlobalPhase: 1, DoubleExcitation: 1, C(Z,3,1,0): 2, C(PhaseShift,3,1,0): 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following mapping into fundamental gates.

            .. math::

                &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
                &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
                &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

            Specifically, the resources are given by one :class:`~.ResourceDoubleExcitation`, one
            :class:`~.ResourcePhaseShift` gate, two multi-controlled Z-gates controlled on 3 qubits,
            and two multi-controlled phase shift gates controlled on 3 qubits.
        """
        phase = re.ResourceGlobalPhase.resource_rep()
        double = re.ResourceDoubleExcitation.resource_rep()
        ctrl_z = re.ResourceControlled.resource_rep(re.ResourceZ, {}, 3, 1, 0)
        ctrl_phase = re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0)

        gate_types = {}
        gate_types[phase] = 1
        gate_types[double] = 1
        gate_types[ctrl_z] = 2
        gate_types[ctrl_phase] = 2

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceDoubleExcitationPlus(qml.DoubleExcitationPlus, re.ResourceOperator):
    r"""Resource class for the DoubleExcitationPlus gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::

            &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
            &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
            &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

        Specifically, the resources are given by one :class:`~.ResourceDoubleExcitation`, one
        :class:`~.ResourcePhaseShift` gate, two multi-controlled Z-gates controlled on 3 qubits,
        and two multi-controlled phase shift gates controlled on 3 qubits.

    .. seealso:: :class:`~.DoubleExcitationPlus`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceDoubleExcitationPlus.resources()
    {GlobalPhase: 1, DoubleExcitation: 1, C(Z,3,1,0): 2, C(PhaseShift,3,1,0): 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following mapping into fundamental gates.

            .. math::

                &|0011\rangle \rightarrow \cos(\phi/2) |0011\rangle - \sin(\phi/2) |1100\rangle\\
                &|1100\rangle \rightarrow \cos(\phi/2) |1100\rangle + \sin(\phi/2) |0011\rangle\\
                &|x\rangle \rightarrow e^{-i\phi/2} |x\rangle,

            Specifically, the resources are given by one :class:`~.ResourceDoubleExcitation`, one
            :class:`~.ResourcePhaseShift` gate, two multi-controlled Z-gates controlled on 3 qubits,
            and two multi-controlled phase shift gates controlled on 3 qubits.
        """
        phase = re.ResourceGlobalPhase.resource_rep()
        double = re.ResourceDoubleExcitation.resource_rep()
        ctrl_z = re.ResourceControlled.resource_rep(re.ResourceZ, {}, 3, 1, 0)
        ctrl_phase = re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0)

        gate_types = {}
        gate_types[phase] = 1
        gate_types[double] = 1
        gate_types[ctrl_z] = 2
        gate_types[ctrl_phase] = 2

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceOrbitalRotation(qml.OrbitalRotation, re.ResourceOperator):
    r"""Resource class for the OrbitalRotation gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following mapping into fundamental gates.

        .. math::
            &|\Phi_{0}\rangle = \cos(\phi/2)|\Phi_{0}\rangle - \sin(\phi/2)|\Phi_{1}\rangle\\
            &|\Phi_{1}\rangle = \cos(\phi/2)|\Phi_{0}\rangle + \sin(\phi/2)|\Phi_{1}\rangle,

        Specifically, the resources are given by two :class:`~.ResourceSingleExcitation` gates and
        two :class:`~.ResourceFermionicSWAP` gates.

    .. seealso:: :class:`~.OrbitalRotation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceOrbitalRotation.resources()
    {FermionicSWAP: 2, SingleExcitation: 2}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following mapping into fundamental gates.

            .. math::
                &|\Phi_{0}\rangle = \cos(\phi/2)|\Phi_{0}\rangle - \sin(\phi/2)|\Phi_{1}\rangle\\
                &|\Phi_{1}\rangle = \cos(\phi/2)|\Phi_{0}\rangle + \sin(\phi/2)|\Phi_{1}\rangle,

            Specifically, the resources are given by two :class:`~.ResourceSingleExcitation` gates and
            two :class:`~.ResourceFermionicSWAP` gates.
        """
        fermionic_swap = re.ResourceFermionicSWAP.resource_rep()
        single_excitation = re.ResourceSingleExcitation.resource_rep()

        gate_types = {}
        gate_types[fermionic_swap] = 2
        gate_types[single_excitation] = 2

        return gate_types

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})


class ResourceFermionicSWAP(qml.FermionicSWAP, re.ResourceOperator):
    r"""Resource class for the FermionicSWAP gate.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wires the operation acts on
        id (str or None): String representing the operation (optional)

    Resources:
        The resources are obtained by decomposing the following matrix into fundamental gates.

        .. math:: U(\phi) = \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & e^{i \phi/2} \cos(\phi/2) & -ie^{i \phi/2} \sin(\phi/2) & 0 \\
                    0 & -ie^{i \phi/2} \sin(\phi/2) & e^{i \phi/2} \cos(\phi/2) & 0 \\
                    0 & 0 & 0 & e^{i \phi}
                \end{bmatrix}.

        The cost for implementing this transformation is given by:

        .. code-block:: bash

            0: ──H─╭MultiRZ──H──RX─╭MultiRZ──RX──RZ─╭GlobalPhase─┤
            1: ──H─╰MultiRZ──H──RX─╰MultiRZ──RX──RZ─╰GlobalPhase─┤

    .. seealso:: :class:`~.FermionicSWAP`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceFermionicSWAP.resources()
    {Hadamard: 4, MultiRZ: 2, RX: 4, RZ: 2, GlobalPhase: 1}
    """

    @staticmethod
    def _resource_decomp(**kwargs):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Resources:
            The resources are obtained by decomposing the following matrix into fundamental gates.

            .. math:: U(\phi) = \begin{bmatrix}
                        1 & 0 & 0 & 0 \\
                        0 & e^{i \phi/2} \cos(\phi/2) & -ie^{i \phi/2} \sin(\phi/2) & 0 \\
                        0 & -ie^{i \phi/2} \sin(\phi/2) & e^{i \phi/2} \cos(\phi/2) & 0 \\
                        0 & 0 & 0 & e^{i \phi}
                    \end{bmatrix}.

            The cost for implementing this transformation is given by:

            .. code-block:: bash

                0: ──H─╭MultiRZ──H──RX─╭MultiRZ──RX──RZ─╭GlobalPhase─┤
                1: ──H─╰MultiRZ──H──RX─╰MultiRZ──RX──RZ─╰GlobalPhase─┤

        """
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

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: Empty dictionary. The resources of this operation don't depend on any additional parameters.
        """
        return {}

    @classmethod
    def resource_rep(cls):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation."""
        return re.CompressedResourceOp(cls, {})
