# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains resource operators for state preparation templates."""

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import ResourceOperator, CompressedResourceOp, GateCount, resource_rep
from pennylane.wires import WiresLike

class MottonenStatePreparation(ResourceOperator):
    r"""Resource class for Mottonen state preparation.

    Args:
        num_wires (int): the number of wires the operation acts on
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        Resources are described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
        The resources are defined as :math:`2^{N+2} - 5` :class:`~.pennylane.estimator.ops.qubit.RZ` gates and
        :math:`2^{N+2} - 4N - 4` :class:`~.pennylane.estimator.ops.op_math.CNOT` gates.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> mottonen_state = qre.MottonenStatePreparation(10)
    >>> gate_set = {"RZ", "CNOT"}
    >>> print(qre.estimate(mottonen_state, gate_set=gate_set))
    --- Resources: ---
     Total wires: 10
       algorithmic wires: 10
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.841E+5
       'RZ': 4.091E+3,
       'CNOT': 4.052E+3

    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int, wires: WiresLike | None = None):
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires that the operation acts on
        """
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Resources:
            Resources are described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
            The resources are defined as :math:`2^{N+2} - 5` :class:`~.pennylane.estimator.ops.qubit.RZ` gates and
            :math:`2^{N+2} - 4N - 4` :class:`~pennylane.estimator.ops.op_math.CNOT` gates.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_lst = []

        rz = resource_rep(qre.RZ)
        cnot = resource_rep(qre.CNOT)

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        gate_lst.append(GateCount(rz, r_count))
        gate_lst.append(GateCount(cnot, cnot_count))

        return gate_lst


class CosineWindow(ResourceOperator):
    r"""Resource class for preparing an initial state with a cosine wave function.

    The wave function is defined below where :math:`m` is the number of wires.

    .. math::

        |\psi\rangle = \sqrt{2^{1-m}} \sum_{k=0}^{2^m-1} \cos(\frac{\pi k}{2^m} - \frac{\pi}{2}) |k\rangle,

    .. note::

        The wave function is shifted by :math:`\frac{\pi}{2}` units so that the window is centered.

    Args:
        num_wires (int): the number of wires the operation acts on
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources were obtained from Figure 6 in arXiv:2110.09590 `<https://arxiv.org/pdf/2110.09590>`_.


    .. seealso:: :class:`~.CosineWindow`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> cosine_state = qre.CosineWindow(5)
    >>> print(qre.estimate(cosine_state))
    --- Resources: ---
     Total wires: 5
       algorithmic wires: 5
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1.616E+3
       'T': 1.584E+3,
       'CNOT': 26,
       'Hadamard': 6
    """

    resource_keys = {"num_wires"}

    def __init__(self, num_wires: int, wires: WiresLike | None = None):
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_wires (int): the number of wires that the operation acts on
        """
        return {"num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Resources:
            The resources were obtained from Figure 6 in arXiv:2110.09590 `<https://arxiv.org/pdf/2110.09590>`_

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        hadamard = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ)
        iqft = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": resource_rep(qre.QFT, {"num_wires": num_wires})},
        )
        phase_shift = resource_rep(qre.PhaseShift)

        return [
            GateCount(hadamard, 1),
            GateCount(rz, 1),
            GateCount(iqft, 1),
            GateCount(phase_shift, num_wires),
        ]


class SumOfSlatersPrep(ResourceOperator):
    r"""Resource class for preparing an initial state with the sum-of-Slaters technique.

    The operation prepares an arbitrary state

    .. math::

        |\psi\rangle = \sum_{l \in L} c_l |l \rangle


    Args:
        num_coeffs (int): number of coefficients of the sparse state to prepare.
        num_wires (int): number of wires on which the state is being prepared.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources were obtained from Sec. III A of
        `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__
        and is tailored to sparse states.

    .. seealso:: :class:`~.SumOfSlatersPrep`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> sos_state = qre.SumOfSlatersPrep(num_coeffs=100, num_wires=8)
    >>> print(qre.estimate(sos_state))

    """

    resource_keys = {"num_wires"}

    def __init__(self, num_coeffs:int, num_wires: int, wires: WiresLike | None = None):
        self.num_coeffs = num_coeffs
        self.num_wires = num_wires
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_coeffs(int): number of coefficients of the sparse state to prepare
                * num_wires (int): the number of wires that the state is being prepared on
        """
        return {"num_coeffs":self.num_coeffs, "num_wires": self.num_wires}

    @classmethod
    def resource_rep(cls, num_wires: int) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_coeffs": num_coeffs, "num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_coeffs: int, num_wires: int):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_coeffs(int): number of coefficients of the sparse state to prepare
            num_wires (int): the number of wires the state is being prepared on

        Resources:
            The resources were obtained from Sec. III A of
            `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__
            and is tailored to sparse states.


        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        hadamard = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ)
        iqft = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": resource_rep(qre.QFT, {"num_wires": num_wires})},
        )
        phase_shift = resource_rep(qre.PhaseShift)

        return [
            GateCount(hadamard, 1),
            GateCount(rz, 1),
            GateCount(iqft, 1),
            GateCount(phase_shift, num_wires),
        ]