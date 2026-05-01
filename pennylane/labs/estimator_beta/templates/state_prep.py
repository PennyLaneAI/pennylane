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

import numpy as np

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.wires import WiresLike

# pylint: disable=arguments-differ, too-many-arguments


class LabsMottonenStatePreparation(ResourceOperator):
    r"""Resource class for Mottonen state preparation.

    Args:
        num_wires (int): the number of wires the operation acts on
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        Resources are described in `Mottonen et al. (2008) <https://arxiv.org/pdf/quant-ph/0407010>`_.
        The resources are defined as :math:`2^{n+2} - 5` :class:`~.pennylane.estimator.ops.qubit.RZ` gates and
        :math:`2^{n+2} - 4n - 4` :class:`~.pennylane.estimator.ops.op_math.CNOT` gates.

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
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
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
            The resources are defined as :math:`2^{n+2} - 5` :class:`~.pennylane.estimator.ops.qubit.RZ` gates and
            :math:`2^{n+2} - 4n - 4` :class:`~.pennylane.estimator.ops.op_math.CNOT` gates.

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


class LabsCosineWindow(ResourceOperator):
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
        The resources were obtained from Figure 6 in `arXiv:2110.09590 <https://arxiv.org/pdf/2110.09590>`_.

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
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        return CompressedResourceOp(cls, num_wires, {"num_wires": num_wires})

    @classmethod
    def resource_decomp(cls, num_wires: int):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_wires (int): the number of wires that the operation acts on

        Resources:
            The resources were obtained from Figure 6 in `arXiv:2110.09590 <https://arxiv.org/pdf/2110.09590>`_.

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


class LabsSumOfSlatersPrep(ResourceOperator):
    r"""Resource class for preparing an initial state with the sum-of-Slaters technique.

    The operation prepares an arbitrary state

    .. math::

        |\psi\rangle = \sum_{l \in L} c_l |l \rangle

    Args:
        num_coeffs (int): number of coefficients of the sparse state to prepare.
        num_wires (int): number of wires on which the state is being prepared.
        num_bits (int | None): number of bits that is sufficient to uniquely identify every Slater determinant in the
            target state, as defined in Sec. III A of `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.
        stateprep_op (ResourceOperator | None): An optional argument to set the subroutine used to perform the condensed state preparation. If :code:`None`
            is provided, the resources will be computed assuming the condensed state preparation is performed using
            :class:`~.pennylane.labs.estimator_beta.templates.state_prep.LabsMottonenStatePreparation`.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM` used to trade-off extra qubits for reduced circuit depth.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources were obtained from Sec. III A of
        `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.

    .. seealso:: :class:`~.SumOfSlatersPrep`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.labs.estimator_beta as qre
    >>> sos_state = qre.SumOfSlatersPrep(num_coeffs=100, num_wires=10)
    >>> print(qre.estimate(sos_state))
    --- Resources: ---
     Total wires: 32
       algorithmic wires: 10
       allocated wires: 22
         zero state: 22
         any state: 0
     Total gates : 2.877E+4
       'Toffoli': 949,
       'T': 2.231E+4,
       'CNOT': 1.888E+3,
       'X': 1.107E+3,
       'Hadamard': 2.520E+3

    """

    resource_keys = {
        "num_coeffs",
        "num_wires",
        "num_bits",
        "stateprep_cmpr_op",
        "select_swap_depth",
    }

    def __init__(
        self,
        num_coeffs: int,
        num_wires: int,
        num_bits: int | None = None,
        stateprep_op: ResourceOperator | None = None,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        _dequeue(stateprep_op)

        if num_coeffs > 2**num_wires:
            raise ValueError(
                f"Number of coefficients {num_coeffs} cannot be greater than 2^num_wires, {2**num_wires}."
            )

        if num_bits is not None and num_bits > num_wires:
            raise ValueError(f"num_bits {num_bits} cannot be greater than num_wires, {num_wires}.")

        self.num_coeffs = num_coeffs
        self.num_wires = num_wires
        self.num_bits = num_bits
        self.stateprep_cmpr_op = stateprep_op.resource_rep_from_op() if stateprep_op else None
        self.select_swap_depth = select_swap_depth
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_coeffs(int): number of coefficients of the sparse state to prepare
                * num_wires (int): the number of wires that the state is being prepared on
                * num_bits (int | None): number of bits that is sufficient to uniquely identify
                    every Slater determinant in the target state, as defined in Sec. III A of
                    `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.
                * stateprep_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional argument to
                    set the subroutine used to perform the condensed state preparation. If :code:`None` is provided, the resources will be computed
                    assuming the condensed state preparation is performed using :class:`~.pennylane.labs.estimator_beta.templates.state_prep.LabsMottonenStatePreparation`.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM` used to trade-off extra qubits for reduced circuit depth.

        """
        return {
            "num_coeffs": self.num_coeffs,
            "num_wires": self.num_wires,
            "num_bits": self.num_bits,
            "stateprep_cmpr_op": self.stateprep_cmpr_op,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        num_coeffs: int,
        num_wires: int,
        num_bits: int | None = None,
        stateprep_cmpr_op: CompressedResourceOp | None = None,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if num_coeffs > 2**num_wires:
            raise ValueError(
                f"Number of coefficients {num_coeffs} cannot be greater than 2^num_wires, {2**num_wires}."
            )

        if num_bits is not None and num_bits > num_wires:
            raise ValueError(f"num_bits {num_bits} cannot be greater than num_wires, {num_wires}.")

        return CompressedResourceOp(
            cls,
            num_wires,
            {
                "num_coeffs": num_coeffs,
                "num_wires": num_wires,
                "num_bits": num_bits,
                "stateprep_cmpr_op": stateprep_cmpr_op,
                "select_swap_depth": select_swap_depth,
            },
        )

    @classmethod
    def resource_decomp(
        cls,
        num_coeffs: int,
        num_wires: int,
        num_bits: int | None = None,
        stateprep_cmpr_op: CompressedResourceOp | None = None,
        select_swap_depth: int | None = None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list represents a gate and the
        number of times it occurs in the circuit.

        Args:
            num_coeffs(int): number of coefficients of the sparse state to prepare
            num_wires (int): the number of wires the state is being prepared on
            num_bits (int | None): number of bits that is sufficient to uniquely identify every
                Slater determinant in the target state, as defined in Sec. III A of
                `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.
            stateprep_cmpr_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional argument to set the subroutine used to
                perform the condensed state preparation. If :code:`None`
                is provided, the resources will be computed assuming the condensed state preparation is performed
                using :class:`~.pennylane.labs.estimator_beta.templates.state_prep.LabsMottonenStatePreparation`.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM` used to trade-off extra qubits for reduced circuit depth.

        Resources:
            The resources were obtained from Sec. III A of
            `Fomichev et al., PRX Quantum 5, 040339 <https://doi.org/10.1103/PRXQuantum.5.040339>`__.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        gate_list = []

        if num_coeffs == 1:
            return [GateCount(resource_rep(qre.BasisState, {"num_wires": num_wires}), 1)]

        # Step 1: Prepare the condensed state
        condensed_state_qubits = int(np.ceil(np.log2(num_coeffs)))

        if num_bits is None:
            num_bits = min(num_wires, 2 * condensed_state_qubits - 1)

        enumeration_reg = qre.Allocate(condensed_state_qubits, restored=True)
        gate_list.append(enumeration_reg)  # enumeration register d
        if stateprep_cmpr_op is None:
            stateprep_cmpr_op = resource_rep(
                LabsMottonenStatePreparation, {"num_wires": condensed_state_qubits}
            )

        gate_list.append(GateCount(stateprep_cmpr_op, 1))

        # Step 2: Use QROM to load v_bits into system register

        qrom = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": num_coeffs,
                "size_bitstring": num_wires,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )

        gate_list.append(GateCount(qrom, 1))

        m = min(num_bits, 2 * condensed_state_qubits - 1)
        identity_encoding = num_bits == m

        # Steps 3-4 and 6: Encode/uncompute the identification register using multi-controlled Toffoli and CNOT gates
        # if identity encoding is True, then the identification register fits inside the system register, so no extra allocation needed
        if not identity_encoding:
            identification_reg = qre.Allocate(m, restored=True)
            gate_list.append(identification_reg)  # identification register
            cnot = resource_rep(qre.CNOT)
            gate_list.append(GateCount(cnot, 2 * num_wires * m))

        # Step 5: Use identification register to uncompute the enumeration register
        # Taking the upper-bound

        x = resource_rep(qre.X)
        gate_list.append(GateCount(x, num_coeffs * m))
        mcx = resource_rep(qre.MultiControlledX, {"num_ctrl_wires": m, "num_zero_ctrl": 0})
        num_mcx = num_coeffs - 1
        gate_list.append(GateCount(mcx, num_mcx))

        if not identity_encoding:
            gate_list.append(
                qre.Deallocate(allocated_register=identification_reg)
            )  # deallocate identification register
        gate_list.append(
            qre.Deallocate(allocated_register=enumeration_reg)
        )  # deallocate enumeration register

        return gate_list
